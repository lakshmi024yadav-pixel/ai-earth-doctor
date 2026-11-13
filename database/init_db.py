"""
init_db.py
-----------
WHAT: Creates the SQLite database and all tables for AI Earth Doctor.
WHY:  The full schema exists from the start, so table names never change.
      Re-running is safe: it never deletes data.
RUN (project root):  python database/init_db.py

STAGE 3 CHANGE: the images table gets a meta_json column (stores the
file's size, coordinate system, pixel size, etc.). ensure_columns()
adds it to a database created in Stage 1 without losing anything.
"""

import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.db")


def get_connection():
    """Open a connection to the SQLite database file."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'normal',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS regions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    is_demo INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    region_id INTEGER NOT NULL,
    years_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (region_id) REFERENCES regions(id)
);

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    year INTEGER NOT NULL,
    filepath TEXT NOT NULL,
    original_filename TEXT,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    meta_json TEXT,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    year INTEGER NOT NULL,
    vegetation_km2 REAL,
    water_km2 REAL,
    urban_km2 REAL,
    ndvi_avg REAL,
    ndwi_avg REAL,
    ndbi_avg REAL,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
);

CREATE TABLE IF NOT EXISTS change_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    from_year INTEGER NOT NULL,
    to_year INTEGER NOT NULL,
    category TEXT NOT NULL,
    change_km2 REAL,
    change_percent REAL,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
);

CREATE TABLE IF NOT EXISTS risk_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    risk_type TEXT NOT NULL,
    confidence REAL,
    description TEXT,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
);

CREATE TABLE IF NOT EXISTS suitability_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    zone_name TEXT NOT NULL,
    score REAL,
    category TEXT,
    environmental_sensitivity TEXT,
    flood_risk TEXT,
    vegetation_impact TEXT,
    infrastructure_score TEXT,
    population_demand TEXT,
    reasoning TEXT,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
);

CREATE TABLE IF NOT EXISTS recovery_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    zone_name TEXT NOT NULL,
    restoration_score REAL,
    recommendation TEXT,
    reasoning TEXT,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
);

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER NOT NULL,
    filepath TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
);
"""


def ensure_columns(conn):
    """
    Add columns that were introduced after Stage 1 to an existing database.
    CREATE TABLE IF NOT EXISTS does not change a table that already exists,
    so a database made in Stage 1 needs this small upgrade.
    """
    image_columns = [row[1] for row in conn.execute("PRAGMA table_info(images)")]
    if "meta_json" not in image_columns:
        conn.execute("ALTER TABLE images ADD COLUMN meta_json TEXT")


def init_db(verbose=True):
    """Create all tables, upgrade older ones, optionally print what exists."""
    conn = get_connection()
    conn.executescript(SCHEMA)
    ensure_columns(conn)
    conn.commit()

    if verbose:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
        )
        tables = [row[0] for row in cursor.fetchall()]
        print("Database ready at:", DB_PATH)
        print("Tables:", tables)
    conn.close()


if __name__ == "__main__":
    init_db()
