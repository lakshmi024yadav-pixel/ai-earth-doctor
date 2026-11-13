"""
db_helpers.py
--------------
WHAT: Small functions that read and write the database.
WHY:  Keeps SQL out of app.py so the routes stay easy to read.

STAGE 4 ADDITIONS: get_analysis_images(), set_analysis_status()
STAGE 5 ADDITION:  save_metrics()
STAGE 6 ADDITIONS: save_change_results(), get_latest_analysis_id()
"""

import json
import os

from database.init_db import get_connection


def save_analysis(region_name, description, is_demo, year_results, base_dir):
    """
    Save one prepared dataset.

    year_results = { 2018: {"path": ..., "original_filename": ..., "meta": {...}}, ... }

    Creates (or reuses, for the sample region) a row in regions, one new row
    in analyses (status 'pending' = data ready, analysis not run yet) and
    one row per year in images.
    Returns (region_id, analysis_id).
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()

        region_id = None
        if is_demo:
            row = cursor.execute(
                "SELECT id FROM regions WHERE is_demo = 1 AND name = ?", (region_name,)
            ).fetchone()
            if row:
                region_id = row[0]

        if region_id is None:
            cursor.execute(
                "INSERT INTO regions (name, description, is_demo) VALUES (?, ?, ?)",
                (region_name, description, 1 if is_demo else 0),
            )
            region_id = cursor.lastrowid

        years_sorted = sorted(year_results.keys())
        cursor.execute(
            "INSERT INTO analyses (region_id, years_json, status) VALUES (?, ?, 'pending')",
            (region_id, json.dumps(years_sorted)),
        )
        analysis_id = cursor.lastrowid

        for year in years_sorted:
            info = year_results[year]
            relative_path = os.path.relpath(info["path"], base_dir)
            cursor.execute(
                "INSERT INTO images (analysis_id, year, filepath, original_filename, meta_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    analysis_id,
                    year,
                    relative_path,
                    info.get("original_filename"),
                    json.dumps(info["meta"]),
                ),
            )

        conn.commit()
        return region_id, analysis_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_analysis_images(analysis_id):
    """Return [{"year": 2018, "filepath": "data/..."}, ...] for one analysis (empty list if none)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT year, filepath FROM images WHERE analysis_id = ? ORDER BY year",
            (analysis_id,),
        ).fetchall()
        return [{"year": row[0], "filepath": row[1]} for row in rows]
    finally:
        conn.close()


def set_analysis_status(analysis_id, status):
    """Update analyses.status (pending -> preprocessed -> done / failed)."""
    conn = get_connection()
    try:
        conn.execute("UPDATE analyses SET status = ? WHERE id = ?", (status, analysis_id))
        conn.commit()
    finally:
        conn.close()


def save_metrics(analysis_id, rows):
    """
    Store one row per year in the metrics table (replaces older rows of the
    same analysis, so running the analysis twice never duplicates data).

    rows = [{"year": 2018, "vegetation_km2": 7.1, "water_km2": 0.02, "urban_km2": 12.7,
             "ndvi_avg": 0.24, "ndwi_avg": -0.29, "ndbi_avg": 0.0}, ...]
    """
    conn = get_connection()
    try:
        conn.execute("DELETE FROM metrics WHERE analysis_id = ?", (analysis_id,))
        for r in rows:
            conn.execute(
                "INSERT INTO metrics (analysis_id, year, vegetation_km2, water_km2, urban_km2, "
                "ndvi_avg, ndwi_avg, ndbi_avg) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (analysis_id, r["year"], r["vegetation_km2"], r["water_km2"], r["urban_km2"],
                 r["ndvi_avg"], r["ndwi_avg"], r["ndbi_avg"]),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def save_change_results(analysis_id, rows):
    """
    Store the net change of every category for every pair of years in change_results
    (replaces older rows of the same analysis).

    rows = [{"from_year": 2018, "to_year": 2026, "category": "Vegetation",
             "change_km2": 1.02, "change_percent": 14.3}, ...]     (change_percent may be None)
    """
    conn = get_connection()
    try:
        conn.execute("DELETE FROM change_results WHERE analysis_id = ?", (analysis_id,))
        for r in rows:
            conn.execute(
                "INSERT INTO change_results (analysis_id, from_year, to_year, category, change_km2, change_percent) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (analysis_id, r["from_year"], r["to_year"], r["category"], r["change_km2"], r["change_percent"]),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_latest_analysis_id(status="done"):
    """Id of the newest analysis with this status, or None."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id FROM analyses WHERE status = ? ORDER BY id DESC LIMIT 1", (status,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()
