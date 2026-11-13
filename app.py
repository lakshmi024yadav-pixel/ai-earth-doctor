"""
app.py
-------
WHAT: The main Flask application for AI Earth Doctor.
WHY:  Flask connects web pages (HTML) to our Python code.
RUN:  python app.py   then open http://127.0.0.1:5000

STAGE 3 ADDITIONS:
  POST /api/prepare/sample   read + check the Sentinel-2 files in data/sample/demo_region
  POST /api/prepare/upload   receive 3 uploaded GeoTIFFs, check them, save them
  Both save records in SQLite and return the dataset summary as JSON.

STAGE 4 ADDITION:
  POST /api/preprocess/<analysis_id>   clean the images (normalise, align,
  cloud mask, optional noise filter) and return the preprocessing report.

STAGE 5 ADDITION:
  POST /api/indices/<analysis_id>      NDVI, NDWI, NDBI + rule-based land cover,
  saved in the metrics table and returned for the dashboard.

STAGE 7 ADDITIONS:
  GET  /api/ml/model                   is a trained classifier available? (and how it was trained)
  POST /api/ml/<analysis_id>           classify the years with the trained model + compare with the rules
  POST /api/change/<analysis_id>       now also accepts {"method": "rules" | "ml"}

STAGE 6 ADDITIONS:
  POST /api/change/<analysis_id>       compare the years (area gained/lost, % change,
                                       category changes) and save change_results
  GET  /api/change/<analysis_id>       return the saved change report
  /change-detection                    the Change Detection page (before/after slider, maps)
"""

import json
import os
import re
import shutil
import sqlite3
import uuid

from flask import Flask, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

import config
from database.db_helpers import (get_analysis_images, get_latest_analysis_id, save_analysis,
                                 save_change_results, save_metrics, set_analysis_status)
from database.init_db import init_db
from ml import change_detection as change_engine   # (not named change_detection: that is a page function below)
from ml import data_loader, indices, land_classification, preprocessing

# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------
app = Flask(__name__)
# Reject huge uploads early (3 files + a little extra for the form fields)
app.config["MAX_CONTENT_LENGTH"] = (config.MAX_FILE_MB * 3 + 5) * 1024 * 1024

DB_PATH = os.path.join(config.BASE_DIR, "database", "database.db")

# Make sure the folders and the database exist every time the app starts
os.makedirs(config.UPLOAD_DIR, exist_ok=True)
os.makedirs(config.PROCESSED_DIR, exist_ok=True)
os.makedirs(config.SAMPLE_REGION_DIR, exist_ok=True)
os.makedirs(config.PREVIEW_DIR, exist_ok=True)
init_db(verbose=False)

NAV_ITEMS = [
    {"endpoint": "dashboard",        "label": "Dashboard",           "icon": "🏠"},
    {"endpoint": "earth_health",     "label": "Earth Health",        "icon": "💚"},
    {"endpoint": "change_detection", "label": "Change Detection",    "icon": "🛰️"},
    {"endpoint": "disaster_watch",   "label": "Disaster Watch",      "icon": "🚨"},
    {"endpoint": "planner",          "label": "Sustainable Planner", "icon": "🏗️"},
    {"endpoint": "recovery",         "label": "Earth Recovery",      "icon": "🌱"},
    {"endpoint": "reports",          "label": "Reports",             "icon": "📄"},
    {"endpoint": "doctor",           "label": "Ask Earth Doctor",    "icon": "🩺"},
]


@app.context_processor
def inject_nav():
    """Make NAV_ITEMS available in every template."""
    return {"nav_items": NAV_ITEMS}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def check_database_ready():
    """Return (database_exists, list_of_table_names) for the status page."""
    if not os.path.exists(DB_PATH):
        return False, []
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
    )
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return True, tables


def coming_soon(title, icon, stage, description):
    """Page for modules that are built in a later stage."""
    return render_template(
        "coming_soon.html", page_title=title, icon=icon, stage=stage, description=description
    )


def api_error(message, code=400, **extra):
    """Standard JSON error reply used by every /api route."""
    return jsonify(ok=False, error=message, **extra), code


def build_response(mode, region_name, region_id, analysis_id, result, scenes):
    """Turn the loader result into the JSON the dashboard displays."""
    items = []
    for year in sorted(result["years"]):
        info = result["years"][year]
        scene = scenes.get(str(year), {}) if isinstance(scenes, dict) else {}
        if not isinstance(scene, dict):
            scene = {}
        items.append({
            "year": year,
            "source_kind": info["source_kind"],
            "original_filename": info.get("original_filename", os.path.basename(info["path"])),
            "meta": info["meta"],
            "scene": {k: v for k, v in scene.items() if v},   # only filled-in fields
        })
    return jsonify(
        ok=True,
        mode=mode,
        region_name=region_name,
        region_id=region_id,
        analysis_id=analysis_id,
        years=items,
        warnings=result["warnings"],
        band_order=data_loader.BAND_ORDER,
        band_info=data_loader.BAND_INFO,
    )


def add_change_urls(report, analysis_id):
    """Add the picture addresses (static files) to a change report."""
    folder = "previews/analysis_%d/" % analysis_id
    report["analysis_id"] = analysis_id
    report["year_pictures"] = {
        str(year): {key: url_for("static", filename=folder + "%s_%d.png" % (key, year))
                    for key in ("rgb", "ndvi", "ndwi", "ndbi", "landcover")}
        for year in report["years"]
    }
    for pair in report["pairs"]:
        a, b = pair["from_year"], pair["to_year"]
        pair["pictures"] = {
            "change": url_for("static", filename=folder + "change_%d_%d.png" % (a, b)),
            "ndvi_difference": url_for("static", filename=folder + "ndvidiff_%d_%d.png" % (a, b)),
        }
    return report


# ---------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------
@app.route("/")
def home():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/status")
def status():
    db_exists, tables = check_database_ready()
    return render_template("index.html", db_exists=db_exists, tables=tables, table_count=len(tables))


@app.route("/earth-health")
def earth_health():
    return coming_soon(
        "Earth Health", "💚", "Stage 14",
        "One page that brings the indicators, risks and the Earth Health score together. "
        "The indices and change numbers already appear on the dashboard.",
    )


@app.route("/change-detection")
def change_detection():
    """Change Detection page. ?analysis=<id> chooses the analysis, otherwise the newest finished one."""
    analysis_id = request.args.get("analysis", type=int) or get_latest_analysis_id("done")
    return render_template("change_detection.html", analysis_id=analysis_id)


@app.route("/disaster-watch")
def disaster_watch():
    return coming_soon(
        "Disaster Watch", "🚨", "Stage 8",
        "Possible flood-affected areas, vegetation stress and burn-affected land, shown with "
        "confidence values. These are indicators, not predictions.",
    )


@app.route("/planner")
def planner():
    return coming_soon(
        "Sustainable Land Planner", "🏗️", "Stages 9 and 10",
        "An optional decision-support tool: explainable land suitability scores and development "
        "scenarios. It is never a construction approval.",
    )


@app.route("/recovery")
def recovery():
    return coming_soon(
        "Earth Recovery", "🌱", "Stage 11",
        "Areas where restoration (revegetation, green corridors, water-body restoration) may be beneficial.",
    )


@app.route("/reports")
def reports():
    return coming_soon(
        "Reports", "📄", "Stage 13",
        "Download a report with metrics, risks, explanations, recommendations and limitations.",
    )


@app.route("/doctor")
def doctor():
    return coming_soon(
        "Ask Earth Doctor", "🩺", "Stage 12",
        "Ask questions about your analysis. Answers will be generated only from the actual detected results.",
    )


# ---------------------------------------------------------------------
# API: sample region
# ---------------------------------------------------------------------
@app.route("/api/prepare/sample", methods=["POST"])
def api_prepare_sample():
    """Read and check the Sentinel-2 files in data/sample/demo_region."""
    try:
        processed = os.path.join(config.PROCESSED_DIR, "demo_region")
        result = data_loader.prepare_region(config.SAMPLE_REGION_DIR, config.SAMPLE_YEARS, processed)
        if result["problems"]:
            return api_error(
                "The sample dataset is not ready. Fix the points below and try again.",
                problems=result["problems"], warnings=result["warnings"],
            )

        info = data_loader.read_region_info(config.SAMPLE_REGION_DIR)
        description = info.get("description") or "Sentinel-2 scenes supplied by the student."
        region_id, analysis_id = save_analysis(
            config.SAMPLE_REGION_NAME, description, True, result["years"], config.BASE_DIR
        )
        return build_response(
            "sample", config.SAMPLE_REGION_NAME, region_id, analysis_id, result, info.get("scenes", {})
        )
    except Exception:
        app.logger.exception("Sample dataset preparation failed")
        return api_error("Something went wrong while reading the sample dataset. See the terminal for details.", 500)


# ---------------------------------------------------------------------
# API: user upload
# ---------------------------------------------------------------------
@app.route("/api/prepare/upload", methods=["POST"])
def api_prepare_upload():
    """Receive three GeoTIFFs, validate them, save them, store records."""
    upload_folder = None
    saved_ok = False
    try:
        # ---- 1. check the form fields ----
        region_name = (request.form.get("region_name") or "").strip()
        if not region_name:
            return api_error("Please enter a region name.")
        if len(region_name) > 60 or not re.fullmatch(r"[\w .,()\-]+", region_name):
            return api_error("Region name may use letters, numbers, spaces and . , ( ) - _ only (max 60 characters).")

        slots = []
        for i in (1, 2, 3):
            try:
                year = int(request.form.get("year%d" % i, ""))
            except ValueError:
                return api_error("Slot %d: please enter a valid year." % i)
            if year < config.MIN_YEAR or year > config.MAX_YEAR:
                return api_error("Slot %d: the year must be between %d and %d." % (i, config.MIN_YEAR, config.MAX_YEAR))
            file = request.files.get("file%d" % i)
            if file is None or file.filename == "":
                return api_error("Slot %d (%d): please choose a GeoTIFF file." % (i, year))
            extension = os.path.splitext(file.filename)[1].lower()
            if extension not in config.ALLOWED_EXTENSIONS:
                return api_error("Slot %d (%d): only .tif or .tiff files are accepted." % (i, year))
            slots.append((year, file))

        years = sorted(year for year, _ in slots)
        if len(set(years)) != 3:
            return api_error("Each slot needs a different year.")

        # ---- 2. save files into a new private folder (our own file names) ----
        upload_folder = os.path.join(config.UPLOAD_DIR, uuid.uuid4().hex[:12])
        os.makedirs(upload_folder)
        originals = {}
        for year, file in slots:
            target = os.path.join(upload_folder, "%d.tif" % year)
            file.save(target)
            originals[year] = secure_filename(file.filename) or ("%d.tif" % year)
            size = os.path.getsize(target)
            if size == 0:
                return api_error("The file for %d is empty." % year)
            if size > config.MAX_FILE_MB * 1048576:
                return api_error("The file for %d is larger than %d MB. Crop a smaller area." % (year, config.MAX_FILE_MB))
            if not data_loader.is_tiff_file(target):
                return api_error("The file for %d is not a real TIFF image (was the extension renamed?)." % year)

        # ---- 3. read and check the rasters ----
        processed = os.path.join(config.PROCESSED_DIR, os.path.basename(upload_folder))
        result = data_loader.prepare_region(upload_folder, years, processed)
        if result["problems"]:
            return api_error(
                "Your files could not be used. Fix the points below and try again.",
                problems=result["problems"], warnings=result["warnings"],
            )
        for year in result["years"]:
            result["years"][year]["original_filename"] = originals[year]

        # ---- 4. store records ----
        region_id, analysis_id = save_analysis(
            region_name, "Region uploaded by the user.", False, result["years"], config.BASE_DIR
        )
        saved_ok = True
        return build_response("upload", region_name, region_id, analysis_id, result, {})

    except Exception:
        app.logger.exception("Upload preparation failed")
        return api_error("Something went wrong while processing your files. See the terminal for details.", 500)
    finally:
        # If anything failed, do not leave half-saved files behind
        if upload_folder and not saved_ok:
            shutil.rmtree(upload_folder, ignore_errors=True)


# ---------------------------------------------------------------------
# API: preprocessing (Stage 4)
# ---------------------------------------------------------------------
@app.route("/api/preprocess/<int:analysis_id>", methods=["POST"])
def api_preprocess(analysis_id):
    """Clean the images of one analysis and return the preprocessing report."""
    try:
        options = request.get_json(silent=True) or {}
        mask_clouds = bool(options.get("mask_clouds", True))
        denoise_on = bool(options.get("denoise", False))

        images = get_analysis_images(analysis_id)
        if not images:
            return api_error("Analysis #%d was not found. Click Analyze Region again." % analysis_id, 404)

        year_paths = {}
        for row in images:
            path = os.path.normpath(os.path.join(config.BASE_DIR, row["filepath"]))
            if not path.startswith(config.DATA_DIR + os.sep):
                return api_error("The stored file path for %d is outside the data folder." % row["year"])
            if not os.path.exists(path):
                return api_error("The image file for %d is missing. Click Analyze Region again." % row["year"])
            year_paths[row["year"]] = path

        out_dir = os.path.join(config.PROCESSED_DIR, "analysis_%d" % analysis_id)
        preview_dir = os.path.join(config.PREVIEW_DIR, "analysis_%d" % analysis_id)
        report = preprocessing.run_pipeline(year_paths, out_dir, preview_dir, mask_clouds, denoise_on)

        for item in report["years"]:
            folder = "previews/analysis_%d/" % analysis_id
            item["rgb_url"] = url_for("static", filename=folder + "rgb_%d.png" % item["year"])
            item["mask_url"] = url_for("static", filename=folder + "mask_%d.png" % item["year"])

        set_analysis_status(analysis_id, "preprocessed")
        return jsonify(ok=True, analysis_id=analysis_id, **report)

    except ValueError as error:
        # Problems we expect (no common area, only one year, empty image...)
        return api_error("Preprocessing could not finish: %s" % error)
    except Exception:
        app.logger.exception("Preprocessing failed")
        return api_error("Something went wrong during preprocessing. See the terminal for details.", 500)


# ---------------------------------------------------------------------
# API: environmental indices (Stage 5)
# ---------------------------------------------------------------------
@app.route("/api/indices/<int:analysis_id>", methods=["POST"])
def api_indices(analysis_id):
    """Calculate NDVI, NDWI, NDBI and the rule-based land cover for one analysis."""
    try:
        if not get_analysis_images(analysis_id):
            return api_error("Analysis #%d was not found. Click Analyze Region again." % analysis_id, 404)

        out_dir = os.path.join(config.PROCESSED_DIR, "analysis_%d" % analysis_id)
        preview_dir = os.path.join(config.PREVIEW_DIR, "analysis_%d" % analysis_id)
        report = indices.run_indices(out_dir, preview_dir)

        rows = []
        for item in report["years"]:
            folder = "previews/analysis_%d/" % analysis_id
            item["pictures"] = {
                key: url_for("static", filename=folder + "%s_%d.png" % (key, item["year"]))
                for key in ("rgb", "ndvi", "ndwi", "ndbi", "landcover")
            }
            rows.append({
                "year": item["year"],
                "vegetation_km2": item["class_km2"]["Vegetation"],
                "water_km2": item["class_km2"]["Water"],
                "urban_km2": item["class_km2"]["Built-up"],
                "ndvi_avg": item["mean_ndvi"],
                "ndwi_avg": item["mean_ndwi"],
                "ndbi_avg": item["mean_ndbi"],
            })
        save_metrics(analysis_id, rows)
        set_analysis_status(analysis_id, "indexed")
        return jsonify(ok=True, analysis_id=analysis_id, **report)

    except ValueError as error:
        return api_error("Index calculation could not finish: %s" % error)
    except Exception:
        app.logger.exception("Index calculation failed")
        return api_error("Something went wrong while calculating the indices. See the terminal for details.", 500)


# ---------------------------------------------------------------------
# API: machine-learning classification (Stage 7)
# ---------------------------------------------------------------------
@app.route("/api/ml/model", methods=["GET"])
def api_ml_model():
    """Tell the dashboard whether a trained model exists, and how it was trained."""
    info = land_classification.model_info()
    if info is None:
        return jsonify(ok=True, trained=False,
                       message="Training required: run  python train_model.py  to train the model.")
    summary = {key: info.get(key) for key in ("kind", "kind_label", "trained_at", "trained_on", "classes",
                                              "excluded_classes", "caveat")}
    summary["holdout_agreement"] = info["holdout"]["accuracy"]
    return jsonify(ok=True, trained=True, info=summary)


@app.route("/api/ml/<int:analysis_id>", methods=["POST"])
def api_ml_run(analysis_id):
    """Classify the years of one analysis with the trained model and compare with the rule-based map."""
    try:
        if not get_analysis_images(analysis_id):
            return api_error("Analysis #%d was not found. Click Analyze Region again." % analysis_id, 404)

        out_dir = os.path.join(config.PROCESSED_DIR, "analysis_%d" % analysis_id)
        preview_dir = os.path.join(config.PREVIEW_DIR, "analysis_%d" % analysis_id)
        report = land_classification.run_ml_classification(out_dir, preview_dir)

        folder = "previews/analysis_%d/" % analysis_id
        for item in report["years"]:
            item["pictures"] = {
                "rule": url_for("static", filename=folder + "landcover_%d.png" % item["year"]),
                "ml": url_for("static", filename=folder + "mlcover_%d.png" % item["year"]),
                "confidence": url_for("static", filename=folder + "mlconf_%d.png" % item["year"]),
            }
        return jsonify(ok=True, analysis_id=analysis_id, **report)

    except ValueError as error:
        text = str(error)
        missing = "no trained model" in text or "could not be loaded" in text
        return api_error("Machine-learning classification could not finish: %s" % text, model_missing=missing)
    except Exception:
        app.logger.exception("Machine-learning classification failed")
        return api_error("Something went wrong during machine-learning classification. See the terminal for details.", 500)


# ---------------------------------------------------------------------
# API: change detection (Stage 6)
# ---------------------------------------------------------------------
@app.route("/api/change/<int:analysis_id>", methods=["POST"])
def api_change_run(analysis_id):
    """Compare the years of one analysis and save the results."""
    try:
        if not get_analysis_images(analysis_id):
            return api_error("Analysis #%d was not found. Click Analyze Region again." % analysis_id, 404)

        options = request.get_json(silent=True) or {}
        method = options.get("method", "rules")
        if method not in ("rules", "ml"):
            return api_error("Unknown classification method. Use rules or ml.")

        out_dir = os.path.join(config.PROCESSED_DIR, "analysis_%d" % analysis_id)
        preview_dir = os.path.join(config.PREVIEW_DIR, "analysis_%d" % analysis_id)
        report = change_engine.run_change(out_dir, preview_dir, method)

        rows = []
        for pair in report["pairs"]:
            for category in pair["categories"]:
                rows.append({
                    "from_year": pair["from_year"], "to_year": pair["to_year"],
                    "category": category["name"],
                    "change_km2": category["net_km2"], "change_percent": category["net_percent"],
                })
        save_change_results(analysis_id, rows)
        set_analysis_status(analysis_id, "done")
        return jsonify(ok=True, **add_change_urls(report, analysis_id))

    except ValueError as error:
        return api_error("Change detection could not finish: %s" % error)
    except Exception:
        app.logger.exception("Change detection failed")
        return api_error("Something went wrong during change detection. See the terminal for details.", 500)


@app.route("/api/change/<int:analysis_id>", methods=["GET"])
def api_change_get(analysis_id):
    """Return the saved change report of an analysis (used by the Change Detection page)."""
    path = os.path.join(config.PROCESSED_DIR, "analysis_%d" % analysis_id, "change_report.json")
    if not os.path.exists(path):
        return api_error("There are no change results for analysis #%d yet. "
                         "Go to the dashboard and click Analyze Region." % analysis_id, 404)
    try:
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
        return jsonify(ok=True, **add_change_urls(report, analysis_id))
    except (OSError, ValueError):
        app.logger.exception("Could not read the change report")
        return api_error("The saved change report could not be read. Run Analyze Region again.", 500)


# ---------------------------------------------------------------------
# Error pages
# ---------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return "Page not found. Go back to the <a href='/'>dashboard</a>.", 404


@app.errorhandler(413)
def too_large(e):
    return api_error(
        "The upload is too large. Each file must be under %d MB. Crop a smaller area and try again." % config.MAX_FILE_MB,
        413,
    )


@app.errorhandler(500)
def server_error(e):
    return (
        "Something went wrong on the server. Check the terminal for details, then go back to the "
        "<a href='/'>dashboard</a>.",
        500,
    )


if __name__ == "__main__":
    app.run(debug=True)
