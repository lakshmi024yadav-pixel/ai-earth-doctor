"""
data_loader.py
---------------
WHAT:   Finds, stacks, reads and checks Sentinel-2 GeoTIFF files.
WHY:    Every later stage (indices, change detection, ML) needs the same
        thing: ONE 5-band image per year, in a known band order, already
        checked for problems. This file does that in one place.
HOW:    Uses ml/geotiff_io.py (built on tifffile, pure Python) to read GeoTIFF files.
INPUT:  a folder holding one file (or one folder of band files) per year.
OUTPUT: for each year: path of a ready 5-band GeoTIFF + metadata
        (size, coordinate system, pixel size, area, value ranges).
LIMITS: It checks files; it does not fix clouds or alignment (Stage 4).
"""

import json
import math
import os

import numpy as np

from ml import geotiff_io

# The band order every file MUST have. Do not change it casually:
# the index formulas in Stage 5 rely on this order.
BAND_ORDER = ["B02", "B03", "B04", "B08", "B11"]

BAND_INFO = {
    "B02": {"name": "Blue",  "wavelength_nm": 490,  "resolution_m": 10, "used_for": "True-colour picture"},
    "B03": {"name": "Green", "wavelength_nm": 560,  "resolution_m": 10, "used_for": "NDWI (water)"},
    "B04": {"name": "Red",   "wavelength_nm": 665,  "resolution_m": 10, "used_for": "NDVI (vegetation)"},
    "B08": {"name": "NIR (near infrared)",       "wavelength_nm": 842,  "resolution_m": 10, "used_for": "NDVI, NDWI, NDBI"},
    "B11": {"name": "SWIR (shortwave infrared)", "wavelength_nm": 1610, "resolution_m": 20, "used_for": "NDBI (built-up)"},
}

# First bytes of every real TIFF / BigTIFF file
TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


# ---------------------------------------------------------------------
# Finding and stacking files
# ---------------------------------------------------------------------
def is_tiff_file(path):
    """True if the file really starts like a TIFF (not just named .tif)."""
    try:
        with open(path, "rb") as f:
            return f.read(4) in TIFF_MAGIC
    except OSError:
        return False


def find_year_source(region_dir, year):
    """
    Look for the data of one year. Returns (kind, path):
      ("file", ".../2018.tif")   a multi-band file whose name starts with the year
      ("folder", ".../2018")     a folder of single-band files
      (None, None)               nothing found
    """
    prefix = str(year)
    for name in sorted(os.listdir(region_dir)):
        full = os.path.join(region_dir, name)
        if os.path.isfile(full) and name.startswith(prefix) and name.lower().endswith((".tif", ".tiff")):
            return "file", full
    folder = os.path.join(region_dir, prefix)
    if os.path.isdir(folder):
        return "folder", folder
    return None, None


def merge_band_files(folder, out_path):
    """
    Stack single-band files (names contain B02, B03, B04, B08, B11)
    into ONE 5-band GeoTIFF in BAND_ORDER. Skips the work if the output
    is already up to date.
    """
    band_files = {}
    for name in sorted(os.listdir(folder)):
        lower = name.lower()
        if not lower.endswith((".tif", ".tiff")):
            continue
        for band in BAND_ORDER:
            if band.lower() in lower and band not in band_files:
                band_files[band] = os.path.join(folder, name)

    missing = [b for b in BAND_ORDER if b not in band_files]
    if missing:
        raise ValueError(
            "folder '%s' has no file for band(s): %s" % (os.path.basename(folder), ", ".join(missing))
        )

    if os.path.exists(out_path) and all(
        os.path.getmtime(p) <= os.path.getmtime(out_path) for p in band_files.values()
    ):
        return  # already merged and nothing changed

    layers, first = [], None
    for band in BAND_ORDER:
        raster = geotiff_io.read_geotiff(band_files[band])
        name = os.path.basename(band_files[band])
        if raster["data"].shape[0] != 1:
            raise ValueError("%s has %d bands; a band file must have exactly 1" % (name, raster["data"].shape[0]))
        if raster["transform"] is None or raster["crs"] is None:
            raise ValueError("%s is not georeferenced (it has no coordinates)" % name)
        if first is None:
            first = raster
        elif (raster["data"].shape != first["data"].shape or raster["crs"] != first["crs"]
              or raster["transform"] != first["transform"]):
            raise ValueError(
                "band %s does not cover exactly the same area/size as B02. "
                "Export all bands with the same area box and resolution." % band
            )
        layers.append(raster["data"][0].astype("float32"))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        geotiff_io.write_geotiff(out_path, np.stack(layers), first["transform"], first["crs"], first["nodata"])
    except Exception:
        if os.path.exists(out_path):
            os.remove(out_path)
        raise


# ---------------------------------------------------------------------
# Reading metadata
# ---------------------------------------------------------------------
def inspect_raster(path):
    """
    Read a GeoTIFF's metadata and simple statistics. To stay fast, the
    statistics use a reduced copy of the image (max 512 pixels on the long side).
    """
    raster = geotiff_io.read_geotiff(path)
    data, crs, transform = raster["data"], raster["crs"], raster["transform"]
    bands, height, width = data.shape
    nodata = raster["nodata"]
    if nodata is not None and not math.isfinite(nodata):
        nodata = None

    meta = {
        "file": os.path.basename(path),
        "file_size_mb": round(os.path.getsize(path) / 1048576, 2),
        "width": int(width),
        "height": int(height),
        "bands": int(bands),
        "dtype": str(data.dtype),
        "crs": crs.to_string() if crs else None,
        "nodata": nodata,
        "bounds_latlon": None,        # [[south, west], [north, east]] for the map
        "pixel_width_m": None,
        "pixel_height_m": None,
        "pixel_area_km2": None,
        "area_km2": None,
    }

    if crs is not None and transform is not None:
        try:
            south, west, north, east = geotiff_io.bounds_latlon(transform, crs, width, height)
        except ValueError:
            pass                      # unsupported coordinate system: validate_meta reports it
        else:
            meta["bounds_latlon"] = [
                [round(float(south), 6), round(float(west), 6)],
                [round(float(north), 6), round(float(east), 6)],
            ]
            px, py = geotiff_io.pixel_size_m(crs, transform, (south + north) / 2)
            if px and py:
                meta["pixel_width_m"] = round(float(px), 2)
                meta["pixel_height_m"] = round(float(py), 2)
                meta["pixel_area_km2"] = float(px * py) / 1e6
                meta["area_km2"] = round(meta["pixel_area_km2"] * width * height, 2)

    # Statistics from a reduced copy of the image
    arr = geotiff_io.shrink(data.astype("float32"), 512).copy()
    if nodata is not None:
        arr[arr == nodata] = np.nan

    # A pixel is "valid" if every band is a real number and not all bands are 0
    valid = np.isfinite(arr).all(axis=0) & (arr != 0).any(axis=0)
    meta["valid_pixel_percent"] = round(float(valid.mean() * 100), 1)

    stats = []
    for b in range(bands):
        values = arr[b][valid]
        if values.size == 0:
            stats.append({"min": None, "max": None, "mean": None})
        else:
            stats.append({
                "min": round(float(values.min()), 4),
                "max": round(float(values.max()), 4),
                "mean": round(float(values.mean()), 4),
            })
    meta["band_stats"] = stats

    # Are values reflectance (0-1) or integers scaled by 10000? Stage 4 uses this.
    meta["value_scale"] = "unknown"
    if valid.any():
        p99 = float(np.percentile(arr[:, valid], 99))
        if p99 <= 2.0:
            meta["value_scale"] = "reflectance (0-1)"
        elif p99 <= 20000:
            meta["value_scale"] = "scaled integers (0-10000)"

    return meta


# ---------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------
def validate_meta(meta):
    """Return a list of PROBLEMS that stop us from using this file."""
    problems = []
    if meta["bands"] != len(BAND_ORDER):
        problems.append(
            "the file has %d band(s) but exactly 5 are needed, in this order: "
            "B02 (Blue), B03 (Green), B04 (Red), B08 (NIR), B11 (SWIR)." % meta["bands"]
        )
    if meta["crs"] is None:
        problems.append("the file has no coordinate system (it is not georeferenced), so areas in km2 cannot be calculated.")
    elif meta["pixel_width_m"] is None:
        problems.append("the coordinate system (%s) is not supported. Export with WGS84 (EPSG:4326) or UTM." % meta["crs"])
    if meta["valid_pixel_percent"] == 0:
        problems.append("the file contains no valid pixel data (all zero or empty).")
    return problems


def meta_warnings(meta):
    """Return WARNINGS: the file is usable, but the user should know."""
    warnings = []
    if 0 < meta["valid_pixel_percent"] < 90:
        warnings.append(
            "only %.0f%% of the pixels contain data. The area box may extend outside the satellite scene."
            % meta["valid_pixel_percent"]
        )
    if meta["value_scale"] == "unknown" and meta["valid_pixel_percent"] > 0:
        warnings.append("the pixel values look unusual (not reflectance 0-1 and not 0-10000).")
    return warnings


def _overlap_fraction(a, b):
    """How much of the smaller box is covered by the other (0 to 1)."""
    south = max(a[0][0], b[0][0])
    north = min(a[1][0], b[1][0])
    west = max(a[0][1], b[0][1])
    east = min(a[1][1], b[1][1])
    if north <= south or east <= west:
        return 0.0
    inter = (north - south) * (east - west)
    area_a = (a[1][0] - a[0][0]) * (a[1][1] - a[0][1])
    area_b = (b[1][0] - b[0][0]) * (b[1][1] - b[0][1])
    return inter / min(area_a, area_b)


def check_alignment(metas):
    """
    Compare every year with the earliest year.
    Returns (problems, warnings). Different sizes are only warnings,
    because Stage 4 resizes and aligns images. Different AREAS are a problem.
    """
    problems, warnings = [], []
    years = sorted(metas)
    if len(years) < 2:
        return problems, warnings

    base_year = years[0]
    base = metas[base_year]
    for year in years[1:]:
        m = metas[year]
        if m["bounds_latlon"] and base["bounds_latlon"]:
            overlap = _overlap_fraction(base["bounds_latlon"], m["bounds_latlon"])
            if overlap < 0.5:
                problems.append(
                    "%d and %d cover mostly different places (overlap %.0f%%). Use the SAME area box for every year."
                    % (year, base_year, overlap * 100)
                )
            elif overlap < 0.95:
                warnings.append(
                    "%d and %d overlap by only %.0f%%; only the common area can be compared."
                    % (year, base_year, overlap * 100)
                )
        if (m["width"], m["height"]) != (base["width"], base["height"]):
            warnings.append(
                "%d is %d x %d pixels but %d is %d x %d. They will be aligned during preprocessing (Stage 4)."
                % (year, m["width"], m["height"], base_year, base["width"], base["height"])
            )
        if m["crs"] != base["crs"]:
            warnings.append(
                "%d uses coordinate system %s but %d uses %s (handled in Stage 4)."
                % (year, m["crs"], base_year, base["crs"])
            )
    return problems, warnings


# ---------------------------------------------------------------------
# Putting it together
# ---------------------------------------------------------------------
def read_region_info(region_dir):
    """Read the optional region.json (description + scene dates). {} if missing or broken."""
    path = os.path.join(region_dir, "region.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            info = json.load(f)
        return info if isinstance(info, dict) else {}
    except (OSError, ValueError):
        return {}


def prepare_region(region_dir, years, processed_dir):
    """
    Find, merge (if needed), read and check the data for every year.

    Returns {"years": {year: {"path", "source_kind", "meta"}},
             "problems": [...],   # must be fixed
             "warnings": [...]}   # good to know
    """
    result = {"years": {}, "problems": [], "warnings": []}

    if not os.path.isdir(region_dir):
        result["problems"].append("Dataset folder not found: %s" % region_dir)
        return result

    metas = {}
    for year in years:
        kind, source = find_year_source(region_dir, year)
        if kind is None:
            result["problems"].append(
                "%d: no Sentinel-2 data found. Put a 5-band GeoTIFF named %d.tif, or a folder named %d "
                "containing the band files, inside %s" % (year, year, year, region_dir)
            )
            continue

        try:
            if kind == "folder":
                path = os.path.join(processed_dir, "%d.tif" % year)
                merge_band_files(source, path)
            else:
                path = source
            if not is_tiff_file(path):
                result["problems"].append("%d: '%s' is not a valid TIFF file." % (year, os.path.basename(path)))
                continue
            try:
                meta = inspect_raster(path)
            except ValueError:
                if kind != "folder":
                    raise
                os.remove(path)               # a merged file made by an older version: build it again once
                merge_band_files(source, path)
                meta = inspect_raster(path)
        except Exception as error:
            result["problems"].append("%d: could not read the data (%s)" % (year, error))
            continue

        for problem in validate_meta(meta):
            result["problems"].append("%d: %s" % (year, problem))
        for warning in meta_warnings(meta):
            result["warnings"].append("%d: %s" % (year, warning))

        result["years"][year] = {"path": path, "source_kind": kind, "meta": meta}
        metas[year] = meta

    align_problems, align_warnings = check_alignment(metas)
    result["problems"].extend(align_problems)
    result["warnings"].extend(align_warnings)
    return result
