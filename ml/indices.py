"""
indices.py
-----------
WHAT:   Turns the 5 cleaned Sentinel-2 bands into three environmental indices
        and a simple rule-based land-cover map.
WHY:    Raw bands are hard to read. An index combines two bands so that ONE
        kind of surface stands out:
          NDVI  vegetation   = (NIR - Red)   / (NIR + Red)
          NDWI  water        = (Green - NIR) / (Green + NIR)
          NDBI  built-up     = (SWIR - NIR)  / (SWIR + NIR)
        Each value is between -1 and +1.
HOW:    Plain NumPy arithmetic on the clean arrays from Stage 4 (no model).
INPUT:  data/processed/analysis_<id>/clean_<year>.npz, common_mask.npy, grid.json
OUTPUT: indices_<year>.npz  (ndvi, ndwi, ndbi, landcover) + colour pictures
        + a report with mean index values and area (km2) of each land class.
LIMITS: - Indices are estimates, not ground truth.
        - Thresholds (config.py) are common textbook values; they can be wrong
          for a specific place or season.
        - NDBI also reacts to bare soil, so "built-up" can include dry bare land.
        - B11 (SWIR) is natively 20 m, so NDBI is a little blurrier than NDVI.
        - Water that is small, shaded or muddy may be missed.
"""

import json
import os

import cv2
import numpy as np

import config
from ml.data_loader import BAND_ORDER

BAND_INDEX = {band: i for i, band in enumerate(BAND_ORDER)}

# Land-cover codes stored in the landcover array
CLASS_NODATA, CLASS_WATER, CLASS_VEGETATION, CLASS_BUILTUP, CLASS_BARE = 0, 1, 2, 3, 4
CLASS_NAMES = {
    CLASS_WATER: "Water",
    CLASS_VEGETATION: "Vegetation",
    CLASS_BUILTUP: "Built-up",
    CLASS_BARE: "Bare / other",
}
CLASS_COLOURS = {   # RGB
    CLASS_NODATA: (110, 110, 110),
    CLASS_WATER: (30, 96, 145),
    CLASS_VEGETATION: (46, 158, 91),
    CLASS_BUILTUP: (224, 138, 30),
    CLASS_BARE: (154, 165, 160),
}

# Colour scales used for the pictures: (value, (R, G, B))
COLOUR_SCALES = {
    "ndvi": {"title": "NDVI (vegetation)", "stops": [
        (-0.1, (140, 90, 50)), (0.1, (220, 200, 150)), (0.3, (200, 220, 90)),
        (0.5, (80, 170, 60)), (0.8, (10, 90, 30))]},
    "ndwi": {"title": "NDWI (water)", "stops": [
        (-0.6, (150, 110, 70)), (-0.3, (230, 215, 170)), (0.0, (160, 210, 230)),
        (0.4, (10, 60, 170))]},
    "ndbi": {"title": "NDBI (built-up)", "stops": [
        (-0.5, (30, 110, 40)), (-0.2, (190, 210, 130)), (0.0, (240, 230, 200)),
        (0.2, (230, 150, 70)), (0.4, (170, 30, 30))]},
}

FORMULAS = [
    {"key": "ndvi", "name": "NDVI", "formula": "(NIR − Red) / (NIR + Red)", "bands": "B08 and B04",
     "meaning": "Healthy plants reflect lots of near-infrared light and absorb red light, so high NDVI = more or greener vegetation."},
    {"key": "ndwi", "name": "NDWI", "formula": "(Green − NIR) / (Green + NIR)", "bands": "B03 and B08",
     "meaning": "Water reflects green light but absorbs near-infrared, so NDWI above 0 suggests open water."},
    {"key": "ndbi", "name": "NDBI", "formula": "(SWIR − NIR) / (SWIR + NIR)", "bands": "B11 and B08",
     "meaning": "Concrete, asphalt and bare soil reflect more shortwave infrared than near-infrared, so high NDBI suggests built-up or bare land."},
]


# ---------------------------------------------------------------------
# The three indices
# ---------------------------------------------------------------------
def normalized_difference(a, b):
    """(a - b) / (a + b). Where a + b is 0 the result is NaN (not a number)."""
    total = a + b
    result = np.full(a.shape, np.nan, dtype="float32")
    np.divide(a - b, total, out=result, where=(total != 0))
    return result


def calc_ndvi(bands):
    return normalized_difference(bands[BAND_INDEX["B08"]], bands[BAND_INDEX["B04"]])   # (NIR - Red)/(NIR + Red)


def calc_ndwi(bands):
    return normalized_difference(bands[BAND_INDEX["B03"]], bands[BAND_INDEX["B08"]])   # (Green - NIR)/(Green + NIR)


def calc_ndbi(bands):
    return normalized_difference(bands[BAND_INDEX["B11"]], bands[BAND_INDEX["B08"]])   # (SWIR - NIR)/(SWIR + NIR)


# ---------------------------------------------------------------------
# Rule-based land cover (a baseline; Stage 7 adds machine learning)
# ---------------------------------------------------------------------
def classify_landcover(ndvi, ndwi, ndbi):
    """
    Give every pixel ONE class using simple threshold rules (set in config.py):
        water       NDWI  > NDWI_WATER_MIN
        vegetation  NDVI  > NDVI_VEGETATION_MIN
        built-up    NDBI  > NDBI_BUILTUP_MIN  and  NDVI < NDVI_BUILTUP_MAX
        otherwise   bare / other
    If rules overlap, water wins, then vegetation, then built-up.
    """
    usable = np.isfinite(ndvi) & np.isfinite(ndwi) & np.isfinite(ndbi)
    classes = np.full(ndvi.shape, CLASS_NODATA, dtype="uint8")
    classes[usable] = CLASS_BARE

    with np.errstate(invalid="ignore"):
        builtup = usable & (ndbi > config.NDBI_BUILTUP_MIN) & (ndvi < config.NDVI_BUILTUP_MAX)
        vegetation = usable & (ndvi > config.NDVI_VEGETATION_MIN)
        water = usable & (ndwi > config.NDWI_WATER_MIN)

    classes[builtup] = CLASS_BUILTUP
    classes[vegetation] = CLASS_VEGETATION
    classes[water] = CLASS_WATER
    return classes


# ---------------------------------------------------------------------
# Numbers for the report
# ---------------------------------------------------------------------
def summarise(ndvi, ndwi, ndbi, landcover, measure_mask, pixel_area_km2):
    """
    Mean index values and class areas. ONLY pixels in measure_mask are used
    (the pixels that are usable in ALL years), so every year measures the
    same piece of ground and a cloud cannot change the totals.
    """
    use = measure_mask & (landcover != CLASS_NODATA)
    count = int(use.sum())
    if count == 0:
        raise ValueError("no pixel is usable in all years")

    class_km2, class_percent = {}, {}
    for code, name in CLASS_NAMES.items():
        n = int((landcover[use] == code).sum())
        class_km2[name] = round(n * pixel_area_km2, 4)
        class_percent[name] = round(n / count * 100, 2)

    return {
        "mean_ndvi": round(float(np.mean(ndvi[use])), 4),
        "mean_ndwi": round(float(np.mean(ndwi[use])), 4),
        "mean_ndbi": round(float(np.mean(ndbi[use])), 4),
        "class_km2": class_km2,
        "class_percent": class_percent,
        "analysed_km2": round(count * pixel_area_km2, 3),
    }


# ---------------------------------------------------------------------
# Pictures
# ---------------------------------------------------------------------
def colorize(values, stops):
    """Turn index values into a colour picture (BGR for OpenCV). NaN -> grey."""
    xs = [s[0] for s in stops]
    colours = np.array([s[1] for s in stops], dtype="float64")
    finite = np.isfinite(values)
    safe = np.where(finite, values, xs[0])
    rgb = np.zeros(values.shape + (3,), dtype="uint8")
    for channel in range(3):
        rgb[..., channel] = np.interp(safe, xs, colours[:, channel]).astype("uint8")
    rgb[~finite] = (110, 110, 110)
    return rgb[..., ::-1].copy()


def landcover_picture(landcover):
    lookup = np.zeros((5, 3), dtype="uint8")
    for code, rgb in CLASS_COLOURS.items():
        lookup[code] = rgb
    return lookup[landcover][..., ::-1].copy()


def save_png(path, image, max_width=900):
    h, w = image.shape[:2]
    if w > max_width:
        image = cv2.resize(image, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(path, image)


def _legends():
    legends = []
    for key, scale in COLOUR_SCALES.items():
        lo, hi = scale["stops"][0][0], scale["stops"][-1][0]
        legends.append({
            "key": key, "title": scale["title"], "min": lo, "max": hi,
            "stops": [{"value": v, "percent": round((v - lo) / (hi - lo) * 100, 1),
                       "color": "rgb(%d,%d,%d)" % rgb} for v, rgb in scale["stops"]],
        })
    legends.append({
        "key": "landcover", "title": "Land cover (rule-based)",
        "classes": [{"name": CLASS_NAMES[c], "color": "rgb(%d,%d,%d)" % CLASS_COLOURS[c]} for c in CLASS_NAMES]
                   + [{"name": "No data / excluded", "color": "rgb(%d,%d,%d)" % CLASS_COLOURS[CLASS_NODATA]}],
    })
    return legends


# ---------------------------------------------------------------------
# The whole stage
# ---------------------------------------------------------------------
def run_indices(out_dir, preview_dir):
    """
    Read the clean files of Stage 4, calculate indices + land cover for every
    year, save them, and return a report (plain dict) for the dashboard.
    Raises ValueError with a friendly message if Stage 4 has not been run.
    """
    grid_path = os.path.join(out_dir, "grid.json")
    mask_path = os.path.join(out_dir, "common_mask.npy")
    if not (os.path.exists(grid_path) and os.path.exists(mask_path)):
        raise ValueError("the images have not been preprocessed yet")
    with open(grid_path, "r", encoding="utf-8") as f:
        grid = json.load(f)
    common = np.load(mask_path)
    os.makedirs(preview_dir, exist_ok=True)

    year_reports = []
    for year in grid["years"]:
        clean_path = os.path.join(out_dir, "clean_%d.npz" % year)
        if not os.path.exists(clean_path):
            raise ValueError("the clean file for %d is missing" % year)
        bands = np.load(clean_path)["bands"]

        ndvi, ndwi, ndbi = calc_ndvi(bands), calc_ndwi(bands), calc_ndbi(bands)
        landcover = classify_landcover(ndvi, ndwi, ndbi)

        np.savez_compressed(os.path.join(out_dir, "indices_%d.npz" % year),
                            ndvi=ndvi, ndwi=ndwi, ndbi=ndbi, landcover=landcover)
        for key, values in (("ndvi", ndvi), ("ndwi", ndwi), ("ndbi", ndbi)):
            save_png(os.path.join(preview_dir, "%s_%d.png" % (key, year)), colorize(values, COLOUR_SCALES[key]["stops"]))
        save_png(os.path.join(preview_dir, "landcover_%d.png" % year), landcover_picture(landcover))

        summary = summarise(ndvi, ndwi, ndbi, landcover, common, grid["pixel_area_km2"])
        summary["year"] = year
        year_reports.append(summary)

    warnings = []
    if all(r["class_km2"]["Water"] == 0 for r in year_reports):
        warnings.append("No water was detected in the analysed area in any year (NDWI above %.2f). "
                        "This is normal for an area without lakes or rivers." % config.NDWI_WATER_MIN)
    if all(r["class_percent"]["Vegetation"] < 1 for r in year_reports):
        warnings.append("Less than 1% of the area is classed as vegetation.")

    return {
        "years": year_reports,
        "formulas": FORMULAS,
        "legends": _legends(),
        "thresholds": {
            "Water": "NDWI > %.2f" % config.NDWI_WATER_MIN,
            "Vegetation": "NDVI > %.2f" % config.NDVI_VEGETATION_MIN,
            "Built-up": "NDBI > %.2f and NDVI < %.2f" % (config.NDBI_BUILTUP_MIN, config.NDVI_BUILTUP_MAX),
            "Bare / other": "everything else",
        },
        "notes": [
            "Only pixels that are usable in ALL years are measured, so every year covers the same ground.",
            "Classes come from simple index thresholds. They are estimates, and Stage 7 adds a machine-learning classifier to compare against.",
            "NDBI also reacts to dry bare soil, so 'built-up' can include bare land.",
        ],
        "warnings": warnings,
    }
