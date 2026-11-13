"""
change_detection.py
--------------------
WHAT:   Compares the land cover of different years and measures what changed.
WHY:    This is the heart of the project: "how did this place change?"
HOW:    Post-classification comparison, pixel by pixel:
          1. For every pair of years (2018->2019, 2019->2026, 2018->2026) look at
             the land class of each pixel in the first and in the second year.
          2. Net change of a class  = area in the later year - area in the earlier year
             Percentage change      = net change / earlier area x 100
          3. A pixel whose class differs has CHANGED - but only if the index behind the
             change (NDVI, NDWI or NDBI) moved by MORE than the normal difference between
             two dates. That normal difference ("noise") is measured on pixels that did NOT
             change class. Tiny changed patches (fewer than CHANGE_MIN_PIXELS pixels) are
             also ignored.
          4. Count the category changes ("Vegetation -> Built-up", ...) in km2.
          5. Vegetation stress = vegetation that is still vegetation, but whose NDVI fell a lot.
INPUT:  indices_<year>.npz (landcover, ndvi) + common_mask.npy + grid.json from Stages 4-5.
OUTPUT: change_report.json, change maps and NDVI-difference pictures, numbers for the dashboard.
LIMITS: - Errors in the two land-cover maps add up, so small changes are unreliable.
        - Different seasons, sun angle or thin haze look like change.
        - Years that are only 1-2 years apart show little real change.
        - Only ground that is usable in ALL years is compared.
"""

import json
import os

import cv2
import numpy as np

import config
from ml import indices
from ml.indices import (CLASS_BARE, CLASS_BUILTUP, CLASS_NAMES, CLASS_NODATA,
                        CLASS_VEGETATION, CLASS_WATER)

CATEGORY_CODES = [CLASS_WATER, CLASS_VEGETATION, CLASS_BUILTUP, CLASS_BARE]

# Types of change shown on the change map: code -> (name, RGB colour)
CHANGE_NONE = 1
CHANGE_TYPES = {
    2: ("Vegetation → Built-up", (170, 30, 30)),
    3: ("Vegetation → Bare / other", (240, 140, 40)),
    4: ("Bare → Built-up", (160, 60, 200)),
    5: ("Vegetation gain", (30, 200, 80)),
    6: ("Water loss", (150, 100, 50)),
    7: ("Water gain", (30, 120, 230)),
    8: ("Built-up → Bare / other", (235, 215, 60)),
}

# Colour scale for the NDVI difference picture (value, RGB)
DIFF_STOPS = [(-0.4, (180, 30, 30)), (-0.15, (240, 150, 90)), (0.0, (235, 235, 235)),
              (0.15, (150, 210, 130)), (0.4, (20, 120, 40))]


def _build_lookup():
    """Table that turns (class in first year, class in second year) into a change type code."""
    lut = np.zeros((5, 5), dtype="uint8")
    for f in CATEGORY_CODES:
        for t in CATEGORY_CODES:
            if f == t:
                code = CHANGE_NONE
            elif t == CLASS_WATER:
                code = 7
            elif f == CLASS_WATER:
                code = 6
            elif f == CLASS_VEGETATION and t == CLASS_BUILTUP:
                code = 2
            elif f == CLASS_VEGETATION and t == CLASS_BARE:
                code = 3
            elif f == CLASS_BARE and t == CLASS_BUILTUP:
                code = 4
            elif f == CLASS_BUILTUP and t == CLASS_BARE:
                code = 8
            else:
                code = 5          # bare or built-up -> vegetation
            lut[f, t] = code
    return lut


CHANGE_LOOKUP = _build_lookup()


# ---------------------------------------------------------------------
# Core maths
# ---------------------------------------------------------------------
def keep_large_patches(changed):
    """Keep only changed pixels that belong to a patch of at least CHANGE_MIN_PIXELS pixels."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(changed.astype("uint8"), connectivity=8)
    big = stats[:, cv2.CC_STAT_AREA] >= config.CHANGE_MIN_PIXELS
    big[0] = False                      # label 0 is the background
    return big[labels]


def noise_margins(lc_from, lc_to, measured, differences):
    """
    How much does an index normally differ between two dates, even where nothing
    changed? We measure it on pixels whose class stayed the same: the margin is the
    CHANGE_NOISE_PERCENTILE-th percentile of |difference|. A class change smaller
    than this margin is probably just noise (haze, season, sun angle).
    differences = {"ndvi": array, "ndwi": array, "ndbi": array}
    """
    stable = measured & (lc_from == lc_to)
    margins = {}
    for name, diff in differences.items():
        values = np.abs(diff[stable])
        values = values[np.isfinite(values)]
        if values.size >= 1000:
            margin = float(np.percentile(values, config.CHANGE_NOISE_PERCENTILE))
        else:
            margin = config.CHANGE_FALLBACK_MARGIN
        margins[name] = max(margin, config.CHANGE_MIN_MARGIN)
    return margins


def analyse_pair(year_from, year_to, lc_from, lc_to, idx_from, idx_to, common, pixel_km2):
    """
    Compare two land-cover maps. Returns (result_dict, change_codes, ndvi_difference).
    idx_from / idx_to = {"ndvi": array, "ndwi": array, "ndbi": array} of each year.
    Only pixels that are usable in ALL years (common) and classified in both years are compared.
    """
    measured = common & (lc_from != CLASS_NODATA) & (lc_to != CLASS_NODATA)
    n = int(measured.sum())
    if n == 0:
        raise ValueError("no ground is usable in both %d and %d" % (year_from, year_to))

    with np.errstate(invalid="ignore"):
        differences = {k: (idx_to[k] - idx_from[k]).astype("float32") for k in ("ndvi", "ndwi", "ndbi")}
        margins = noise_margins(lc_from, lc_to, measured, differences)

        # For each pixel pick the index that matters for its change:
        # water involved -> NDWI, else vegetation involved -> NDVI, else NDBI (built-up / bare)
        water_involved = (lc_from == CLASS_WATER) | (lc_to == CLASS_WATER)
        veg_involved = (lc_from == CLASS_VEGETATION) | (lc_to == CLASS_VEGETATION)
        used_diff = np.where(water_involved, differences["ndwi"], np.where(veg_involved, differences["ndvi"], differences["ndbi"]))
        used_margin = np.where(water_involved, margins["ndwi"], np.where(veg_involved, margins["ndvi"], margins["ndbi"]))
        significant = np.abs(used_diff) >= used_margin

    # Which pixels really changed (class differs, index moved enough, patch big enough)
    changed = keep_large_patches(measured & (lc_from != lc_to) & significant)
    change_codes = np.zeros(lc_from.shape, dtype="uint8")        # 0 = no data
    change_codes[measured] = CHANGE_NONE
    change_codes[changed] = CHANGE_LOOKUP[lc_from, lc_to][changed]

    # Category by category
    categories = []
    for code in CATEGORY_CODES:
        area_from = int(((lc_from == code) & measured).sum()) * pixel_km2
        area_to = int(((lc_to == code) & measured).sum()) * pixel_km2
        gained = int(((lc_to == code) & (lc_from != code) & changed).sum()) * pixel_km2
        lost = int(((lc_from == code) & (lc_to != code) & changed).sum()) * pixel_km2
        net = area_to - area_from
        percent = round(net / area_from * 100, 2) if area_from >= config.MIN_BASE_AREA_KM2 else None
        categories.append({
            "name": CLASS_NAMES[code],
            "area_from_km2": round(area_from, 4), "area_to_km2": round(area_to, 4),
            "net_km2": round(net, 4), "net_percent": percent,
            "gained_km2": round(gained, 4), "lost_km2": round(lost, 4),
        })

    # Category changes ("Vegetation -> Built-up", ...)
    key = lc_from[changed].astype(int) * 10 + lc_to[changed].astype(int)
    unique, counts = np.unique(key, return_counts=True)
    transitions = sorted(
        ({"from": CLASS_NAMES[u // 10], "to": CLASS_NAMES[u % 10],
          "km2": round(int(c) * pixel_km2, 4), "percent_of_area": round(int(c) / n * 100, 3)}
         for u, c in zip(unique, counts)),
        key=lambda t: -t["km2"])

    change_types = []
    for code, (name, rgb) in CHANGE_TYPES.items():
        count = int((change_codes == code).sum())
        change_types.append({"name": name, "color": "rgb(%d,%d,%d)" % rgb,
                             "km2": round(count * pixel_km2, 4), "percent": round(count / n * 100, 3)})

    # NDVI difference and vegetation stress
    difference = differences["ndvi"].copy()
    difference[~measured] = np.nan
    stress_drop = max(config.STRESS_NDVI_DROP, margins["ndvi"])
    with np.errstate(invalid="ignore"):
        veg_start = int(((lc_from == CLASS_VEGETATION) & measured).sum())
        weakened = (lc_from == CLASS_VEGETATION) & (lc_to == CLASS_VEGETATION) & measured & \
                   (difference <= -stress_drop)
    stress = round(int(weakened.sum()) / veg_start * 100, 2) if veg_start > 0 else None

    changed_km2 = int(changed.sum()) * pixel_km2
    result = {
        "from_year": year_from, "to_year": year_to,
        "label": "%d → %d" % (year_from, year_to),
        "years_apart": year_to - year_from,
        "analysed_km2": round(n * pixel_km2, 3),
        "changed_km2": round(changed_km2, 4),
        "changed_percent": round(int(changed.sum()) / n * 100, 2),
        "categories": categories,
        "transitions": transitions,
        "change_types": change_types,
        "vegetation_stress_percent": stress,
        "stress_ndvi_drop": round(stress_drop, 3),
        "noise_margins": {"NDVI": round(margins["ndvi"], 3), "NDWI": round(margins["ndwi"], 3), "NDBI": round(margins["ndbi"], 3)},
        "mean_ndvi_change": round(float(np.nanmean(difference[measured])), 4),
        "warnings": [],
    }
    if result["changed_percent"] > 25:
        result["warnings"].append(
            "%.0f%% of the ground changed class. That is a lot: check for clouds, haze or a different season in one of the images." % result["changed_percent"])
    if result["years_apart"] <= 1:
        result["warnings"].append(
            "The two images are only %d year apart, so much of the difference may be seasonal or noise, not real land change." % result["years_apart"])
    return result, change_codes, difference


def health_values(pair):
    """The four numbers shown on the dashboard Earth Health cards."""
    by_name = {c["name"]: c for c in pair["categories"]}
    return {
        "vegetation": by_name["Vegetation"]["net_percent"],
        "water": by_name["Water"]["net_percent"],
        "urban": by_name["Built-up"]["net_percent"],
        "cropstress": pair["vegetation_stress_percent"],
    }


# ---------------------------------------------------------------------
# Pictures
# ---------------------------------------------------------------------
def _soft_background(bands):
    """Pale grey-ish version of the true-colour picture, so coloured changes stand out."""
    rgb = np.nan_to_num(np.stack([bands[2], bands[1], bands[0]], axis=-1), nan=0.0)
    rgb = np.clip(rgb / 0.30, 0, 1) ** (1 / 1.8)
    soft = 0.55 * rgb.mean(axis=2, keepdims=True) + 0.45 * rgb
    return (np.clip(soft * 255 * 1.05, 0, 255).astype("uint8"))[:, :, ::-1].copy()   # BGR


def change_picture(change_codes, background):
    image = background.copy()
    image[change_codes == 0] = (110, 110, 110)
    for code, (_, rgb) in CHANGE_TYPES.items():
        image[change_codes == code] = rgb[::-1]
    return image


def _legends():
    lo, hi = DIFF_STOPS[0][0], DIFF_STOPS[-1][0]
    return {
        "change": [{"name": name, "color": "rgb(%d,%d,%d)" % rgb} for name, rgb in CHANGE_TYPES.values()]
                  + [{"name": "No change", "color": "rgb(200,200,200)"}, {"name": "No data / excluded", "color": "rgb(110,110,110)"}],
        "difference": {"title": "Change in NDVI (later − earlier)", "min": lo, "max": hi,
                       "stops": [{"value": v, "percent": round((v - lo) / (hi - lo) * 100, 1),
                                  "color": "rgb(%d,%d,%d)" % rgb} for v, rgb in DIFF_STOPS]},
    }


# ---------------------------------------------------------------------
# The whole stage
# ---------------------------------------------------------------------
METHOD_LABELS = {"rules": "Rule-based (index thresholds)", "ml": "Machine learning"}


def run_change(out_dir, preview_dir, method="rules"):
    """
    Compare every pair of years, save pictures + change_report.json, return the report.
    method = "rules" compares the rule-based land cover (Stage 5),
             "ml"    compares the machine-learning land cover (Stage 7).
    Raises ValueError with a friendly message if the needed earlier stage has not been run.
    """
    if method not in METHOD_LABELS:
        raise ValueError("unknown classification method '%s' (use rules or ml)" % method)
    grid_path = os.path.join(out_dir, "grid.json")
    mask_path = os.path.join(out_dir, "common_mask.npy")
    if not (os.path.exists(grid_path) and os.path.exists(mask_path)):
        raise ValueError("the images have not been preprocessed yet")
    with open(grid_path, "r", encoding="utf-8") as f:
        grid = json.load(f)
    years = grid["years"]
    if len(years) < 2:
        raise ValueError("at least two years are needed")

    maps = {}
    for year in years:
        path = os.path.join(out_dir, "indices_%d.npz" % year)
        if not os.path.exists(path):
            raise ValueError("the indices for %d have not been calculated yet" % year)
        data = np.load(path)
        landcover = data["landcover"]
        if method == "ml":
            ml_path = os.path.join(out_dir, "ml_%d.npz" % year)
            if not os.path.exists(ml_path):
                raise ValueError("the machine-learning classification has not been run for %d yet" % year)
            landcover = np.load(ml_path)["landcover"]
        maps[year] = {"landcover": landcover,
                      "idx": {"ndvi": data["ndvi"], "ndwi": data["ndwi"], "ndbi": data["ndbi"]}}
    common = np.load(mask_path)
    os.makedirs(preview_dir, exist_ok=True)

    pairs_to_compare = list(zip(years[:-1], years[1:]))
    if len(years) > 2:
        pairs_to_compare.append((years[0], years[-1]))      # first -> last, the overall change

    pairs = []
    for year_from, year_to in pairs_to_compare:
        result, codes, difference = analyse_pair(
            year_from, year_to,
            maps[year_from]["landcover"], maps[year_to]["landcover"],
            maps[year_from]["idx"], maps[year_to]["idx"],
            common, grid["pixel_area_km2"])
        np.savez_compressed(os.path.join(out_dir, "change_%d_%d.npz" % (year_from, year_to)),
                            change=codes, ndvi_difference=difference)

        bands = np.load(os.path.join(out_dir, "clean_%d.npz" % year_to))["bands"]
        indices.save_png(os.path.join(preview_dir, "change_%d_%d.png" % (year_from, year_to)),
                         change_picture(codes, _soft_background(bands)))
        indices.save_png(os.path.join(preview_dir, "ndvidiff_%d_%d.png" % (year_from, year_to)),
                         indices.colorize(difference, DIFF_STOPS))
        pairs.append(result)

    method_label = METHOD_LABELS[method]
    ml_report_path = os.path.join(out_dir, "ml_report.json")
    if method == "ml" and os.path.exists(ml_report_path):
        try:
            with open(ml_report_path, "r", encoding="utf-8") as f:
                method_label = "Machine learning (%s)" % json.load(f)["model"]["kind_label"]
        except (OSError, ValueError, KeyError):
            pass

    overall = pairs[-1]
    report = {
        "method": method,
        "method_label": method_label,
        "years": years,
        "pairs": pairs,
        "overall_label": overall["label"],
        "health": health_values(overall),
        "legends": _legends(),
        "settings": {
            "min_patch_pixels": config.CHANGE_MIN_PIXELS,
            "noise_percentile": config.CHANGE_NOISE_PERCENTILE,
            "min_base_area_km2": config.MIN_BASE_AREA_KM2,
            "stress_ndvi_drop": config.STRESS_NDVI_DROP,
        },
        "notes": [
            "Each pixel's land class in the earlier year is compared with its class in the later year. Only ground that is usable in ALL years is compared.",
            "A class change only counts when the index behind it (NDVI for vegetation, NDWI for water, NDBI for built-up/bare) moved by more than the normal difference between the two dates. That normal difference is measured on pixels whose class did not change (their %dth percentile). This stops pixels that sit right at a threshold from flipping back and forth." % config.CHANGE_NOISE_PERCENTILE,
            "Net change = area in the later year minus area in the earlier year. 'Gained' and 'lost' count only changed patches of at least %d pixels; smaller patches are treated as noise, so gained minus lost can differ slightly from the net change." % config.CHANGE_MIN_PIXELS,
            "A percentage change is not shown when a class covered less than %.1f km² at the start, because it would be unreliable." % config.MIN_BASE_AREA_KM2,
            "Vegetation stress is the share of the starting vegetation that is still vegetation but whose NDVI fell by more than the noise margin (at least %.2f). It is an indicator, not a crop-health diagnosis." % config.STRESS_NDVI_DROP,
            "Different dates, seasons, haze and classification errors can look like change. Treat small changes with care.",
            ("The land-cover maps being compared come from the machine-learning classifier. It was trained on confident rule-labelled pixels, so it is a second opinion, not a proven more accurate method."
             if method == "ml" else
             "The land-cover maps being compared come from simple index thresholds (rules). Switch to machine learning on the dashboard to compare."),
        ],
    }
    with open(os.path.join(out_dir, "change_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    return report
