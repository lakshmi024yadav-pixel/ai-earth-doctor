"""
preprocessing.py
-----------------
WHAT:   Cleans the Sentinel-2 images so the years can be compared fairly.
WHY:    Raw satellite images have clouds, empty pixels, different sizes and
        different value scales. If we skip cleaning, "change" would often just
        mean "a cloud was there in one year".
HOW:    Seven simple steps (each is one function below):
          1. load          read the 5-band GeoTIFF of each year
          2. normalise     convert values to reflectance 0-1
          3. missing data  mark empty pixels as NaN ("not a number")
          4. align         put all years on ONE common pixel grid
          5. cloud mask    mark bright cloud-like pixels as unusable
          6. denoise       optional 3x3 median filter
          7. save          write clean arrays + pictures for the next stages
INPUT:  {year: path of 5-band GeoTIFF}
OUTPUT: data/processed/analysis_<id>/clean_<year>.npz  (bands, usable, cloud)
        data/processed/analysis_<id>/common_mask.npy   (usable in ALL years)
        data/processed/analysis_<id>/grid.json         (size, pixel area, ...)
        static/previews/analysis_<id>/rgb_<year>.png and mask_<year>.png
LIMITS: The cloud test is a simple brightness + whiteness rule. It misses thin
        haze and cloud shadows, and can still flag very bright white surfaces.
"""

import json
import math
import os

import cv2
import numpy as np

from ml import geotiff_io
from ml.data_loader import BAND_ORDER

# ---- Settings for the cloud test (reflectance values, 0 to 1) ----
CLOUD_BLUE_MIN = 0.25        # clouds are very bright in blue light
CLOUD_VISIBLE_MIN = 0.25     # ... and bright on average in blue, green, red
CLOUD_NIR_MIN = 0.20         # ... and bright in near-infrared (bare soil is dark in blue)
CLOUD_WHITENESS_MIN = 0.90   # clouds are WHITE: blue >= 0.9 x red. Roofs and soil are redder than blue
CLOUD_MIN_BLOB_PIXELS = 20   # ignore bright specks smaller than this (roofs, cars)
CLOUD_GROW_PIXELS = 2        # grow each cloud by this many pixels (cloud edges are thin)
MEDIAN_KERNEL = 3            # size of the optional noise filter (3x3 pixels)


# ---------------------------------------------------------------------
# Steps 1-3: load, normalise, missing data
# ---------------------------------------------------------------------
def load_stack(path):
    """Step 1. Read a 5-band GeoTIFF. Returns the pixel data and where it is on Earth."""
    raster = geotiff_io.read_geotiff(path)
    if raster["data"].shape[0] != len(BAND_ORDER):
        raise ValueError(
            "%s has %d bands, expected %d" % (os.path.basename(path), raster["data"].shape[0], len(BAND_ORDER))
        )
    if raster["transform"] is None or raster["crs"] is None:
        raise ValueError("%s is not georeferenced (it has no coordinates)" % os.path.basename(path))
    nodata = raster["nodata"]
    if nodata is not None and not math.isfinite(nodata):
        nodata = None
    return {
        "data": raster["data"].astype("float32"),   # shape: (5 bands, height, width)
        "transform": raster["transform"],           # pixel position -> map coordinates
        "crs": raster["crs"],                       # coordinate system
        "nodata": nodata,
    }


def to_reflectance(data, nodata):
    """
    Steps 2 and 3. Returns (reflectance, valid_mask, info).

    - A pixel is INVALID if any band is not a number, equals the nodata value,
      or all five bands are exactly 0 (outside the satellite swath).
    - Values are divided by 10000 if the file stores scaled integers.
    - Reflectance is clipped to 0..1 (very bright pixels can exceed 1).
    - Invalid pixels become NaN, so they are never counted as land.
    """
    invalid = ~np.isfinite(data).all(axis=0)
    if nodata is not None:
        invalid |= (data == nodata).any(axis=0)
    invalid |= (data == 0).all(axis=0)
    valid = ~invalid
    if not valid.any():
        raise ValueError("the image has no valid pixels")

    # 99th percentile tells us the scale: ~0.3 means 0-1, ~3000 means 0-10000
    scale = 10000.0 if float(np.percentile(data[:, valid], 99)) > 2.0 else 1.0
    refl = data / scale

    clipped = float(((refl[:, valid] > 1.0) | (refl[:, valid] < 0.0)).mean() * 100)
    refl = np.clip(refl, 0.0, 1.0)
    refl[:, ~valid] = np.nan
    return refl, valid, {"scale": scale, "clipped_percent": round(clipped, 3)}


# ---------------------------------------------------------------------
# Step 4: alignment (put all years on the same pixel grid)
# ---------------------------------------------------------------------
def _grid_of(info):
    t = info["transform"]
    if abs(t.b) > 1e-12 or abs(t.d) > 1e-12:
        raise ValueError("rotated images are not supported (export a north-up GeoTIFF)")
    return {"a": t.a, "c": t.c, "e": t.e, "f": t.f, "width": info["width"], "height": info["height"]}


def common_grid(infos):
    """
    Build ONE grid that all years share: the area covered by ALL years,
    using the pixel size and pixel positions of the earliest year.
    Returns (grid, same_for_all).
    """
    years = sorted(infos)
    ref_year = years[0]
    ref = _grid_of(infos[ref_year])

    for year in years[1:]:
        if infos[year]["crs"] != infos[ref_year]["crs"]:
            raise ValueError(
                "%d and %d use different coordinate systems. Export every year with the same coordinate system."
                % (year, ref_year)
            )

    # Overlap of all images, in map coordinates
    left, right, top, bottom = -1e30, 1e30, 1e30, -1e30
    grids = {}
    for year in years:
        g = _grid_of(infos[year])
        grids[year] = g
        left = max(left, g["c"])
        right = min(right, g["c"] + g["a"] * g["width"])
        top = min(top, g["f"])
        bottom = max(bottom, g["f"] + g["e"] * g["height"])
    if right <= left or top <= bottom:
        raise ValueError("the years have no area in common. Use the same area box for every year.")

    # Snap the overlap to the reference pixel grid
    col0 = math.ceil((left - ref["c"]) / ref["a"] - 1e-6)
    col1 = math.floor((right - ref["c"]) / ref["a"] + 1e-6)
    row0 = math.ceil((top - ref["f"]) / ref["e"] - 1e-6)
    row1 = math.floor((bottom - ref["f"]) / ref["e"] + 1e-6)
    width, height = col1 - col0, row1 - row0
    if width <= 0 or height <= 0:
        raise ValueError("the common area is smaller than one pixel")

    grid = {
        "a": ref["a"], "e": ref["e"],
        "c": ref["c"] + col0 * ref["a"],
        "f": ref["f"] + row0 * ref["e"],
        "width": width, "height": height,
    }
    same_for_all = all(_same_grid(grids[y], grid) for y in years)
    return grid, same_for_all


def _same_grid(g1, g2):
    tol = abs(g2["a"]) * 1e-3     # a thousandth of a pixel
    return (
        g1["width"] == g2["width"] and g1["height"] == g2["height"]
        and abs(g1["a"] - g2["a"]) <= tol and abs(g1["e"] - g2["e"]) <= tol
        and abs(g1["c"] - g2["c"]) <= tol and abs(g1["f"] - g2["f"]) <= tol
    )


def resample_to_grid(refl, info, grid):
    """
    Copy an image onto the common grid (crop and/or shift, bilinear if pixel
    sizes differ). Pixels outside the original image become NaN.
    """
    src = _grid_of(info)
    if _same_grid(src, grid):
        return refl, False

    sx = grid["a"] / src["a"]
    sy = grid["e"] / src["e"]
    tx = (grid["c"] - src["c"]) / src["a"] + 0.5 * sx - 0.5
    ty = (grid["f"] - src["f"]) / src["e"] + 0.5 * sy - 0.5
    matrix = np.array([[sx, 0.0, tx], [0.0, sy, ty]], dtype=np.float64)   # new pixel -> old pixel

    out = np.empty((refl.shape[0], grid["height"], grid["width"]), dtype="float32")
    for b in range(refl.shape[0]):
        out[b] = cv2.warpAffine(
            refl[b], matrix, (grid["width"], grid["height"]),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"),
        )
    return out, True


def pixel_area_km2(crs, grid):
    """Area of one pixel in km2 (needed to turn pixel counts into km2 later)."""
    transform = geotiff_io.Transform(grid["a"], grid["c"], grid["e"], grid["f"])
    x_mid = grid["c"] + grid["a"] * grid["width"] / 2.0
    y_mid = grid["f"] + grid["e"] * grid["height"] / 2.0
    lat_mid, _ = geotiff_io.to_latlon(crs, x_mid, y_mid)
    width_m, height_m = geotiff_io.pixel_size_m(crs, transform, lat_mid)
    if width_m is None:
        raise ValueError("the coordinate system %s is not supported" % crs.to_string())
    return width_m * height_m / 1e6


# ---------------------------------------------------------------------
# Steps 5-6: cloud mask, noise reduction
# ---------------------------------------------------------------------
def detect_clouds(refl):
    """
    Step 5. A pixel is cloud-like if it is
        - very bright in blue,
        - bright on average in the visible bands,
        - bright in near-infrared, AND
        - "white": blue is at least 0.9 x red (clouds reflect all colours equally,
          while bright roofs, concrete and bare soil are redder than they are blue).
    Small bright specks are dropped (roofs, cars) and real clouds are grown a
    little to cover their thin edges.
    """
    b02, b03, b04, b08 = refl[0], refl[1], refl[2], refl[3]
    with np.errstate(invalid="ignore"):
        visible = (b02 + b03 + b04) / 3.0
        candidate = (
            (b02 > CLOUD_BLUE_MIN) & (visible > CLOUD_VISIBLE_MIN) & (b08 > CLOUD_NIR_MIN)
            & (b02 >= CLOUD_WHITENESS_MIN * b04)
        )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype("uint8"), connectivity=8)
    big = stats[:, cv2.CC_STAT_AREA] >= CLOUD_MIN_BLOB_PIXELS
    big[0] = False                       # label 0 is the background
    keep = big[labels].astype("uint8")   # True only inside blobs that are big enough

    if keep.any() and CLOUD_GROW_PIXELS > 0:
        size = 2 * CLOUD_GROW_PIXELS + 1
        keep = cv2.dilate(keep, np.ones((size, size), dtype="uint8"))
    return keep.astype(bool)


def denoise(refl):
    """Step 6 (optional). 3x3 median filter on each band. NaN pixels stay NaN."""
    out = refl.copy()
    for b in range(out.shape[0]):
        band = out[b]
        good = np.isfinite(band)
        if not good.any():
            continue
        filled = np.where(good, band, np.median(band[good])).astype("float32")
        smooth = cv2.medianBlur(filled, MEDIAN_KERNEL)
        smooth[~good] = np.nan
        out[b] = smooth
    return out


# ---------------------------------------------------------------------
# Pictures for the browser
# ---------------------------------------------------------------------
def _true_colour(bands):
    """Red, green, blue picture (B04, B03, B02) as an 8-bit image (BGR order for OpenCV)."""
    rgb = np.stack([bands[2], bands[1], bands[0]], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    rgb = np.clip(rgb / 0.30, 0, 1) ** (1 / 1.8)          # brighten so it looks natural
    rgb = (rgb * 255).astype("uint8")
    return rgb[:, :, ::-1].copy()                          # RGB -> BGR


def _save_png(path, image, max_width=900):
    h, w = image.shape[:2]
    if w > max_width:
        image = cv2.resize(image, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, image)


def save_previews(preview_dir, year, bands, valid, cloud):
    """rgb_<year>.png = true colour; mask_<year>.png = same picture, clouds magenta, no-data grey."""
    picture = _true_colour(bands)
    _save_png(os.path.join(preview_dir, "rgb_%d.png" % year), picture)

    marked = picture.copy()
    marked[~valid] = (128, 128, 128)       # no data -> grey
    marked[cloud] = (255, 0, 255)          # cloud-like -> magenta (BGR)
    _save_png(os.path.join(preview_dir, "mask_%d.png" % year), marked)


# ---------------------------------------------------------------------
# The whole pipeline
# ---------------------------------------------------------------------
def run_pipeline(year_paths, out_dir, preview_dir, mask_clouds=True, denoise_on=False):
    """
    Run steps 1-7 for every year.

    year_paths = {2018: "path/2018.tif", 2019: "...", 2026: "..."}
    Returns a report (plain dict) that the dashboard displays.
    Raises ValueError with a friendly message if the images cannot be prepared.
    """
    years = sorted(year_paths)
    if len(years) < 2:
        raise ValueError("at least two years are needed")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)

    # Steps 1-3 for each year
    loaded, infos, norm = {}, {}, {}
    for year in years:
        stack = load_stack(year_paths[year])
        refl, valid, info = to_reflectance(stack["data"], stack["nodata"])
        loaded[year] = refl
        norm[year] = info
        infos[year] = {
            "transform": stack["transform"], "crs": stack["crs"],
            "width": refl.shape[2], "height": refl.shape[1],
        }
    native_sizes = {y: (infos[y]["width"], infos[y]["height"]) for y in years}

    # Step 4: common grid
    grid, same_for_all = common_grid(infos)
    crs = infos[years[0]]["crs"]
    cell_km2 = pixel_area_km2(crs, grid)

    year_reports, usable_masks, warnings = [], [], []
    resampled_years = []
    for year in years:
        refl, was_resampled = resample_to_grid(loaded[year], infos[year], grid)
        if was_resampled:
            resampled_years.append(year)
        valid = np.isfinite(refl).all(axis=0)

        # Step 5: clouds
        cloud = detect_clouds(refl) if mask_clouds else np.zeros(valid.shape, dtype=bool)
        cloud &= valid

        # Step 6: noise (optional)
        if denoise_on:
            refl = denoise(refl)

        usable = valid & ~cloud
        refl[:, ~usable] = np.nan

        # Step 7: save
        np.savez_compressed(
            os.path.join(out_dir, "clean_%d.npz" % year),
            bands=refl.astype("float32"), usable=usable, cloud=cloud,
        )
        save_previews(preview_dir, year, np.where(np.isfinite(refl), refl, 0.0), valid, cloud)
        usable_masks.append(usable)

        total = valid.size
        cloud_pct = float(cloud.sum()) / total * 100
        valid_pct = float(valid.sum()) / total * 100
        year_reports.append({
            "year": year,
            "native_size": "%d x %d" % native_sizes[year],
            "resampled": was_resampled,
            "scale_divided_by": norm[year]["scale"],
            "clipped_percent": norm[year]["clipped_percent"],
            "valid_percent": round(valid_pct, 1),
            "cloud_percent": round(cloud_pct, 2),
            "usable_percent": round(float(usable.sum()) / total * 100, 1),
        })
        if cloud_pct > 20:
            warnings.append("%d: %.0f%% of the area looks cloudy, so results for this year may be unreliable." % (year, cloud_pct))
        elif valid_pct < 80:
            warnings.append("%d: only %.0f%% of the common area has data." % (year, valid_pct))

    common = np.logical_and.reduce(usable_masks)
    np.save(os.path.join(out_dir, "common_mask.npy"), common)
    common_pct = float(common.sum()) / common.size * 100
    if common_pct < 60:
        warnings.append("Only %.0f%% of the area is usable in ALL years, so comparisons cover a limited area." % common_pct)
    if not mask_clouds:
        warnings.append("Cloud masking was switched off.")

    grid_info = {
        "width": grid["width"], "height": grid["height"],
        "a": grid["a"], "c": grid["c"], "e": grid["e"], "f": grid["f"],
        "crs": crs.to_string(),
        "pixel_area_km2": cell_km2,
        "area_km2": round(cell_km2 * grid["width"] * grid["height"], 2),
        "reference_year": years[0],
        "years": years,
        "band_order": BAND_ORDER,
        "common_usable_percent": round(common_pct, 1),
    }
    with open(os.path.join(out_dir, "grid.json"), "w", encoding="utf-8") as f:
        json.dump(grid_info, f, indent=2)

    steps = [
        {"name": "1. Load", "applied": True,
         "why": "Read the 5 bands (Blue, Green, Red, NIR, SWIR) of every year.",
         "detail": "%d images loaded." % len(years)},
        {"name": "2. Normalise to reflectance", "applied": True,
         "why": "Satellite files can store values as 0-1 or as 0-10000. Indices need one common scale.",
         "detail": "Values converted to 0-1 and clipped to that range."},
        {"name": "3. Missing data", "applied": True,
         "why": "Empty pixels must never be counted as land, water or buildings.",
         "detail": "Empty pixels were set to NaN (not a number) and are ignored from now on."},
        {"name": "4. Alignment", "applied": True,
         "why": "To compare years pixel by pixel, every year needs the same size and position.",
         "detail": ("All years already shared the same grid (%d x %d pixels). Nothing was resampled."
                    % (grid["width"], grid["height"])) if same_for_all else
                   ("Years %s were cropped/resampled to the common area (%d x %d pixels)."
                    % (", ".join(str(y) for y in resampled_years), grid["width"], grid["height"]))},
        {"name": "5. Cloud mask", "applied": bool(mask_clouds),
         "why": "A cloud in one year would look like land change. Cloud-like pixels are excluded.",
         "detail": "Simple rule: very bright AND white (blue, visible, near-infrared), small specks ignored."
                   if mask_clouds else "Skipped (switched off)."},
        {"name": "6. Noise reduction", "applied": bool(denoise_on),
         "why": "A 3x3 median filter removes single noisy pixels. It also blurs small details, so it is optional.",
         "detail": "3x3 median filter applied." if denoise_on else "Skipped (Sentinel-2 L2A data is already clean)."},
        {"name": "7. Save", "applied": True,
         "why": "The next stages (indices, change detection, ML) read these clean files.",
         "detail": "Clean arrays, masks and pictures saved."},
    ]

    return {
        "grid": grid_info,
        "years": year_reports,
        "steps": steps,
        "warnings": warnings,
        "options": {"mask_clouds": bool(mask_clouds), "denoise": bool(denoise_on)},
    }
