"""
check_dataset.py
-----------------
WHAT: Checks your Sentinel-2 sample files from the command line.
WHY:  The quickest way to find download problems, before opening the app.
RUN:  python check_dataset.py
"""

import os

import config
from ml import data_loader


def fmt(value):
    return "n/a" if value is None else "%.4f" % value


def main():
    processed = os.path.join(config.PROCESSED_DIR, "demo_region")
    result = data_loader.prepare_region(config.SAMPLE_REGION_DIR, config.SAMPLE_YEARS, processed)

    print("Folder checked:", config.SAMPLE_REGION_DIR)
    print("Years expected:", config.SAMPLE_YEARS)

    for year in sorted(result["years"]):
        info = result["years"][year]
        m = info["meta"]
        kind = "band files merged into one" if info["source_kind"] == "folder" else "multi-band file"
        print("\n=== %d ===" % year)
        print("  file         : %s (%s)" % (m["file"], kind))
        print("  size         : %d x %d pixels, %d band(s), %s" % (m["width"], m["height"], m["bands"], m["dtype"]))
        print("  coordinates  :", m["crs"])
        if m["pixel_width_m"]:
            print("  pixel size   : %.1f m x %.1f m" % (m["pixel_width_m"], m["pixel_height_m"]))
        if m["area_km2"]:
            print("  area covered : %.1f km2" % m["area_km2"])
        print("  valid pixels : %.1f %%" % m["valid_pixel_percent"])
        print("  value scale  :", m["value_scale"])
        for i, s in enumerate(m["band_stats"]):
            label = data_loader.BAND_ORDER[i] if i < len(data_loader.BAND_ORDER) else "band %d" % (i + 1)
            print("  %-5s min %s   max %s   mean %s" % (label, fmt(s["min"]), fmt(s["max"]), fmt(s["mean"])))

    if result["warnings"]:
        print("\nWARNINGS:")
        for w in result["warnings"]:
            print("  -", w)

    if result["problems"]:
        print("\nPROBLEMS (fix these):")
        for p in result["problems"]:
            print("  -", p)
    else:
        print("\nDataset OK: all years found and valid.")

if __name__ == "__main__":
    main()
