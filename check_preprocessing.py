"""
check_preprocessing.py
-----------------------
WHAT: Runs the Stage 4 preprocessing on the sample region from the command line.
WHY:  Shows exactly what each cleaning step did, without opening the browser.
RUN:  python check_preprocessing.py
OUTPUT: prints a report and writes pictures to static/previews/check_run/
"""

import os

import config
from ml import data_loader, preprocessing


def main():
    processed = os.path.join(config.PROCESSED_DIR, "demo_region")
    result = data_loader.prepare_region(config.SAMPLE_REGION_DIR, config.SAMPLE_YEARS, processed)
    if result["problems"]:
        print("The dataset has problems. Run  python check_dataset.py  first:")
        for p in result["problems"]:
            print("  -", p)
        return

    year_paths = {year: info["path"] for year, info in result["years"].items()}
    out_dir = os.path.join(config.PROCESSED_DIR, "check_run")
    preview_dir = os.path.join(config.PREVIEW_DIR, "check_run")
    report = preprocessing.run_pipeline(year_paths, out_dir, preview_dir, mask_clouds=True, denoise_on=False)

    g = report["grid"]
    print("Common grid : %d x %d pixels, %.1f km2, %s" % (g["width"], g["height"], g["area_km2"], g["crs"]))
    print("Pixel area  : %.6f km2" % g["pixel_area_km2"])
    print("Usable in ALL years: %.1f %%" % g["common_usable_percent"])

    print("\nPer year:")
    for y in report["years"]:
        print("  %d  size %s | valid %.1f%% | cloud-like %.2f%% | usable %.1f%% | resampled: %s"
              % (y["year"], y["native_size"], y["valid_percent"], y["cloud_percent"], y["usable_percent"], y["resampled"]))

    print("\nSteps:")
    for s in report["steps"]:
        print("  %-30s %s" % (s["name"], "done   " if s["applied"] else "skipped"), "-", s["detail"])

    if report["warnings"]:
        print("\nWARNINGS:")
        for w in report["warnings"]:
            print("  -", w)

    print("\nClean files : %s" % out_dir)
    print("Pictures    : %s" % preview_dir)
    print("Preprocessing finished.")


if __name__ == "__main__":
    main()
