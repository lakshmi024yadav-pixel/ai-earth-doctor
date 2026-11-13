"""
check_indices.py
-----------------
WHAT: Runs Stages 3-5 on the sample region from the command line and prints
      the index values and land-class areas.
RUN:  python check_indices.py
"""

import os

import config
from ml import data_loader, indices, preprocessing


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
    preprocessing.run_pipeline(year_paths, out_dir, preview_dir)
    report = indices.run_indices(out_dir, preview_dir)

    print("Formulas:")
    for f in report["formulas"]:
        print("  %-5s = %-28s (bands %s)" % (f["name"], f["formula"], f["bands"]))

    print("\nRules used for land classes:")
    for name, rule in report["thresholds"].items():
        print("  %-13s %s" % (name, rule))

    print("\nPer year (only ground usable in all years is measured):")
    for y in report["years"]:
        print("  %d  mean NDVI %6.3f | NDWI %6.3f | NDBI %6.3f | analysed %.2f km2"
              % (y["year"], y["mean_ndvi"], y["mean_ndwi"], y["mean_ndbi"], y["analysed_km2"]))
        for name, km2 in y["class_km2"].items():
            print("        %-13s %8.3f km2   %5.1f %%" % (name, km2, y["class_percent"][name]))

    for w in report["warnings"]:
        print("\nNOTE:", w)
    print("\nPictures written to:", preview_dir)
    print("Indices finished.")


if __name__ == "__main__":
    main()
