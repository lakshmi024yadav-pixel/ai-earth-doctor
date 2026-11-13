"""
check_change.py
----------------
WHAT: Runs Stages 3-6 on the sample region from the command line and prints the change numbers.
RUN:  python check_change.py
"""

import os

import config
from ml import change_detection, data_loader, indices, preprocessing


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
    indices.run_indices(out_dir, preview_dir)
    report = change_detection.run_change(out_dir, preview_dir)

    for pair in report["pairs"]:
        print("\n=== %s ===" % pair["label"])
        print("analysed %.2f km2 | changed %.3f km2 (%.1f %%) | mean NDVI change %+.4f" % (
            pair["analysed_km2"], pair["changed_km2"], pair["changed_percent"], pair["mean_ndvi_change"]))
        print("noise margins:", pair["noise_margins"], "| vegetation stress:", pair["vegetation_stress_percent"], "%")
        print("%-13s %10s %10s %10s %9s" % ("class", "was km2", "now km2", "net km2", "net %"))
        for c in pair["categories"]:
            percent = "%+.1f%%" % c["net_percent"] if c["net_percent"] is not None else "n/a"
            print("%-13s %10.3f %10.3f %+10.3f %9s" % (c["name"], c["area_from_km2"], c["area_to_km2"], c["net_km2"], percent))
        print("largest category changes:")
        for t in pair["transitions"][:4]:
            print("   %-13s -> %-13s %7.3f km2" % (t["from"], t["to"], t["km2"]))
        for w in pair["warnings"]:
            print("NOTE:", w)

    print("\nDashboard cards (%s):" % report["overall_label"], report["health"])
    print("Pictures written to:", preview_dir)
    print("Change detection finished.")


if __name__ == "__main__":
    main()
