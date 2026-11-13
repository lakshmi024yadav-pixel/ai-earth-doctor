"""
check_ml.py
------------
WHAT: Runs Stages 3-5 on the sample region, then classifies it with the trained model
      and compares the machine-learning map with the rule-based map.
RUN:  python check_ml.py        (train first with:  python train_model.py)
"""

import os

import config
from ml import data_loader, indices, land_classification, preprocessing


def main():
    info = land_classification.model_info()
    if info is None:
        print("Training required: no trained model found. Run  python train_model.py  first.")
        return
    print("Model: %s (trained %s on %s)" % (info["kind_label"], info["trained_at"], info["trained_on"]["source"]))

    processed = os.path.join(config.PROCESSED_DIR, "demo_region")
    result = data_loader.prepare_region(config.SAMPLE_REGION_DIR, config.SAMPLE_YEARS, processed)
    if result["problems"]:
        print("The dataset has problems. Run  python check_dataset.py  first:")
        for p in result["problems"]:
            print("  -", p)
        return
    year_paths = {year: item["path"] for year, item in result["years"].items()}
    out_dir = os.path.join(config.PROCESSED_DIR, "check_run")
    preview_dir = os.path.join(config.PREVIEW_DIR, "check_run")
    preprocessing.run_pipeline(year_paths, out_dir, preview_dir)
    indices.run_indices(out_dir, preview_dir)

    try:
        report = land_classification.run_ml_classification(out_dir, preview_dir)
    except ValueError as error:
        print("Could not classify:", error)
        return

    classes = ["Water", "Vegetation", "Built-up", "Bare / other"]
    for y in report["years"]:
        print("\n=== %d ===" % y["year"])
        print("ML agrees with the rule-based map on %.1f %% of the ground | mean confidence %.2f | uncertain pixels %.1f %%"
              % (y["agreement_percent"], y["mean_confidence"], y["low_confidence_percent"]))
        print("%-13s %12s %12s" % ("class", "rules km2", "ML km2"))
        for name in classes:
            print("%-13s %12.3f %12.3f" % (name, y["rule_km2"][name], y["ml_km2"][name]))
    print("\nNOTE: agreement is NOT accuracy. No ground-truth map exists for this area.")
    print("Pictures written to:", preview_dir)
    print("Machine-learning classification finished.")


if __name__ == "__main__":
    main()
