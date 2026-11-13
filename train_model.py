"""
train_model.py
---------------
WHAT:  Trains the machine-learning land-cover classifier (Stage 7) and saves it in models/.
WHY:   The dashboard's "Machine learning" option needs a trained model.
HOW:   1. DATA      the cleaned Sentinel-2 bands of every year (the sample region by default)
       2. LABELS    only CONFIDENT pixels get a label (strict index rules, see config.py)
       3. FEATURES  the five bands B02, B03, B04, B08, B11
       4. TRAIN     Random Forest (scikit-learn), or the built-in classifier if scikit-learn is unavailable
       5. TEST      random hold-out + leave-one-year-out, with a confusion matrix
       6. SAVE      models/land_classifier.pkl and models/land_classifier_info.json

USAGE (from the project folder, with the virtual environment active):
    python train_model.py                      train on the sample region
    python train_model.py --analysis 3         train on analysis #3 (one you ran in the dashboard)
    python train_model.py --analysis 3 --analysis 5     combine several analyses
    python train_model.py --model mlc          use the built-in classifier instead of scikit-learn
"""

import argparse
import os
import sys

import config
from ml import data_loader, indices, land_classification as lc, preprocessing


def load_sample_datasets():
    """Prepare the sample region: check the files, clean the images, calculate the indices."""
    processed = os.path.join(config.PROCESSED_DIR, "demo_region")
    result = data_loader.prepare_region(config.SAMPLE_REGION_DIR, config.SAMPLE_YEARS, processed)
    if result["problems"]:
        print("The sample dataset has problems. Run  python check_dataset.py  first:")
        for p in result["problems"]:
            print("  -", p)
        sys.exit(1)
    year_paths = {year: info["path"] for year, info in result["years"].items()}
    out_dir = os.path.join(config.PROCESSED_DIR, "training_run")
    preview_dir = os.path.join(config.PREVIEW_DIR, "training_run")
    preprocessing.run_pipeline(year_paths, out_dir, preview_dir)
    indices.run_indices(out_dir, preview_dir)
    return lc.load_datasets(out_dir), "sample region (data/sample/demo_region)"


def print_metrics(title, metrics):
    print("\n%s" % title)
    print("  agreement (accuracy) : %.1f %%   (%d test samples)" % (metrics["accuracy"] * 100, metrics["n_test"]))
    print("  %-14s %9s %9s %9s %9s" % ("class", "precision", "recall", "F1", "samples"))
    for name, m in metrics["per_class"].items():
        print("  %-14s %9.3f %9.3f %9.3f %9d" % (name, m["precision"], m["recall"], m["f1"], m["support"]))
    labels = metrics["confusion"]["labels"]
    print("  confusion matrix (rows = rule label, columns = model prediction):")
    print("  %-14s" % "" + "".join("%13s" % l[:12] for l in labels))
    for label, row in zip(labels, metrics["confusion"]["matrix"]):
        print("  %-14s" % label[:14] + "".join("%13d" % v for v in row))


def main():
    parser = argparse.ArgumentParser(description="Train the AI Earth Doctor land-cover classifier.")
    parser.add_argument("--model", choices=["auto", "rf", "mlc"], default="auto",
                        help="auto = Random Forest if scikit-learn works, otherwise the built-in classifier")
    parser.add_argument("--analysis", type=int, action="append",
                        help="id of an analysis you already ran in the dashboard (can be repeated)")
    args = parser.parse_args()

    print("=== STEP 1: data ===")
    if args.analysis:
        datasets, source = [], "analysis " + ", ".join(str(a) for a in args.analysis)
        for analysis_id in args.analysis:
            folder = os.path.join(config.PROCESSED_DIR, "analysis_%d" % analysis_id)
            try:
                datasets.extend(lc.load_datasets(folder))
            except ValueError as error:
                print("Cannot use analysis %d: %s" % (analysis_id, error))
                sys.exit(1)
    else:
        datasets, source = load_sample_datasets()
    print("Years: %s | source: %s" % ([d["year"] for d in datasets], source))

    print("\n=== STEP 2-5: labels, features, training, testing ===")
    print("Label rules (confident pixels only):")
    for name, rule in lc.label_rules_text().items():
        print("  %-13s %s" % (name, rule))
    print("Features: %s" % ", ".join(lc.BAND_ORDER))
    try:
        model, info = lc.train_and_evaluate(datasets, kind=args.model, source_text=source)
    except ValueError as error:
        print("\nTraining stopped: %s" % error)
        sys.exit(1)

    print("\nModel: %s" % info["kind_label"])
    print_metrics("TEST 1 - random hold-out", info["holdout"])
    if info["leave_one_year_out"]:
        lo = info["leave_one_year_out"]
        print_metrics("TEST 2 - leave one year out (trained on %s, tested on %d)" % (lo["train_years"], lo["test_year"]), lo)
    k = info["kmeans_check"]
    print("\nK-MEANS cross-check (no labels, %d): %.1f %% of pixels are in a cluster that matches the rule-based class" % (k["year"], k["agreement_percent"]))
    for c in k["clusters"]:
        print("  cluster %d: %5.1f %% of pixels, mostly %-13s (%.0f %% pure)" % (c["cluster"], c["share_percent"], c["main_rule_class"], c["purity_percent"]))
    if info.get("feature_importance"):
        print("\nFeature importance (how much each band helped): " + ", ".join("%s %.2f" % (b, v) for b, v in info["feature_importance"].items()))
    if info["excluded_classes"]:
        print("\nClasses NOT learned:")
        for name, why in info["excluded_classes"].items():
            print("  %s: %s" % (name, why))

    lc.save_model(model, info)
    print("\nSaved: %s" % os.path.join(config.MODEL_DIR, lc.MODEL_FILE))
    print("Saved: %s" % os.path.join(config.MODEL_DIR, lc.INFO_FILE))
    print("\nREAD THIS: " + lc.CAVEAT)
    print("Training finished. In the dashboard choose 'Machine learning' and click Analyze Region.")


if __name__ == "__main__":
    main()
