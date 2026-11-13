"""
land_classification.py
-----------------------
WHAT:   Machine-learning land-cover classification of the Sentinel-2 pixels.
WHY:    The rule-based map (Stage 5) uses fixed thresholds. A classifier LEARNS what
        each land class looks like in all five bands and can decide the ambiguous
        pixels that sit close to a threshold. We then compare the two methods.
HOW:    1. LABELS   Only "confident" pixels get a label: pixels that clear STRICT index rules
                    with a margin (config.py, ML_* settings). This is weak supervision:
                    we have no ground-truth map of the area.
        2. FEATURES The five reflectance bands B02, B03, B04, B08, B11 (NOT the indices,
                    otherwise the model would only copy the rules).
        3. TRAIN    Random Forest (scikit-learn), or - if scikit-learn is missing or blocked -
                    a built-in Gaussian maximum-likelihood classifier (pure NumPy).
        4. TEST     Random hold-out split and a "leave one year out" test, with accuracy,
                    precision, recall, F1 and a confusion matrix.
        5. PREDICT  Classify EVERY usable pixel and give a confidence for each.
        K-Means (unsupervised, no labels) is used as an independent cross-check.
INPUT:  clean_<year>.npz and indices_<year>.npz from Stages 4-5.
OUTPUT: models/land_classifier.pkl + models/land_classifier_info.json (training),
        ml_<year>.npz (land cover + confidence) and pictures (prediction).
LIMITS: The test numbers measure agreement with CONFIDENT RULE LABELS, not accuracy
        against the real ground. Without ground-truth labels nobody can say whether the
        ML map is more accurate than the rule map; it is a second opinion.
"""

import json
import os
import pickle
import time

import numpy as np

import config
from ml.data_loader import BAND_ORDER
from ml.indices import (CLASS_BARE, CLASS_BUILTUP, CLASS_NAMES, CLASS_NODATA, CLASS_VEGETATION,
                        CLASS_WATER, colorize, landcover_picture, save_png)

MODEL_FILE = "land_classifier.pkl"
INFO_FILE = "land_classifier_info.json"

KIND_LABELS = {
    "rf": "Random Forest (scikit-learn)",
    "mlc": "Gaussian maximum-likelihood classifier (built in)",
}

CONFIDENCE_STOPS = [(0.3, (190, 40, 40)), (0.6, (240, 190, 60)), (0.8, (150, 200, 90)), (1.0, (20, 120, 40))]

CAVEAT = ("These test numbers measure how well the model reproduces CONFIDENT rule-based labels. "
          "They are NOT accuracy against the real ground, because no ground-truth map is available.")


# ---------------------------------------------------------------------
# 1. Labels: only confident pixels
# ---------------------------------------------------------------------
def confident_labels(ndvi, ndwi, ndbi, usable):
    """
    Label a pixel ONLY if it clearly belongs to one class. Everything else stays 0 (unlabelled).
    The strict rules and their margins are the ML_* settings in config.py.
    """
    labels = np.zeros(ndvi.shape, dtype="uint8")
    with np.errstate(invalid="ignore"):
        bare = (ndvi >= config.ML_BARE_NDVI_MIN) & (ndvi <= config.ML_BARE_NDVI_MAX) & (ndbi < config.ML_BARE_NDBI_MAX)
        built = (ndbi > config.ML_BUILT_NDBI_MIN) & (ndvi < config.ML_BUILT_NDVI_MAX)
        vegetation = ndvi > config.ML_VEG_NDVI_MIN
        water = ndwi > config.ML_WATER_NDWI_MIN
    labels[usable & bare] = CLASS_BARE
    labels[usable & built] = CLASS_BUILTUP
    labels[usable & vegetation] = CLASS_VEGETATION
    labels[usable & water] = CLASS_WATER           # water wins if rules overlap
    return labels


def label_rules_text():
    return {
        "Water": "NDWI > %.2f" % config.ML_WATER_NDWI_MIN,
        "Vegetation": "NDVI > %.2f" % config.ML_VEG_NDVI_MIN,
        "Built-up": "NDBI > %.2f and NDVI < %.2f" % (config.ML_BUILT_NDBI_MIN, config.ML_BUILT_NDVI_MAX),
        "Bare / other": "NDVI %.2f to %.2f and NDBI < %.2f" % (config.ML_BARE_NDVI_MIN, config.ML_BARE_NDVI_MAX, config.ML_BARE_NDBI_MAX),
    }


def build_training_set(datasets, rng):
    """
    datasets = [{"year", "bands" (5,H,W), "ndvi", "ndwi", "ndbi", "usable"}, ...]
    Returns X (n, 5), y (n,), year_of (n,) and the number of confident pixels found per class.
    At most ML_SAMPLES_PER_CLASS_PER_YEAR pixels are drawn per class and year.
    """
    xs, ys, years = [], [], []
    found = {code: 0 for code in CLASS_NAMES}
    for data in datasets:
        labels = confident_labels(data["ndvi"], data["ndwi"], data["ndbi"], data["usable"])
        for code in CLASS_NAMES:
            rows, cols = np.nonzero(labels == code)
            found[code] += len(rows)
            if len(rows) == 0:
                continue
            take = min(len(rows), config.ML_SAMPLES_PER_CLASS_PER_YEAR)
            pick = rng.choice(len(rows), take, replace=False)
            xs.append(data["bands"][:, rows[pick], cols[pick]].T)       # (take, 5)
            ys.append(np.full(take, code, dtype="uint8"))
            years.append(np.full(take, data["year"], dtype="int32"))
    if not xs:
        raise ValueError("no confident training pixels were found")
    return (np.concatenate(xs).astype("float64"), np.concatenate(ys), np.concatenate(years), found)


# ---------------------------------------------------------------------
# 2. Measuring a classifier (plain NumPy, easy to explain)
# ---------------------------------------------------------------------
def split_stratified(y, test_fraction, rng):
    """Random split that keeps the class proportions. Returns (train_index, test_index)."""
    train, test = [], []
    for code in np.unique(y):
        index = np.nonzero(y == code)[0]
        rng.shuffle(index)
        cut = max(1, int(round(len(index) * test_fraction)))
        test.extend(index[:cut])
        train.extend(index[cut:])
    return np.array(train), np.array(test)


def evaluate(y_true, y_pred, codes):
    """Accuracy, per-class precision / recall / F1 and the confusion matrix (rows = truth, columns = prediction)."""
    codes = list(codes)
    matrix = np.zeros((len(codes), len(codes)), dtype="int64")
    for i, truth in enumerate(codes):
        for j, guess in enumerate(codes):
            matrix[i, j] = int(((y_true == truth) & (y_pred == guess)).sum())
    per_class = {}
    for i, code in enumerate(codes):
        tp = matrix[i, i]
        precision = tp / matrix[:, i].sum() if matrix[:, i].sum() else 0.0     # of what we called X, how much was X
        recall = tp / matrix[i, :].sum() if matrix[i, :].sum() else 0.0        # of the real X, how much we found
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[CLASS_NAMES[code]] = {"precision": round(float(precision), 4), "recall": round(float(recall), 4),
                                        "f1": round(float(f1), 4), "support": int(matrix[i, :].sum())}
    return {
        "accuracy": round(float(np.trace(matrix) / matrix.sum()), 4),
        "per_class": per_class,
        "confusion": {"labels": [CLASS_NAMES[c] for c in codes], "matrix": matrix.tolist()},
        "n_test": int(matrix.sum()),
    }


# ---------------------------------------------------------------------
# 3. The classifiers
# ---------------------------------------------------------------------
class GaussianMaximumLikelihood:
    """
    Classic remote-sensing classifier, pure NumPy.
    Each class is described by the mean and covariance of its pixels in the 5 bands (a bell-shaped
    cloud). A new pixel goes to the class whose cloud it is most likely to belong to.
    """

    def __init__(self):
        self.classes_ = None
        self.means_, self.inv_covs_, self.log_dets_ = [], [], []

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.means_, self.inv_covs_, self.log_dets_ = [], [], []
        for code in self.classes_:
            part = X[y == code]
            cov = np.cov(part, rowvar=False) + np.eye(X.shape[1]) * 1e-6      # tiny value keeps it invertible
            sign, log_det = np.linalg.slogdet(cov)
            self.means_.append(part.mean(axis=0))
            self.inv_covs_.append(np.linalg.inv(cov))
            self.log_dets_.append(log_det)
        return self

    def _log_likelihood(self, X):
        scores = np.empty((len(X), len(self.classes_)))
        for k in range(len(self.classes_)):
            diff = X - self.means_[k]
            distance = np.einsum("ij,jk,ik->i", diff, self.inv_covs_[k], diff)   # squared Mahalanobis distance
            scores[:, k] = -0.5 * (self.log_dets_[k] + distance)
        return scores

    def predict_proba(self, X):
        scores = self._log_likelihood(np.asarray(X, dtype="float64"))
        scores -= scores.max(axis=1, keepdims=True)
        probabilities = np.exp(scores)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self._log_likelihood(np.asarray(X, dtype="float64")).argmax(axis=1)]


def sklearn_random_forest():
    """The scikit-learn Random Forest class, or (None, reason) if it cannot be loaded (missing or blocked)."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier, None
    except Exception as error:                   # not installed, or Windows blocked a file
        return None, str(error).splitlines()[0][:150] if str(error) else error.__class__.__name__


def resolve_kind(kind):
    """Decide which classifier to use. Returns (kind, note). 'auto' = Random Forest if possible."""
    if kind not in ("auto", "rf", "mlc"):
        raise ValueError("model kind must be auto, rf or mlc")
    if kind == "mlc":
        return "mlc", None
    forest, reason = sklearn_random_forest()
    if forest is not None:
        return "rf", None
    note = "scikit-learn could not be used (%s), so the built-in Gaussian classifier is used instead." % reason
    if kind == "rf":
        raise ValueError(note)
    return "mlc", note


def make_estimator(kind):
    if kind == "rf":
        forest, _ = sklearn_random_forest()
        return forest(n_estimators=40, max_depth=10, min_samples_leaf=10, class_weight="balanced",
                      random_state=42, n_jobs=1)
    return GaussianMaximumLikelihood()


# ---------------------------------------------------------------------
# 4. K-Means: an independent, unsupervised cross-check (no labels used)
# ---------------------------------------------------------------------
def _kmeans_once(X, k, rng, iterations):
    """One K-Means run. Start: K-Means++ (each new centre is picked far from the centres chosen so far)."""
    centers = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        nearest = np.min(((X[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
        if nearest.sum() <= 0:
            centers.append(X[rng.integers(len(X))])
        else:
            centers.append(X[rng.choice(len(X), p=nearest / nearest.sum())])
    centers = np.array(centers)

    labels = np.zeros(len(X), dtype="int64")
    for _ in range(iterations):
        distance = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distance.argmin(axis=1)
        moved = np.array([X[labels == j].mean(axis=0) if (labels == j).any() else centers[j] for j in range(k)])
        if np.allclose(moved, centers, atol=1e-7):
            break
        centers = moved
    inertia = float(((X - centers[labels]) ** 2).sum())        # how tight the clusters are (smaller = better)
    return labels, centers, inertia


def kmeans(X, k, rng, iterations=40, restarts=5):
    """
    K-Means: repeat (1) give every pixel to the nearest centre, (2) move each centre to the mean of its pixels.
    Run it a few times from different starts and keep the tightest result, because one bad start can give a poor answer.
    """
    best = None
    for _ in range(restarts):
        result = _kmeans_once(X, k, rng, iterations)
        if best is None or result[2] < best[2]:
            best = result
    return best[0], best[1]


def kmeans_check(data, rng, k=4, sample=20000):
    """Cluster pixels of one year without labels, then see how the clusters line up with the rule-based classes."""
    rows, cols = np.nonzero(data["usable"])
    pick = rng.choice(len(rows), min(sample, len(rows)), replace=False)
    X = data["bands"][:, rows[pick], cols[pick]].T.astype("float64")
    rule = data["rule_landcover"][rows[pick], cols[pick]]
    labels, centers = kmeans(X, k, rng)
    clusters, matched = [], 0
    for j in range(k):
        members = rule[labels == j]
        if len(members) == 0:
            continue
        counts = {code: int((members == code).sum()) for code in CLASS_NAMES}
        top = max(counts, key=counts.get)
        matched += counts[top]
        clusters.append({"cluster": j + 1, "share_percent": round(len(members) / len(rule) * 100, 1),
                         "main_rule_class": CLASS_NAMES[top], "purity_percent": round(counts[top] / len(members) * 100, 1)})
    return {"year": data["year"], "k": k, "agreement_percent": round(matched / len(rule) * 100, 1), "clusters": clusters}


def load_datasets(out_dir):
    """Read the cleaned bands and the indices of one analysis folder (made by Stages 4 and 5)."""
    grid_path = os.path.join(out_dir, "grid.json")
    if not os.path.exists(grid_path):
        raise ValueError("%s has not been preprocessed. Click Analyze Region in the dashboard first" % os.path.basename(out_dir))
    with open(grid_path, "r", encoding="utf-8") as f:
        grid = json.load(f)
    datasets = []
    for year in grid["years"]:
        clean_path, index_path = os.path.join(out_dir, "clean_%d.npz" % year), os.path.join(out_dir, "indices_%d.npz" % year)
        if not (os.path.exists(clean_path) and os.path.exists(index_path)):
            raise ValueError("%s has no indices for %d. Click Analyze Region in the dashboard first" % (os.path.basename(out_dir), year))
        bands = np.load(clean_path)["bands"]
        indices = np.load(index_path)
        datasets.append({"year": year, "bands": bands, "ndvi": indices["ndvi"], "ndwi": indices["ndwi"],
                         "ndbi": indices["ndbi"], "usable": np.isfinite(bands).all(axis=0),
                         "rule_landcover": indices["landcover"]})
    return datasets


# ---------------------------------------------------------------------
# 5. Training (used by train_model.py)
# ---------------------------------------------------------------------
def train_and_evaluate(datasets, kind="auto", seed=42, source_text="", log=print):
    """
    Build the training set, test the model two ways, train the final model on all samples.
    Returns (estimator, info) where info is a plain dict that is saved next to the model.
    """
    rng = np.random.default_rng(seed)
    kind, note = resolve_kind(kind)
    if note:
        log("NOTE: " + note)

    X, y, year_of, found = build_training_set(datasets, rng)
    log("Confident pixels found: " + ", ".join("%s %d" % (CLASS_NAMES[c], n) for c, n in found.items()))

    # A class needs enough examples, otherwise the model would learn noise
    counts = {code: int((y == code).sum()) for code in CLASS_NAMES}
    excluded, keep = {}, []
    for code, name in CLASS_NAMES.items():
        total_found = found[code]
        if total_found < config.ML_MIN_CLASS_SAMPLES:
            excluded[name] = "only %d confident example pixels (at least %d needed), so this class is not learned" % (total_found, config.ML_MIN_CLASS_SAMPLES)
        else:
            keep.append(code)
    if len(keep) < 2:
        raise ValueError("fewer than two classes have enough confident example pixels, so no model can be trained")
    for name, why in excluded.items():
        log("EXCLUDED class %s: %s" % (name, why))
    use = np.isin(y, keep)
    X, y, year_of = X[use], y[use], year_of[use]
    codes = sorted(keep)
    log("Training samples: " + ", ".join("%s %d" % (CLASS_NAMES[c], int((y == c).sum())) for c in codes))

    # Test 1: random hold-out
    train_i, test_i = split_stratified(y, config.ML_TEST_FRACTION, rng)
    model = make_estimator(kind).fit(X[train_i], y[train_i])
    holdout = evaluate(y[test_i], model.predict(X[test_i]), codes)
    log("Hold-out test (%d samples): agreement %.1f %%" % (holdout["n_test"], holdout["accuracy"] * 100))

    # Test 2: leave one year out (train on the other years, test on the last year)
    years = sorted(set(int(v) for v in year_of))
    leave_out = None
    if len(years) >= 2:
        test_year = years[-1]
        is_test = year_of == test_year
        if set(np.unique(y[~is_test])) >= set(np.unique(y[is_test])):
            other = make_estimator(kind).fit(X[~is_test], y[~is_test])
            leave_out = evaluate(y[is_test], other.predict(X[is_test]), codes)
            leave_out["test_year"] = test_year
            leave_out["train_years"] = years[:-1]
            log("Leave-one-year-out (train %s, test %d): agreement %.1f %%" % (years[:-1], test_year, leave_out["accuracy"] * 100))

    # Final model on all samples
    final = make_estimator(kind).fit(X, y)

    # K-Means cross-check on the last year (no labels)
    last = max(datasets, key=lambda d: d["year"])
    cross = kmeans_check(last, rng, k=len(codes) + 1)
    log("K-Means cross-check (%d): %.1f %% of pixels fall in a cluster that matches the rule-based class" % (last["year"], cross["agreement_percent"]))

    info = {
        "kind": kind,
        "kind_label": KIND_LABELS[kind],
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "trained_on": {"source": source_text, "years": [d["year"] for d in datasets]},
        "features": list(BAND_ORDER),
        "classes": {str(c): CLASS_NAMES[c] for c in codes},
        "excluded_classes": excluded,
        "confident_pixels_found": {CLASS_NAMES[c]: n for c, n in found.items()},
        "training_samples": {CLASS_NAMES[c]: int((y == c).sum()) for c in codes},
        "label_rules": label_rules_text(),
        "holdout": holdout,
        "leave_one_year_out": leave_out,
        "kmeans_check": cross,
        "caveat": CAVEAT,
    }
    if kind == "rf":
        info["feature_importance"] = {b: round(float(v), 4) for b, v in zip(BAND_ORDER, final.feature_importances_)}
        import sklearn
        info["sklearn_version"] = sklearn.__version__
    return final, info


def save_model(estimator, info, model_dir=None):
    model_dir = model_dir or config.MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, MODEL_FILE), "wb") as f:
        pickle.dump({"kind": info["kind"], "estimator": estimator}, f)
    with open(os.path.join(model_dir, INFO_FILE), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)


def model_info(model_dir=None):
    """The saved training report, or None if no model has been trained."""
    path = os.path.join(model_dir or config.MODEL_DIR, INFO_FILE)
    if not (os.path.exists(path) and os.path.exists(os.path.join(model_dir or config.MODEL_DIR, MODEL_FILE))):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_model(model_dir=None):
    """Load the trained model. Raises ValueError with a friendly message if that is not possible."""
    model_dir = model_dir or config.MODEL_DIR
    info = model_info(model_dir)
    if info is None:
        raise ValueError("no trained model found. Training required: run  python train_model.py  first")
    try:
        with open(os.path.join(model_dir, MODEL_FILE), "rb") as f:
            saved = pickle.load(f)
        return saved["estimator"], info
    except Exception as error:
        raise ValueError("the saved model could not be loaded (%s). Train it again with  python train_model.py  "
                         "(add  --model mlc  to use the built-in classifier)" % str(error).splitlines()[0][:100])


# ---------------------------------------------------------------------
# 6. Predicting a whole image
# ---------------------------------------------------------------------
def predict_map(estimator, bands, usable, chunk=100000):
    """
    Classify every usable pixel. Returns (landcover uint8 map, confidence float32 map).
    Confidence = the probability of the chosen class (1.0 = completely sure).
    """
    landcover = np.zeros(usable.shape, dtype="uint8")
    confidence = np.zeros(usable.shape, dtype="float32")
    rows, cols = np.nonzero(usable)
    classes = np.asarray(estimator.classes_)
    for start in range(0, len(rows), chunk):
        r, c = rows[start:start + chunk], cols[start:start + chunk]
        probabilities = estimator.predict_proba(bands[:, r, c].T.astype("float64"))
        best = probabilities.argmax(axis=1)
        landcover[r, c] = classes[best].astype("uint8")
        confidence[r, c] = probabilities[np.arange(len(best)), best]
    return landcover, confidence


def run_ml_classification(out_dir, preview_dir, model_dir=None):
    """
    Classify every year of an analysis with the trained model and compare with the rule-based map.
    Needs Stages 4 and 5 for this analysis. Raises ValueError with a friendly message otherwise.
    """
    estimator, info = load_model(model_dir)
    grid_path, mask_path = os.path.join(out_dir, "grid.json"), os.path.join(out_dir, "common_mask.npy")
    if not (os.path.exists(grid_path) and os.path.exists(mask_path)):
        raise ValueError("the images have not been preprocessed yet")
    with open(grid_path, "r", encoding="utf-8") as f:
        grid = json.load(f)
    common = np.load(mask_path)
    os.makedirs(preview_dir, exist_ok=True)
    pixel_km2 = grid["pixel_area_km2"]

    names = [CLASS_NAMES[c] for c in (CLASS_WATER, CLASS_VEGETATION, CLASS_BUILTUP, CLASS_BARE)]
    codes = [CLASS_WATER, CLASS_VEGETATION, CLASS_BUILTUP, CLASS_BARE]
    years = []
    for year in grid["years"]:
        clean_path, index_path = os.path.join(out_dir, "clean_%d.npz" % year), os.path.join(out_dir, "indices_%d.npz" % year)
        if not (os.path.exists(clean_path) and os.path.exists(index_path)):
            raise ValueError("the indices for %d have not been calculated yet" % year)
        bands = np.load(clean_path)["bands"]
        rule = np.load(index_path)["landcover"]
        usable = np.isfinite(bands).all(axis=0)

        ml_map, confidence = predict_map(estimator, bands, usable)
        np.savez_compressed(os.path.join(out_dir, "ml_%d.npz" % year), landcover=ml_map, confidence=confidence)
        save_png(os.path.join(preview_dir, "mlcover_%d.png" % year), landcover_picture(ml_map))
        shown = np.where(usable, confidence, np.nan)
        save_png(os.path.join(preview_dir, "mlconf_%d.png" % year), colorize(shown, CONFIDENCE_STOPS))

        measured = common & (ml_map != CLASS_NODATA) & (rule != CLASS_NODATA)
        n = int(measured.sum())
        if n == 0:
            raise ValueError("no ground is usable in all years")
        matrix = np.zeros((4, 4))
        for i, a in enumerate(codes):
            for j, b in enumerate(codes):
                matrix[i, j] = ((rule == a) & (ml_map == b) & measured).sum() / n * 100
        years.append({
            "year": year,
            "ml_km2": {CLASS_NAMES[c]: round(int(((ml_map == c) & measured).sum()) * pixel_km2, 4) for c in codes},
            "rule_km2": {CLASS_NAMES[c]: round(int(((rule == c) & measured).sum()) * pixel_km2, 4) for c in codes},
            "agreement_percent": round(float((ml_map == rule)[measured].mean() * 100), 2),
            "agreement_matrix": {"labels": names, "percent": np.round(matrix, 2).tolist()},
            "mean_confidence": round(float(confidence[measured].mean()), 3),
            "low_confidence_percent": round(float((confidence[measured] < config.ML_LOW_CONFIDENCE).mean() * 100), 2),
            "analysed_km2": round(n * pixel_km2, 3),
        })

    report = {
        "model": {k: info[k] for k in ("kind", "kind_label", "trained_at", "trained_on", "classes", "excluded_classes",
                                       "training_samples", "label_rules", "holdout", "leave_one_year_out", "kmeans_check", "caveat")},
        "years": years,
        "low_confidence_threshold": config.ML_LOW_CONFIDENCE,
        "confidence_legend": {"min": 0.3, "max": 1.0, "stops": [
            {"percent": round((v - 0.3) / 0.7 * 100, 1), "color": "rgb(%d,%d,%d)" % rgb} for v, rgb in CONFIDENCE_STOPS]},
        "notes": [
            "The model was trained on CONFIDENT pixels only, then classifies every pixel, including the ambiguous ones near the rule thresholds.",
            "'Agreement' compares the ML map with the rule-based map. It is NOT accuracy: with no ground-truth map, neither method is known to be right.",
            "Confidence is the probability the model gives to its chosen class. Pixels below %.0f%% are uncertain." % (config.ML_LOW_CONFIDENCE * 100),
            "A class that had too few confident training pixels is never predicted (see 'excluded classes').",
        ],
    }
    with open(os.path.join(out_dir, "ml_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    return report
