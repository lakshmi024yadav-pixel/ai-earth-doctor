"""
config.py
----------
WHAT: One place for folder paths and settings used across the project.
WHY:  app.py and check_dataset.py need the same values. Later stages
      add more settings here (for example the suitability weights).
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folders
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLE_REGION_DIR = os.path.join(DATA_DIR, "sample", "demo_region")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PREVIEW_DIR = os.path.join(BASE_DIR, "static", "previews")   # pictures shown in the browser
MODEL_DIR = os.path.join(BASE_DIR, "models")                 # the trained land-cover classifier (Stage 7)

# Sample region (the real Sentinel-2 files you downloaded).
# The three years must match the folder names inside data/sample/demo_region/
SAMPLE_REGION_NAME = "Gurugram region (Sentinel-2 sample)"
SAMPLE_YEARS = [2018, 2019, 2026]

# Upload rules
ALLOWED_EXTENSIONS = (".tif", ".tiff")
MAX_FILE_MB = 80          # per file; a cropped scene is normally 8-25 MB
MIN_YEAR = 2015           # Sentinel-2 started in 2015
MAX_YEAR = 2035

# Stage 5: index thresholds for the rule-based land cover (a baseline).
# Values are common textbook starting points; change them to see the effect.
NDVI_VEGETATION_MIN = 0.35   # NDVI above this  -> vegetation
NDWI_WATER_MIN = 0.0         # NDWI above this  -> water
NDBI_BUILTUP_MIN = 0.0       # NDBI above this (and low NDVI) -> built-up
NDVI_BUILTUP_MAX = 0.20      # built-up needs NDVI below this

# Stage 6: change detection
CHANGE_MIN_PIXELS = 4        # a changed patch smaller than this many pixels is treated as noise
MIN_BASE_AREA_KM2 = 0.1      # if a class covered less than this at the start, no % change is shown
STRESS_NDVI_DROP = 0.15      # vegetation whose NDVI fell by this much counts as "stressed"
CHANGE_NOISE_PERCENTILE = 90 # a class change must be larger than this share of the normal index differences
CHANGE_MIN_MARGIN = 0.03     # ...but never smaller than this
CHANGE_FALLBACK_MARGIN = 0.10  # used when there are too few unchanged pixels to measure the noise

# Stage 7: machine-learning land classification (train_model.py)
# "Confident" training pixels: pixels that clear these STRICT index rules with a clear margin.
ML_VEG_NDVI_MIN = 0.45            # vegetation: NDVI above this
ML_WATER_NDWI_MIN = 0.10          # water: NDWI above this
ML_BUILT_NDBI_MIN = 0.05          # built-up: NDBI above this ...
ML_BUILT_NDVI_MAX = 0.15          # ... and NDVI below this
ML_BARE_NDVI_MIN = 0.12           # bare / other: NDVI between these two ...
ML_BARE_NDVI_MAX = 0.28
ML_BARE_NDBI_MAX = 0.0            # ... and NDBI below this
ML_SAMPLES_PER_CLASS_PER_YEAR = 4000   # training pixels drawn per class and year
ML_MIN_CLASS_SAMPLES = 300        # a class with fewer confident pixels is NOT learned
ML_TEST_FRACTION = 0.3            # share of the samples kept back for testing
ML_LOW_CONFIDENCE = 0.6           # a prediction below this confidence is called "uncertain"
