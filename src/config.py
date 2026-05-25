"""Central configuration for SignalForge.

Single source of truth for paths, the random seed, and experiment
hyperparameters. Importing from here keeps magic numbers out of the
pipeline scripts and makes runs reproducible.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths --------------------------------------------------------------
# config.py lives in src/, so the project root is its parent's parent.
BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = BASE_DIR / "models"
ARTIFACTS_DIR: Path = MODELS_DIR / "artifacts"

RAW_DATA_FILE: Path = RAW_DATA_DIR / "WA_Fn_UseC_Telco_Customer_Churn.csv"
FEATURES_PARQUET: Path = PROCESSED_DATA_DIR / "features.parquet"
FEATURES_CSV: Path = PROCESSED_DATA_DIR / "features.csv"

# --- Reproducibility ----------------------------------------------------
RANDOM_STATE: int = 42

# --- Train / evaluate ---------------------------------------------------
TARGET_COLUMN: str = "churned"
TEST_SIZE: float = 0.20
CV_FOLDS: int = 5
# Trial count is overridable via env so CI can run a fast smoke pass.
N_OPTUNA_TRIALS: int = int(os.environ.get("SIGNALFORGE_N_TRIALS", "20"))
N_BOOTSTRAP: int = 1000
BOOTSTRAP_CI: float = 0.95
AUC_TARGET: float = 0.80

# --- Logging ------------------------------------------------------------
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATEFMT: str = "%H:%M:%S"
