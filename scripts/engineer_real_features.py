#!/usr/bin/env python3
"""SignalForge Feature Engineering - Real IBM Telco Data.

Builds 58 features across 7 groups from the IBM Telco Customer Churn
dataset and writes them to ``data/processed/``.

Run:
    python scripts/engineer_real_features.py

LEAKAGE
-------
Every feature here is row-local: it depends only on a single customer's
own values, never on dataset-wide statistics. Earlier versions included
features derived from the global median / percentile rank and imputed
missing values with the global median, which leak test-set information
when this stage runs over the whole dataset before the train/test split.
Those were removed; any residual NaN/inf is filled with a constant (0),
which carries no information about the data distribution. Model-side
leakage controls (train-only CV, per-fold scaling) live in
``train_with_optuna.py``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT, datefmt=config.LOG_DATEFMT)
logger = logging.getLogger("engineer_features")

SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def load_raw(raw_path: Path) -> pd.DataFrame:
    """Load the raw IBM Telco CSV and validate its shape."""
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {raw_path}. "
            "Run: python scripts/download_real_data.py --source telco"
        )
    df = pd.read_csv(raw_path)
    logger.info("Loaded %s records from IBM Telco dataset", f"{len(df):,}")
    assert "Churn" in df.columns and "MonthlyCharges" in df.columns, "Unexpected raw schema"
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce TotalCharges and encode the binary target."""
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # NOTE: explicit reassignment (not inplace=True on a getitem result),
    # which is a no-op under pandas 3 Copy-on-Write.
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df[config.TARGET_COLUMN] = (df["Churn"] == "Yes").astype(int)
    logger.info("Churn rate: %.1f%%", df[config.TARGET_COLUMN].mean() * 100)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer all feature groups in place and return the frame."""
    # --- Tenure (8) ---
    df["tenure_years"] = df["tenure"] / 12
    df["is_new_customer"] = (df["tenure"] <= 12).astype(int)
    df["is_very_new"] = (df["tenure"] <= 6).astype(int)
    df["is_established"] = (df["tenure"] > 24).astype(int)
    df["is_long_term"] = (df["tenure"] > 48).astype(int)
    df["tenure_risk"] = 1 - (df["tenure"] / 72)
    df["tenure_squared"] = df["tenure"] ** 2
    df["log_tenure"] = np.log1p(df["tenure"])

    # --- Contract (4) ---
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_one_year"] = (df["Contract"] == "One year").astype(int)
    df["is_two_year"] = (df["Contract"] == "Two year").astype(int)
    df["contract_risk"] = df["Contract"].map(
        {"Month-to-month": 1.0, "One year": 0.5, "Two year": 0.0}
    )

    # --- Payment (5) ---
    df["is_auto_payment"] = df["PaymentMethod"].str.contains("automatic", case=False).astype(int)
    df["is_electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["is_paperless"] = (df["PaperlessBilling"] == "Yes").astype(int)
    df["payment_risk"] = df["PaymentMethod"].map(
        {
            "Electronic check": 1.0,
            "Mailed check": 0.7,
            "Bank transfer (automatic)": 0.2,
            "Credit card (automatic)": 0.0,
        }
    )
    df["payment_stability"] = 1 - df["is_electronic_check"] * 0.3

    # --- Service (13) ---
    df["service_count"] = df[SERVICE_COLS].apply(lambda x: (x == "Yes").sum(), axis=1)
    df["has_security"] = (df["OnlineSecurity"] == "Yes").astype(int)
    df["has_tech_support"] = (df["TechSupport"] == "Yes").astype(int)
    df["has_backup"] = (df["OnlineBackup"] == "Yes").astype(int)
    df["has_protection"] = (df["DeviceProtection"] == "Yes").astype(int)
    df["has_internet"] = (df["InternetService"] != "No").astype(int)
    df["is_fiber_optic"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["no_internet"] = (df["InternetService"] == "No").astype(int)
    df["has_phone"] = (df["PhoneService"] == "Yes").astype(int)
    df["has_multiple_lines"] = (df["MultipleLines"] == "Yes").astype(int)
    df["service_adoption_rate"] = df["service_count"] / len(SERVICE_COLS)
    df["no_protection"] = ((df["OnlineSecurity"] == "No") & (df["DeviceProtection"] == "No")).astype(int)
    df["no_support"] = ((df["OnlineSecurity"] == "No") & (df["TechSupport"] == "No")).astype(int)

    # --- Demographic (6) ---
    df["is_senior"] = df["SeniorCitizen"].astype(int)
    df["has_partner"] = (df["Partner"] == "Yes").astype(int)
    df["has_dependents"] = (df["Dependents"] == "Yes").astype(int)
    df["is_single"] = ((df["Partner"] == "No") & (df["Dependents"] == "No")).astype(int)
    df["family_size"] = df["has_partner"] + df["has_dependents"]
    df["is_male"] = (df["gender"] == "Male").astype(int)

    # --- Financial (6) --- all row-local; no dataset-wide statistics (avoids leakage)
    df["avg_monthly_from_total"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["charge_trend"] = df["MonthlyCharges"] - df["avg_monthly_from_total"]
    df["charge_trend_pct"] = df["charge_trend"] / (df["avg_monthly_from_total"] + 0.01)
    df["total_revenue_per_month"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["lifetime_value"] = df["MonthlyCharges"] * (df["tenure"] + 12)
    df["charge_per_service"] = df["MonthlyCharges"] / (df["service_count"] + 1)

    # --- Interaction (5) ---
    df["contract_tenure_risk"] = df["contract_risk"] * df["tenure_risk"]
    df["payment_service_risk"] = df["payment_risk"] * (1 - df["service_adoption_rate"])
    df["value_at_risk"] = df["MonthlyCharges"] * df["contract_risk"]
    df["engagement_score"] = (
        df["service_adoption_rate"] * 0.4
        + (1 - df["tenure_risk"]) * 0.3
        + df["is_auto_payment"] * 0.3
    )
    df["churn_risk_score"] = (
        df["contract_risk"] * 0.35
        + df["tenure_risk"] * 0.25
        + df["payment_risk"] * 0.20
        + (1 - df["service_adoption_rate"]) * 0.10
        + df["is_single"] * 0.10
    )
    return df


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode remaining categoricals, drop raw cols, clean NaN/inf."""
    df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"], drop_first=True)

    drop_cols = [
        "customerID", "Churn", "gender", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Fill inf/residual NaN with a constant (0), NOT a dataset-derived
    # statistic, so no feature value depends on the train/test split.
    # The row-local features above produce no NaN in practice.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df


def write_outputs(df: pd.DataFrame, raw_path: Path) -> None:
    """Persist parquet/csv feature tables and a metadata sidecar."""
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.FEATURES_PARQUET, index=False)
    df.to_csv(config.FEATURES_CSV, index=False)

    n_features = len(df.columns) - 1  # exclude the target
    logger.info("Features: %d | Records: %s", n_features, f"{len(df):,}")
    logger.info("Saved to %s", config.FEATURES_PARQUET)

    metadata = {
        "input_file": str(raw_path),
        "output_file": str(config.FEATURES_PARQUET),
        "n_records": len(df),
        "n_features": n_features,
        "churn_rate": round(float(df[config.TARGET_COLUMN].mean()), 4),
        "feature_groups": {
            "tenure": 8, "contract": 4, "payment": 5, "service": 13,
            "demographic": 6, "financial": 6, "interaction": 5,
            "one_hot": n_features - 47,
        },
        "created_at": datetime.now().isoformat(),
    }
    with open(config.PROCESSED_DATA_DIR / "feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    logger.info("=" * 60)
    logger.info("SignalForge Feature Engineering")
    logger.info("=" * 60)
    df = load_raw(config.RAW_DATA_FILE)
    df = clean(df)
    df = add_features(df)
    df = finalize(df)
    write_outputs(df, config.RAW_DATA_FILE)


if __name__ == "__main__":
    main()
