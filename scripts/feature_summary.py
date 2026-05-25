#!/usr/bin/env python3
"""Print a quick summary of the engineered feature table.

Run after scripts/engineer_real_features.py:
    python scripts/feature_summary.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("feature_summary")

# Names that actually exist in engineer_real_features.py output.
ENGINEERED_FEATURES = [
    "churn_risk_score", "engagement_score", "tenure_risk", "contract_risk",
    "payment_risk", "is_month_to_month", "is_new_customer", "value_at_risk",
    "service_adoption_rate", "lifetime_value", "charge_per_service",
]


def main() -> None:
    logger.info("=" * 80)
    logger.info("SIGNALFORGE - FEATURE ENGINEERING SUMMARY")
    logger.info("=" * 80)

    if not config.FEATURES_PARQUET.exists():
        logger.error("Features not found. Run: python scripts/engineer_real_features.py")
        sys.exit(1)

    features = pd.read_parquet(config.FEATURES_PARQUET)
    logger.info("\n[DATASET]")
    logger.info("   Records: %s", f"{len(features):,}")
    logger.info("   Total Features: %d", len(features.columns))
    logger.info("   Memory: %.1f MB", features.memory_usage(deep=True).sum() / 1024**2)

    numeric_features = features.select_dtypes(include=["int64", "float64", "uint8"])
    binary_features = numeric_features.loc[:, numeric_features.nunique() <= 2]
    categorical_features = features.select_dtypes(include=["object", "category"])

    logger.info("\n[FEATURE TYPES]")
    logger.info("   Numeric Features: %d", len(numeric_features.columns))
    logger.info("   Binary Features: %d", len(binary_features.columns))
    logger.info("   Categorical Features: %d", len(categorical_features.columns))
    logger.info("   Missing Values: %d", int(features.isnull().sum().sum()))

    logger.info("\n[SAMPLE ENGINEERED FEATURES]")
    for i, feat in enumerate((f for f in ENGINEERED_FEATURES if f in features.columns), 1):
        logger.info("   %2d. %-25s mean=%.2f (%s)", i, feat, features[feat].mean(), features[feat].dtype)

    if config.TARGET_COLUMN in features.columns:
        logger.info("\n[TARGET VARIABLE]")
        logger.info("   Churn Rate: %.1f%%", features[config.TARGET_COLUMN].mean() * 100)
        logger.info("   Churned: %s", f"{int(features[config.TARGET_COLUMN].sum()):,}")
        logger.info("   Retained: %s", f"{int((1 - features[config.TARGET_COLUMN]).sum()):,}")

        logger.info("\n[TOP CORRELATIONS WITH CHURN]")
        correlations = numeric_features.corr()[config.TARGET_COLUMN].drop(config.TARGET_COLUMN)
        logger.info("\n   Positive (churn increases with feature):")
        for feat, corr in correlations.nlargest(5).items():
            logger.info("      %-30s +%.3f", feat, corr)
        logger.info("\n   Negative (churn decreases with feature):")
        for feat, corr in correlations.nsmallest(5).items():
            logger.info("      %-30s %.3f", feat, corr)

    logger.info("\n[NEXT STEPS]")
    logger.info("   1. Train models: python scripts/train_with_optuna.py")
    logger.info("   2. Launch dashboard: streamlit run src/app/dashboard.py")
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()
