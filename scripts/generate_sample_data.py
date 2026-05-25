#!/usr/bin/env python3
"""Generate a SYNTHETIC sample dataset in the IBM Telco schema.

This is a deterministic stand-in so the pipeline, tests, and CI can run
end-to-end on a clean clone WITHOUT Kaggle credentials. The data is
random (with planted signal); the resulting metrics are NOT the project's
real results. For real numbers, download the actual dataset:

    python scripts/download_real_data.py --source telco

Run:
    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import logging
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT, datefmt=config.LOG_DATEFMT)
logger = logging.getLogger("generate_sample_data")

N_ROWS = 7043  # matches the real IBM Telco dataset size


def generate(n: int = N_ROWS, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    def yn(p: float) -> np.ndarray:
        return rng.choice(["Yes", "No"], n, p=[p, 1 - p])

    def svc(p_yes: float) -> np.ndarray:
        return rng.choice(["Yes", "No", "No internet service"], n, p=[p_yes, 0.78 - p_yes, 0.22])

    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    tenure = rng.integers(0, 73, n)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21],
    )
    monthly = np.round(rng.uniform(18, 119, n), 2)

    def cust_id() -> str:
        return f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list(string.ascii_uppercase), 5))}"

    df = pd.DataFrame(
        {
            "customerID": [cust_id() for _ in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
            "Partner": yn(0.48),
            "Dependents": yn(0.30),
            "tenure": tenure,
            "PhoneService": yn(0.90),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n, p=[0.42, 0.48, 0.10]),
            "InternetService": internet,
            "OnlineSecurity": svc(0.29),
            "OnlineBackup": svc(0.34),
            "DeviceProtection": svc(0.34),
            "TechSupport": svc(0.29),
            "StreamingTV": svc(0.38),
            "StreamingMovies": svc(0.39),
            "Contract": contract,
            "PaperlessBilling": yn(0.59),
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
        }
    )
    df["TotalCharges"] = np.round(df["MonthlyCharges"] * (df["tenure"] + rng.uniform(0, 1, n)), 2)

    # Plant a learnable churn signal so the trained model discriminates.
    logit = (
        -1.6
        + 1.9 * (contract == "Month-to-month")
        - 0.045 * tenure
        + 0.9 * (payment == "Electronic check")
        + 0.8 * (internet == "Fiber optic")
        + 0.004 * (monthly - 60)
    )
    p_churn = 1 / (1 + np.exp(-logit))
    df["Churn"] = np.where(rng.random(n) < p_churn, "Yes", "No")
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "  # mirror the real dataset's blank TotalCharges
    return df


def main() -> None:
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.to_csv(config.RAW_DATA_FILE, index=False)
    logger.warning("Wrote SYNTHETIC sample data (not real results) to %s", config.RAW_DATA_FILE)
    logger.info("Rows: %s | churn rate: %.1f%%", f"{len(df):,}", (df["Churn"] == "Yes").mean() * 100)


if __name__ == "__main__":
    main()
