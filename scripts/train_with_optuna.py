#!/usr/bin/env python3
"""SignalForge model training with Optuna hyperparameter optimization.

Trains churn models on the IBM Telco features with statistical rigor:

* Train/test split made up front; the test set is touched exactly once,
  for the final reported metrics.
* Hyperparameter search and cross-validation run on the training split
  only. Logistic Regression scales inside a ``Pipeline`` so the scaler is
  refit within every CV fold (no preprocessing leakage).
* Bootstrap 95% confidence intervals (test set) for every model.
* Paired t-tests vs. Logistic Regression on per-fold training CV scores.
* Ridge-learned feature weights (fit on the training split).

Run:
    python scripts/train_with_optuna.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT, datefmt=config.LOG_DATEFMT)
logger = logging.getLogger("train")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def bootstrap_auc_ci(
    y_true: pd.Series, proba: np.ndarray, n_boot: int = config.N_BOOTSTRAP, ci: float = config.BOOTSTRAP_CI
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for AUC on a held-out set."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    y_arr = np.asarray(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        if len(np.unique(y_arr[idx])) < 2:  # need both classes to score AUC
            continue
        aucs.append(roc_auc_score(y_arr[idx], proba[idx]))
    lo = float(np.percentile(aucs, (1 - ci) / 2 * 100))
    hi = float(np.percentile(aucs, (1 + ci) / 2 * 100))
    return lo, hi


def test_metrics(y_true: pd.Series, proba: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Standard binary-classification metrics on the held-out test set."""
    return {
        "auc": round(float(roc_auc_score(y_true, proba)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "brier_score": round(float(brier_score_loss(y_true, proba)), 4),
    }


def _round_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in params.items()}


def load_features() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the engineered feature table and split off the numeric matrix."""
    if not config.FEATURES_PARQUET.exists():
        raise FileNotFoundError(
            f"{config.FEATURES_PARQUET} not found. "
            "Run: python scripts/engineer_real_features.py"
        )
    df = pd.read_parquet(config.FEATURES_PARQUET)
    logger.info("Loaded %s records | churn rate %.1f%%", f"{len(df):,}", df[config.TARGET_COLUMN].mean() * 100)
    y = df[config.TARGET_COLUMN]
    X = df.drop(columns=[config.TARGET_COLUMN]).select_dtypes(include=[np.number])
    return X, y, X.columns.tolist()


def tune_logistic_regression(
    X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold, sampler: optuna.samplers.BaseSampler
) -> dict[str, Any]:
    """Optuna search for LR, scaling inside the CV Pipeline (leak-free)."""

    def objective(trial: optuna.Trial) -> float:
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        params = {
            "C": trial.suggest_float("C", 1e-3, 100, log=True),
            "penalty": penalty,
            "solver": "liblinear" if penalty == "l1" else "lbfgs",
        }
        pipe = _lr_pipeline(params)
        return cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=config.N_OPTUNA_TRIALS, show_progress_bar=False)
    logger.info("LR best: %s", study.best_params)
    return {"params": study.best_params, "best_value": study.best_value, "n_trials": len(study.trials)}


def _lr_pipeline(params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=params["C"],
                    penalty=params["penalty"],
                    solver="liblinear" if params["penalty"] == "l1" else "lbfgs",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=config.RANDOM_STATE,
                ),
            ),
        ]
    )


def tune_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold, sampler: optuna.samplers.BaseSampler
) -> dict[str, Any]:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
        }
        model = RandomForestClassifier(
            **params, class_weight="balanced", random_state=config.RANDOM_STATE, n_jobs=-1
        )
        return cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=config.N_OPTUNA_TRIALS, show_progress_bar=False)
    logger.info("RF best: %s", study.best_params)
    return {"params": study.best_params, "best_value": study.best_value, "n_trials": len(study.trials)}


def tune_gradient_boosting(
    X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold, sampler: optuna.samplers.BaseSampler
) -> dict[str, Any]:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
        }
        model = GradientBoostingClassifier(**params, random_state=config.RANDOM_STATE)
        return cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=config.N_OPTUNA_TRIALS, show_progress_bar=False)
    logger.info("GB best: %s", study.best_params)
    return {"params": study.best_params, "best_value": study.best_value, "n_trials": len(study.trials)}


def main() -> None:
    logger.info("=" * 60)
    logger.info("SignalForge Model Training with Optuna")
    logger.info("=" * 60)
    np.random.seed(config.RANDOM_STATE)

    X, y, feature_names = load_features()
    logger.info("Features: %d", len(feature_names))

    # Split FIRST. X_test is held out and only used for final metrics.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    logger.info("Train: %s | Test: %s", f"{len(X_train):,}", f"{len(X_test):,}")

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}

    # --- Logistic Regression (leak-free scaling via Pipeline) ---
    logger.info("--- Logistic Regression + Optuna ---")
    lr_tune = tune_logistic_regression(X_train, y_train, cv, sampler)
    lr_pipe = _lr_pipeline(lr_tune["params"])
    lr_cv = cross_val_score(lr_pipe, X_train, y_train, cv=cv, scoring="roc_auc")

    # Final inference artifacts: scaler + bare LR saved separately (inference uses train-fit scaler).
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr = lr_pipe.named_steps["clf"]
    lr.fit(X_train_scaled, y_train)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
    lr_pred = lr.predict(X_test_scaled)
    ci_lower, ci_upper = bootstrap_auc_ci(y_test, lr_proba)

    # Feature effects = log-odds coefficients from the standardized logistic
    # regression (the selected production model). On standardized inputs these
    # are directly comparable across features, and they explain the actual
    # model rather than a separately-fit surrogate.
    coefs = lr.coef_[0]
    feat_coefs = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefs, "abs_coefficient": np.abs(coefs)}
    ).sort_values("abs_coefficient", ascending=False)

    lr_metrics = test_metrics(y_test, lr_proba, lr_pred)
    results["logistic_regression"] = {
        **lr_metrics,
        "cv_auc_mean": round(float(lr_cv.mean()), 4),
        "cv_auc_std": round(float(lr_cv.std()), 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "optuna_params": _round_params(lr_tune["params"]),
        "optuna_best_trial": round(lr_tune["best_value"], 4),
        "optuna_n_trials": lr_tune["n_trials"],
    }
    logger.info("LR AUC %.4f | CV %.4f±%.4f | 95%% CI [%.4f, %.4f]",
                lr_metrics["auc"], lr_cv.mean(), lr_cv.std(), ci_lower, ci_upper)
    joblib.dump(lr, config.ARTIFACTS_DIR / "logistic_regression.pkl")
    joblib.dump(scaler, config.ARTIFACTS_DIR / "scaler.pkl")

    # --- Random Forest ---
    logger.info("--- Random Forest + Optuna ---")
    rf_tune = tune_random_forest(X_train, y_train, cv, sampler)
    rf = RandomForestClassifier(
        **rf_tune["params"], class_weight="balanced", random_state=config.RANDOM_STATE, n_jobs=-1
    ).fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_pred = rf.predict(X_test)
    rf_cv = cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc")
    rf_ci = bootstrap_auc_ci(y_test, rf_proba)
    _, rf_p = stats.ttest_rel(rf_cv, lr_cv)
    results["random_forest"] = {
        **test_metrics(y_test, rf_proba, rf_pred),
        "cv_auc_mean": round(float(rf_cv.mean()), 4),
        "cv_auc_std": round(float(rf_cv.std()), 4),
        "ci_lower": round(rf_ci[0], 4),
        "ci_upper": round(rf_ci[1], 4),
        "p_value_vs_lr": round(float(rf_p), 4),
        "optuna_params": _round_params(rf_tune["params"]),
        "optuna_best_trial": round(rf_tune["best_value"], 4),
        "optuna_n_trials": rf_tune["n_trials"],
    }
    logger.info("RF AUC %.4f | CV %.4f±%.4f | p vs LR %.4f",
                results["random_forest"]["auc"], rf_cv.mean(), rf_cv.std(), rf_p)
    joblib.dump(rf, config.ARTIFACTS_DIR / "random_forest.pkl")

    # --- Gradient Boosting ---
    logger.info("--- Gradient Boosting + Optuna ---")
    gb_tune = tune_gradient_boosting(X_train, y_train, cv, sampler)
    gb = GradientBoostingClassifier(**gb_tune["params"], random_state=config.RANDOM_STATE).fit(X_train, y_train)
    gb_proba = gb.predict_proba(X_test)[:, 1]
    gb_pred = gb.predict(X_test)
    gb_cv = cross_val_score(gb, X_train, y_train, cv=cv, scoring="roc_auc")
    gb_ci = bootstrap_auc_ci(y_test, gb_proba)
    _, gb_p = stats.ttest_rel(gb_cv, lr_cv)
    results["gradient_boosting"] = {
        **test_metrics(y_test, gb_proba, gb_pred),
        "cv_auc_mean": round(float(gb_cv.mean()), 4),
        "cv_auc_std": round(float(gb_cv.std()), 4),
        "ci_lower": round(gb_ci[0], 4),
        "ci_upper": round(gb_ci[1], 4),
        "p_value_vs_lr": round(float(gb_p), 4),
        "optuna_params": _round_params(gb_tune["params"]),
        "optuna_best_trial": round(gb_tune["best_value"], 4),
        "optuna_n_trials": gb_tune["n_trials"],
    }
    logger.info("GB AUC %.4f | CV %.4f±%.4f | p vs LR %.4f",
                results["gradient_boosting"]["auc"], gb_cv.mean(), gb_cv.std(), gb_p)
    joblib.dump(gb, config.ARTIFACTS_DIR / "gradient_boosting.pkl")

    # --- Comparison table (single CI method: bootstrap on test, for all models) ---
    comparison = pd.DataFrame(
        [
            {
                "Model": name,
                "AUC": r["auc"],
                "CI_Lower": r["ci_lower"],
                "CI_Upper": r["ci_upper"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1": r["f1"],
                "P_Value": r.get("p_value_vs_lr"),
            }
            for name, r in [
                ("Logistic Regression", results["logistic_regression"]),
                ("Random Forest", results["random_forest"]),
                ("Gradient Boosting", results["gradient_boosting"]),
            ]
        ]
    )
    # Written to both locations: models/ (tests) and models/artifacts/ (dashboard).
    comparison.to_csv(config.MODELS_DIR / "model_comparison.csv", index=False)
    comparison.to_csv(config.ARTIFACTS_DIR / "model_comparison.csv", index=False)

    full_results = {
        "training_date": datetime.now().isoformat(),
        "dataset": "IBM Telco Customer Churn (Kaggle)",
        "n_customers": int(len(X)),
        "n_features": len(feature_names),
        "churn_rate": round(float(y.mean()), 4),
        "business_impact": {
            "monthly_revenue": round(float(X["MonthlyCharges"].sum()), 2)
            if "MonthlyCharges" in X.columns
            else None,
            "churned_monthly_revenue": round(float(X.loc[y == 1, "MonthlyCharges"].sum()), 2)
            if "MonthlyCharges" in X.columns
            else None,
            "annual_revenue_at_risk": round(float(X.loc[y == 1, "MonthlyCharges"].sum() * 12), 2)
            if "MonthlyCharges" in X.columns
            else None,
        },
        "models": results,
        "feature_names": feature_names,
        "feature_coefficients": feat_coefs.head(20).to_dict("records"),
        "validation": {
            "method": f"{config.CV_FOLDS}-fold stratified cross-validation on the training split",
            "hyperparameter_tuning": f"Optuna TPE (seed={config.RANDOM_STATE}, {config.N_OPTUNA_TRIALS} trials per model)",
            "bootstrap_ci_samples": config.N_BOOTSTRAP,
            "significance_test": "paired t-test on per-fold training CV AUC vs. Logistic Regression",
            "feature_importance": "standardized logistic-regression coefficients (log-odds)",
            "leakage_controls": "test split held out before tuning; LR scaling refit inside each CV fold via Pipeline",
        },
    }
    with open(config.ARTIFACTS_DIR / "training_results.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    with open(config.ARTIFACTS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("%-22s %8s %14s %10s", "Model", "Test AUC", "CV AUC", "p vs LR")
    logger.info("%-22s %8.4f %8.4f±%.4f %10s", "Logistic Regression", lr_metrics["auc"], lr_cv.mean(), lr_cv.std(), "baseline")
    logger.info("%-22s %8.4f %8.4f±%.4f %10.4f", "Random Forest", results["random_forest"]["auc"], rf_cv.mean(), rf_cv.std(), rf_p)
    logger.info("%-22s %8.4f %8.4f±%.4f %10.4f", "Gradient Boosting", results["gradient_boosting"]["auc"], gb_cv.mean(), gb_cv.std(), gb_p)
    logger.info("Optuna: %d trials per model (TPE, seed=%d)", config.N_OPTUNA_TRIALS, config.RANDOM_STATE)


if __name__ == "__main__":
    main()
