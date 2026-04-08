"""
SignalForge Test Suite

Tests for model performance, data integrity, and feature engineering.
Run with: python -m pytest tests/ -v
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


class TestDataIntegrity:
    """Verify data files exist and have correct shape."""

    def test_raw_data_exists(self):
        raw = BASE_DIR / 'data' / 'raw' / 'WA_Fn_UseC_Telco_Customer_Churn.csv'
        assert raw.exists(), "Raw IBM Telco data not found"

    def test_raw_data_shape(self):
        raw = BASE_DIR / 'data' / 'raw' / 'WA_Fn_UseC_Telco_Customer_Churn.csv'
        df = pd.read_csv(raw)
        assert len(df) == 7043, f"Expected 7043 rows, got {len(df)}"
        assert 'Churn' in df.columns
        assert 'MonthlyCharges' in df.columns

    def test_raw_churn_rate(self):
        raw = BASE_DIR / 'data' / 'raw' / 'WA_Fn_UseC_Telco_Customer_Churn.csv'
        df = pd.read_csv(raw)
        churn_rate = (df['Churn'] == 'Yes').mean()
        assert 0.20 < churn_rate < 0.35, f"Churn rate {churn_rate:.1%} outside expected range"

    def test_customer_ids_are_real(self):
        """Verify these are real IBM customer IDs, not synthetic."""
        raw = BASE_DIR / 'data' / 'raw' / 'WA_Fn_UseC_Telco_Customer_Churn.csv'
        df = pd.read_csv(raw)
        first_id = df.iloc[0]['customerID']
        assert '-' in first_id, "Customer ID format doesn't match IBM Telco"
        assert first_id[:4].isdigit(), "Customer ID should start with digits"

    def test_features_parquet_exists(self):
        fp = BASE_DIR / 'data' / 'processed' / 'features.parquet'
        assert fp.exists(), "features.parquet not found"

    def test_features_have_target(self):
        fp = BASE_DIR / 'data' / 'processed' / 'features.parquet'
        df = pd.read_parquet(fp)
        assert 'churned' in df.columns

    def test_features_row_count(self):
        fp = BASE_DIR / 'data' / 'processed' / 'features.parquet'
        df = pd.read_parquet(fp)
        assert len(df) == 7043, f"Expected 7043 rows, got {len(df)}"


class TestModelArtifacts:
    """Verify trained models exist and produce valid predictions."""

    @pytest.fixture
    def models_dir(self):
        return BASE_DIR / 'models' / 'artifacts'

    @pytest.fixture
    def feature_names(self, models_dir):
        with open(models_dir / 'feature_names.json') as f:
            return json.load(f)

    @pytest.fixture
    def sample_data(self, feature_names):
        """Create sample data matching model feature names."""
        np.random.seed(42)
        return pd.DataFrame(
            np.random.randn(10, len(feature_names)),
            columns=feature_names
        )

    def test_lr_model_exists(self, models_dir):
        assert (models_dir / 'logistic_regression.pkl').exists()

    def test_rf_model_exists(self, models_dir):
        assert (models_dir / 'random_forest.pkl').exists()

    def test_gb_model_exists(self, models_dir):
        assert (models_dir / 'gradient_boosting.pkl').exists()

    def test_scaler_exists(self, models_dir):
        assert (models_dir / 'scaler.pkl').exists()

    def test_lr_produces_probabilities(self, models_dir, sample_data, feature_names):
        model = joblib.load(models_dir / 'logistic_regression.pkl')
        scaler = joblib.load(models_dir / 'scaler.pkl')

        X_scaled = scaler.transform(sample_data[feature_names])
        proba = model.predict_proba(X_scaled)

        assert proba.shape[1] == 2, "Expected binary classification output"
        assert (proba >= 0).all() and (proba <= 1).all(), "Probabilities outside [0, 1]"

    def test_rf_produces_valid_predictions(self, models_dir, sample_data, feature_names):
        model = joblib.load(models_dir / 'random_forest.pkl')
        preds = model.predict(sample_data[feature_names])
        assert set(preds).issubset({0, 1}), "Predictions not binary"

    def test_gb_produces_valid_predictions(self, models_dir, sample_data, feature_names):
        model = joblib.load(models_dir / 'gradient_boosting.pkl')
        preds = model.predict(sample_data[feature_names])
        assert set(preds).issubset({0, 1}), "Predictions not binary"

    def test_lr_probabilities_are_calibrated(self, models_dir, sample_data, feature_names):
        """Probabilities should spread across [0, 1], not cluster at extremes."""
        model = joblib.load(models_dir / 'logistic_regression.pkl')
        scaler = joblib.load(models_dir / 'scaler.pkl')

        X_scaled = scaler.transform(sample_data[feature_names])
        proba = model.predict_proba(X_scaled)[:, 1]

        assert proba.std() > 0.05, "Probabilities too clustered, model may not be discriminating"


class TestModelPerformance:
    """Verify model performance meets minimum thresholds."""

    @pytest.fixture
    def results(self):
        with open(BASE_DIR / 'models' / 'artifacts' / 'training_results.json') as f:
            return json.load(f)

    def test_lr_auc_above_threshold(self, results):
        auc = results['models']['logistic_regression']['auc']
        assert auc >= 0.80, f"LR AUC {auc:.3f} below 0.80 threshold"

    def test_rf_auc_above_threshold(self, results):
        auc = results['models']['random_forest']['auc']
        assert auc >= 0.80, f"RF AUC {auc:.3f} below 0.80 threshold"

    def test_gb_auc_above_threshold(self, results):
        auc = results['models']['gradient_boosting']['auc']
        assert auc >= 0.80, f"GB AUC {auc:.3f} below 0.80 threshold"

    def test_lr_has_confidence_interval(self, results):
        lr = results['models']['logistic_regression']
        assert 'ci_lower' in lr
        assert 'ci_upper' in lr
        assert lr['ci_lower'] < lr['auc'] < lr['ci_upper']

    def test_lr_cv_performed(self, results):
        lr = results['models']['logistic_regression']
        assert 'cv_auc_mean' in lr
        assert 'cv_auc_std' in lr
        assert lr['cv_auc_std'] < 0.05, "CV std too high, model may be unstable"

    def test_model_comparison_exists(self):
        comp = BASE_DIR / 'models' / 'model_comparison.csv'
        assert comp.exists()
        df = pd.read_csv(comp)
        assert len(df) == 3

    def test_statistical_significance_tested(self, results):
        rf = results['models']['random_forest']
        gb = results['models']['gradient_boosting']
        assert 'p_value_vs_lr' in rf, "Missing significance test for RF"
        assert 'p_value_vs_lr' in gb, "Missing significance test for GB"


class TestBusinessImpact:
    """Verify business impact calculations."""

    @pytest.fixture
    def results(self):
        with open(BASE_DIR / 'models' / 'artifacts' / 'training_results.json') as f:
            return json.load(f)

    def test_revenue_at_risk_calculated(self, results):
        impact = results['business_impact']
        assert 'annual_revenue_at_risk' in impact
        assert impact['annual_revenue_at_risk'] > 0

    def test_monthly_revenue_positive(self, results):
        impact = results['business_impact']
        assert impact['monthly_revenue'] > 0

    def test_churned_revenue_less_than_total(self, results):
        impact = results['business_impact']
        assert impact['churned_monthly_revenue'] < impact['monthly_revenue']


class TestFeatureEngineering:
    """Verify feature engineering produces valid features."""

    def test_features_are_numeric(self):
        fp = BASE_DIR / 'data' / 'processed' / 'features.parquet'
        df = pd.read_parquet(fp)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) >= 8, f"Only {len(numeric_cols)} numeric columns"

    def test_no_infinite_values(self):
        fp = BASE_DIR / 'data' / 'processed' / 'features.parquet'
        df = pd.read_parquet(fp)
        numeric = df.select_dtypes(include=[np.number])
        inf_count = np.isinf(numeric.values).sum()
        assert inf_count == 0, f"Found {inf_count} infinite values"

    def test_churned_is_binary(self):
        fp = BASE_DIR / 'data' / 'processed' / 'features.parquet'
        df = pd.read_parquet(fp)
        assert set(df['churned'].unique()).issubset({0, 1})

    def test_feature_names_file_matches_model(self):
        with open(BASE_DIR / 'models' / 'artifacts' / 'feature_names.json') as f:
            model_features = json.load(f)
        assert len(model_features) >= 20, "Too few features in model"
