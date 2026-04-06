#!/usr/bin/env python3
"""
SignalForge Feature Engineering Pipeline

Transforms raw customer data into ML-ready features.

Usage:
    python scripts/engineer_features.py --input data/raw/telco_processed.csv
    python scripts/engineer_features.py --input data/raw/telco_processed.csv --output data/processed/features.parquet
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import json

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")


class FeatureEngineer:
    """
    Production-grade feature engineering for churn prediction.
    
    Creates 30+ engineered features from raw customer data.
    """
    
    def __init__(self, input_path: str, output_path: str = None):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path) if output_path else Path('data/processed/features.parquet')
        self.df = None
        self.features_created = []
        
    def load_data(self):
        """Load processed dataset."""
        logger.info(f"Loading data from {self.input_path}")
        self.df = pd.read_csv(self.input_path)
        logger.success(f"Loaded {len(self.df):,} records × {len(self.df.columns)} columns")
        return self
    
    def create_tenure_features(self):
        """Create tenure-based features."""
        logger.info("Creating tenure features...")
        
        # Tenure buckets
        self.df['tenure_bucket'] = pd.cut(
            self.df['account_age_months'],
            bins=[-1, 12, 24, 48, 72],
            labels=['0-12mo', '13-24mo', '25-48mo', '49-72mo']
        )
        
        # Tenure risk (new customers = high risk)
        self.df['tenure_risk'] = (
            (self.df['account_age_months'] <= 12).astype(int) * 3 +  # Critical
            ((self.df['account_age_months'] > 12) & (self.df['account_age_months'] <= 24)).astype(int) * 2 +  # High
            ((self.df['account_age_months'] > 24) & (self.df['account_age_months'] <= 48)).astype(int) * 1  # Medium
        )
        
        # Customer lifecycle stage
        self.df['lifecycle_stage'] = pd.cut(
            self.df['account_age_months'],
            bins=[-1, 6, 12, 24, 72],
            labels=['onboarding', 'growth', 'expansion', 'mature']
        )
        
        # Tenure in years
        self.df['tenure_years'] = self.df['account_age_months'] / 12
        
        # Is new customer (0-12 months)
        self.df['is_new_customer'] = (self.df['account_age_months'] <= 12).astype(int)
        
        self.features_created.extend([
            'tenure_bucket', 'tenure_risk', 'lifecycle_stage', 
            'tenure_years', 'is_new_customer'
        ])
        
        logger.success(f"Created {len(self.features_created[-5:])} tenure features")
        return self
    
    def create_contract_features(self):
        """Create contract-based features."""
        logger.info("Creating contract features...")
        
        # Contract risk (month-to-month = highest risk)
        self.df['is_month_to_month'] = (self.df['Contract'] == 'Month-to-month').astype(int)
        self.df['is_annual'] = self.df['Contract'].isin(['One year', 'Two year']).astype(int)
        self.df['is_two_year'] = (self.df['Contract'] == 'Two year').astype(int)
        
        # Contract risk score
        contract_risk_map = {'Month-to-month': 3, 'One year': 1, 'Two year': 0}
        self.df['contract_risk_score'] = self.df['Contract'].map(contract_risk_map)
        
        self.features_created.extend([
            'is_month_to_month', 'is_annual', 'is_two_year', 'contract_risk_score'
        ])
        
        logger.success(f"Created 4 contract features")
        return self
    
    def create_payment_features(self):
        """Create payment-based features."""
        logger.info("Creating payment features...")
        
        # Payment automation
        self.df['is_auto_payment'] = self.df['PaymentMethod'].str.contains('automatic', case=False).astype(int)
        self.df['is_electronic_check'] = (self.df['PaymentMethod'] == 'Electronic check').astype(int)
        self.df['is_paperless'] = (self.df['PaperlessBilling'] == 'Yes').astype(int)
        
        # Payment risk score
        payment_risk_map = {
            'Electronic check': 3,
            'Mailed check': 2,
            'Bank transfer (automatic)': 1,
            'Credit card (automatic)': 0
        }
        self.df['payment_risk_score'] = self.df['PaymentMethod'].map(payment_risk_map)
        
        # Failed payments (proxy based on payment method)
        self.df['payment_stability'] = 1 - (self.df['is_electronic_check'] * 0.3)
        
        self.features_created.extend([
            'is_auto_payment', 'is_electronic_check', 'is_paperless',
            'payment_risk_score', 'payment_stability'
        ])
        
        logger.success(f"Created 5 payment features")
        return self
    
    def create_service_features(self):
        """Create service-based features."""
        logger.info("Creating service features...")
        
        service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies']
        
        # Service count
        self.df['service_count'] = self.df[service_cols].apply(
            lambda x: (x == 'Yes').sum(), axis=1
        )
        
        # Core services (security, support)
        core_services = ['OnlineSecurity', 'TechSupport']
        self.df['core_services_count'] = self.df[core_services].apply(
            lambda x: (x == 'Yes').sum(), axis=1
        )
        
        # Entertainment services
        entertainment_services = ['StreamingTV', 'StreamingMovies']
        self.df['entertainment_count'] = self.df[entertainment_services].apply(
            lambda x: (x == 'Yes').sum(), axis=1
        )
        
        # Has security services
        self.df['has_security'] = (self.df['OnlineSecurity'] == 'Yes').astype(int)
        self.df['has_tech_support'] = (self.df['TechSupport'] == 'Yes').astype(int)
        
        # Service adoption rate (how many services out of possible)
        self.df['service_adoption_rate'] = self.df['service_count'] / len(service_cols)
        
        # No services flag
        self.df['has_no_services'] = (self.df['service_count'] == 0).astype(int)
        
        # Service risk (no security = high risk)
        self.df['service_risk'] = (
            (1 - self.df['has_security']) * 2 +
            (1 - self.df['has_tech_support']) * 2
        )
        
        # Internet service features
        self.df['has_internet'] = (self.df['InternetService'] != 'No').astype(int)
        self.df['is_fiber_optic'] = (self.df['InternetService'] == 'Fiber optic').astype(int)
        self.df['is_dsl'] = (self.df['InternetService'] == 'DSL').astype(int)
        
        # Phone service
        self.df['has_phone'] = (self.df['PhoneService'] == 'Yes').astype(int)
        self.df['has_multiple_lines'] = (self.df['MultipleLines'] == 'Yes').astype(int)
        
        self.features_created.extend([
            'service_count', 'core_services_count', 'entertainment_count',
            'has_security', 'has_tech_support', 'service_adoption_rate',
            'has_no_services', 'service_risk', 'has_internet', 'is_fiber_optic',
            'is_dsl', 'has_phone', 'has_multiple_lines'
        ])
        
        logger.success(f"Created 13 service features")
        return self
    
    def create_demographic_features(self):
        """Create demographic features."""
        logger.info("Creating demographic features...")
        
        # Senior citizen
        self.df['is_senior'] = self.df['SeniorCitizen'].astype(int)
        
        # Family status
        self.df['has_partner'] = (self.df['Partner'] == 'Yes').astype(int)
        self.df['has_dependents'] = (self.df['Dependents'] == 'Yes').astype(int)
        self.df['is_single'] = ((self.df['Partner'] == 'No') & (self.df['Dependents'] == 'No')).astype(int)
        self.df['family_size'] = self.df['has_partner'] + self.df['has_dependents']
        
        # Gender
        self.df['is_male'] = (self.df['gender'] == 'Male').astype(int)
        
        # Demographic risk
        self.df['demographic_risk'] = (
            self.df['is_senior'] * 2 +
            self.df['is_single'] * 1
        )
        
        self.features_created.extend([
            'is_senior', 'has_partner', 'has_dependents', 'is_single',
            'family_size', 'is_male', 'demographic_risk'
        ])
        
        logger.success(f"Created 7 demographic features")
        return self
    
    def create_financial_features(self):
        """Create financial/charge-based features."""
        logger.info("Creating financial features...")
        
        # Charge trends
        self.df['charge_trend'] = self.df['mrr'] - self.df['avg_monthly_charge']
        self.df['charge_trend_pct'] = (
            (self.df['mrr'] - self.df['avg_monthly_charge']) / 
            (self.df['avg_monthly_charge'] + 1)
        )
        
        # High-value customer (top 25% by MRR)
        hv_threshold = self.df['mrr'].quantile(0.75)
        self.df['is_high_value'] = (self.df['mrr'] >= hv_threshold).astype(int)
        
        # MRR percentiles
        self.df['mrr_percentile'] = self.df['mrr'].rank(pct=True)
        
        # Value segment
        self.df['value_segment'] = pd.cut(
            self.df['mrr'],
            bins=[0, 35, 70, 90, 120],
            labels=['low', 'medium', 'high', 'premium']
        )
        
        # Average revenue per month
        self.df['total_revenue_per_month'] = self.df['total_revenue'] / (self.df['account_age_months'] + 1)
        
        # Price sensitivity (higher charges = more likely to churn)
        self.df['price_sensitivity'] = (
            (self.df['mrr'] > self.df['mrr'].median()).astype(int) +
            (self.df['is_month_to_month']).astype(int)
        )
        
        # Lifetime value (proxy)
        self.df['lifetime_value'] = self.df['mrr'] * (self.df['account_age_months'] + 12)
        
        # Charge efficiency (value for money)
        self.df['charge_per_service'] = self.df['mrr'] / (self.df['service_count'] + 1)
        
        self.features_created.extend([
            'charge_trend', 'charge_trend_pct', 'is_high_value', 'mrr_percentile',
            'value_segment', 'total_revenue_per_month', 'price_sensitivity',
            'lifetime_value', 'charge_per_service'
        ])
        
        logger.success(f"Created 9 financial features")
        return self
    
    def create_interaction_features(self):
        """Create interaction features between variables."""
        logger.info("Creating interaction features...")
        
        # Contract × Tenure interaction
        self.df['contract_tenure_risk'] = (
            self.df['contract_risk_score'] * (1 - self.df['account_age_months'] / 72)
        )
        
        # Payment × Services interaction
        self.df['payment_service_risk'] = (
            self.df['payment_risk_score'] * (1 - self.df['service_adoption_rate'])
        )
        
        # Value × Risk interaction
        self.df['value_at_risk'] = self.df['mrr'] * self.df['contract_risk_score']
        
        # Engagement score (composite)
        self.df['engagement_score'] = (
            self.df['service_adoption_rate'] * 0.4 +
            (1 - self.df['tenure_risk'] / 3) * 0.3 +
            self.df['payment_stability'] * 0.3
        )
        
        # Churn risk score (composite - NOT using target variable)
        self.df['churn_risk_score'] = (
            self.df['contract_risk_score'] * 0.3 +
            self.df['tenure_risk'] * 0.25 +
            self.df['payment_risk_score'] * 0.2 +
            self.df['service_risk'] * 0.15 +
            self.df['demographic_risk'] * 0.1
        )
        
        self.features_created.extend([
            'contract_tenure_risk', 'payment_service_risk', 'value_at_risk',
            'engagement_score', 'churn_risk_score'
        ])
        
        logger.success(f"Created 5 interaction features")
        return self
    
    def encode_categoricals(self):
        """Encode categorical features for ML."""
        logger.info("Encoding categorical features...")
        
        # One-hot encode key categoricals
        categorical_cols = ['Contract', 'InternetService', 'PaymentMethod']
        
        for col in categorical_cols:
            if col in self.df.columns:
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.features_created.extend(dummies.columns.tolist())
        
        # Encode ordinal features
        ordinal_mappings = {
            'tenure_bucket': {'0-12mo': 0, '13-24mo': 1, '25-48mo': 2, '49-72mo': 3},
            'lifecycle_stage': {'onboarding': 0, 'growth': 1, 'expansion': 2, 'mature': 3},
            'value_segment': {'low': 0, 'medium': 1, 'high': 2, 'premium': 3}
        }
        
        for col, mapping in ordinal_mappings.items():
            if col in self.df.columns:
                self.df[f'{col}_encoded'] = self.df[col].map(mapping)
                self.features_created.append(f'{col}_encoded')
        
        logger.success(f"Encoded {len(categorical_cols)} categorical features")
        return self
    
    def handle_missing_values(self):
        """Handle missing values in features."""
        logger.info("Handling missing values...")
        
        # Fill numeric NaN with median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        
        # Fill categorical NaN with mode
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
        
        logger.success("Missing values handled")
        return self
    
    def validate_features(self):
        """Validate engineered features."""
        logger.info("Validating features...")
        
        # Check for infinite values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        inf_cols = numeric_cols[(np.abs(self.df[numeric_cols]) == np.inf).any()].tolist()
        
        if inf_cols:
            logger.warning(f"Found infinite values in: {inf_cols}")
            self.df[inf_cols] = self.df[inf_cols].replace([np.inf, -np.inf], np.nan)
        
        # Check for NaN
        nan_counts = self.df.isnull().sum()
        nan_cols = nan_counts[nan_counts > 0]
        
        if len(nan_cols) > 0:
            logger.warning(f"Found NaN in: {nan_cols.to_dict()}")
        
        logger.success("Feature validation complete")
        return self
    
    def save_features(self):
        """Save engineered features to disk."""
        logger.info(f"Saving features to {self.output_path}")
        
        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet (efficient for ML)
        self.df.to_parquet(self.output_path, index=False)
        
        # Also save as CSV for inspection
        csv_path = self.output_path.with_suffix('.csv')
        self.df.to_csv(csv_path, index=False)
        
        logger.success(f"Saved to {self.output_path} and {csv_path}")
        
        # Save feature metadata
        metadata = {
            'input_file': str(self.input_path),
            'output_file': str(self.output_path),
            'n_records': len(self.df),
            'n_features': len(self.df.columns),
            'features_created': self.features_created,
            'feature_groups': {
                'tenure': 5,
                'contract': 4,
                'payment': 5,
                'service': 13,
                'demographic': 7,
                'financial': 9,
                'interaction': 5
            },
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = self.output_path.parent / 'feature_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
        return self
    
    def run(self):
        """Run full feature engineering pipeline."""
        logger.info("=" * 80)
        logger.info("SignalForge Feature Engineering Pipeline")
        logger.info("=" * 80)
        
        (
            self.load_data()
            .create_tenure_features()
            .create_contract_features()
            .create_payment_features()
            .create_service_features()
            .create_demographic_features()
            .create_financial_features()
            .create_interaction_features()
            .encode_categoricals()
            .handle_missing_values()
            .validate_features()
            .save_features()
        )
        
        logger.info("\n" + "=" * 80)
        logger.success(f"Feature Engineering Complete!")
        logger.info("=" * 80)
        
        logger.info(f"\n📊 Summary:")
        logger.info(f"   Input records: {len(self.df):,}")
        logger.info(f"   Output features: {len(self.df.columns)}")
        logger.info(f"   New features created: {len(self.features_created)}")
        logger.info(f"   Output: {self.output_path}")
        
        logger.info(f"\n📋 Feature Groups:")
        logger.info(f"   Tenure features: 5")
        logger.info(f"   Contract features: 4")
        logger.info(f"   Payment features: 5")
        logger.info(f"   Service features: 13")
        logger.info(f"   Demographic features: 7")
        logger.info(f"   Financial features: 9")
        logger.info(f"   Interaction features: 5")
        logger.info(f"   Encoded features: ~20")
        
        return self.df


def main():
    parser = argparse.ArgumentParser(description="Feature engineering for churn prediction")
    parser.add_argument('--input', default='data/raw/telco_processed.csv', help='Input file path')
    parser.add_argument('--output', default='data/processed/features.parquet', help='Output file path')
    
    args = parser.parse_args()
    
    engineer = FeatureEngineer(input_path=args.input, output_path=args.output)
    df = engineer.run()
    
    logger.success("\n🚀 Ready for model training!")


if __name__ == "__main__":
    main()
