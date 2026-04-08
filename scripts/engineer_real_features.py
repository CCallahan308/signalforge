#!/usr/bin/env python3
"""
SignalForge Feature Engineering - Real IBM Telco Data

Properly engineered features from the actual IBM Telco Customer Churn dataset.
Creates 58 features across 7 groups.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent

print("=" * 60)
print("SignalForge Feature Engineering")
print("=" * 60)

# Load raw data
raw_path = BASE_DIR / 'data' / 'raw' / 'WA_Fn_UseC_Telco_Customer_Churn.csv'
df = pd.read_csv(raw_path)
print(f"Loaded {len(df):,} records from IBM Telco dataset")

# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'], inplace=True)

# Encode target
df['churned'] = (df['Churn'] == 'Yes').astype(int)
print(f"Churn rate: {df['churned'].mean():.1%}")

# ============================================================
# TENURE FEATURES (8)
# ============================================================
df['tenure_years'] = df['tenure'] / 12
df['is_new_customer'] = (df['tenure'] <= 12).astype(int)
df['is_very_new'] = (df['tenure'] <= 6).astype(int)
df['is_established'] = (df['tenure'] > 24).astype(int)
df['is_long_term'] = (df['tenure'] > 48).astype(int)
df['tenure_risk'] = 1 - (df['tenure'] / 72)
df['tenure_squared'] = df['tenure'] ** 2
df['log_tenure'] = np.log1p(df['tenure'])

# ============================================================
# CONTRACT FEATURES (4)
# ============================================================
df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
df['is_one_year'] = (df['Contract'] == 'One year').astype(int)
df['is_two_year'] = (df['Contract'] == 'Two year').astype(int)
df['contract_risk'] = df['Contract'].map({
    'Month-to-month': 1.0, 'One year': 0.5, 'Two year': 0.0
})

# ============================================================
# PAYMENT FEATURES (5)
# ============================================================
df['is_auto_payment'] = df['PaymentMethod'].str.contains('automatic', case=False).astype(int)
df['is_electronic_check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
df['is_paperless'] = (df['PaperlessBilling'] == 'Yes').astype(int)
df['payment_risk'] = df['PaymentMethod'].map({
    'Electronic check': 1.0, 'Mailed check': 0.7,
    'Bank transfer (automatic)': 0.2, 'Credit card (automatic)': 0.0
})
df['payment_stability'] = 1 - df['is_electronic_check'] * 0.3

# ============================================================
# SERVICE FEATURES (13)
# ============================================================
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']

df['service_count'] = df[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
df['has_security'] = (df['OnlineSecurity'] == 'Yes').astype(int)
df['has_tech_support'] = (df['TechSupport'] == 'Yes').astype(int)
df['has_backup'] = (df['OnlineBackup'] == 'Yes').astype(int)
df['has_protection'] = (df['DeviceProtection'] == 'Yes').astype(int)
df['has_internet'] = (df['InternetService'] != 'No').astype(int)
df['is_fiber_optic'] = (df['InternetService'] == 'Fiber optic').astype(int)
df['no_internet'] = (df['InternetService'] == 'No').astype(int)
df['has_phone'] = (df['PhoneService'] == 'Yes').astype(int)
df['has_multiple_lines'] = (df['MultipleLines'] == 'Yes').astype(int)
df['service_adoption_rate'] = df['service_count'] / len(service_cols)
df['no_protection'] = ((df['OnlineSecurity'] == 'No') & (df['DeviceProtection'] == 'No')).astype(int)
df['no_support'] = ((df['OnlineSecurity'] == 'No') & (df['TechSupport'] == 'No')).astype(int)

# ============================================================
# DEMOGRAPHIC FEATURES (6)
# ============================================================
df['is_senior'] = df['SeniorCitizen'].astype(int)
df['has_partner'] = (df['Partner'] == 'Yes').astype(int)
df['has_dependents'] = (df['Dependents'] == 'Yes').astype(int)
df['is_single'] = ((df['Partner'] == 'No') & (df['Dependents'] == 'No')).astype(int)
df['family_size'] = df['has_partner'] + df['has_dependents']
df['is_male'] = (df['gender'] == 'Male').astype(int)

# ============================================================
# FINANCIAL FEATURES (9)
# ============================================================
df['avg_monthly_from_total'] = df['TotalCharges'] / (df['tenure'] + 1)
df['charge_trend'] = df['MonthlyCharges'] - df['avg_monthly_from_total']
df['charge_trend_pct'] = df['charge_trend'] / (df['avg_monthly_from_total'] + 0.01)
df['is_high_charge'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)
df['total_revenue_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)
df['lifetime_value'] = df['MonthlyCharges'] * (df['tenure'] + 12)
df['charge_per_service'] = df['MonthlyCharges'] / (df['service_count'] + 1)
df['mrr_percentile'] = df['MonthlyCharges'].rank(pct=True)
df['price_sensitivity'] = df['is_high_charge'] + df['is_month_to_month']

# ============================================================
# INTERACTION FEATURES (5)
# ============================================================
df['contract_tenure_risk'] = df['contract_risk'] * df['tenure_risk']
df['payment_service_risk'] = df['payment_risk'] * (1 - df['service_adoption_rate'])
df['value_at_risk'] = df['MonthlyCharges'] * df['contract_risk']

df['engagement_score'] = (
    df['service_adoption_rate'] * 0.4 +
    (1 - df['tenure_risk']) * 0.3 +
    df['is_auto_payment'] * 0.3
)

df['churn_risk_score'] = (
    df['contract_risk'] * 0.35 +
    df['tenure_risk'] * 0.25 +
    df['payment_risk'] * 0.20 +
    (1 - df['service_adoption_rate']) * 0.10 +
    df['is_single'] * 0.10
)

# ============================================================
# ONE-HOT ENCODE REMAINING CATEGORICALS (8)
# ============================================================
df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=True)

# Drop original columns that were encoded or are non-numeric non-useful
drop_cols = ['customerID', 'Churn', 'gender', 'Partner', 'Dependents',
             'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
             'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
             'PaperlessBilling']
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# Handle NaN and infinity
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Save
output_path = BASE_DIR / 'data' / 'processed' / 'features.parquet'
df.to_parquet(output_path, index=False)
df.to_csv(BASE_DIR / 'data' / 'processed' / 'features.csv', index=False)

n_features = len(df.columns) - 1  # exclude churned
print(f"\nFeatures: {n_features}")
print(f"Records: {len(df):,}")
print(f"Churn rate: {df['churned'].mean():.1%}")
print(f"\nSaved to {output_path}")

# Save metadata
metadata = {
    'input_file': str(raw_path),
    'output_file': str(output_path),
    'n_records': len(df),
    'n_features': n_features,
    'churn_rate': round(float(df['churned'].mean()), 4),
    'feature_groups': {
        'tenure': 8,
        'contract': 4,
        'payment': 5,
        'service': 13,
        'demographic': 6,
        'financial': 9,
        'interaction': 5,
        'one_hot': n_features - 50
    },
    'created_at': datetime.now().isoformat()
}
with open(BASE_DIR / 'data' / 'processed' / 'feature_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
