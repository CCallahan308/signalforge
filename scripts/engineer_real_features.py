#!/usr/bin/env python3
"""
Feature engineering that works with the ACTUAL IBM Telco dataset columns.

IBM Telco columns: customerID, gender, SeniorCitizen, Partner, Dependents,
tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity,
OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

print("=" * 60)
print("SignalForge Feature Engineering (Real IBM Telco Data)")
print("=" * 60)

# Load raw data
df = pd.read_csv('data/raw/WA_Fn_UseC_Telco_Customer_Churn.csv')
print(f"Loaded {len(df):,} records")

# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'], inplace=True)

# Encode target
df['churned'] = (df['Churn'] == 'Yes').astype(int)

# === TENURE FEATURES ===
df['tenure_years'] = df['tenure'] / 12
df['is_new_customer'] = (df['tenure'] <= 12).astype(int)
df['tenure_risk'] = (
    (df['tenure'] <= 6).astype(int) * 4 +
    ((df['tenure'] > 6) & (df['tenure'] <= 12)).astype(int) * 3 +
    ((df['tenure'] > 12) & (df['tenure'] <= 24)).astype(int) * 2 +
    ((df['tenure'] > 24) & (df['tenure'] <= 48)).astype(int) * 1
)
df['tenure_squared'] = df['tenure'] ** 2
df['log_tenure'] = np.log1p(df['tenure'])

# === CONTRACT FEATURES ===
df['is_month_to_month'] = (df['Contract'] == 'Month-to-month').astype(int)
df['is_one_year'] = (df['Contract'] == 'One year').astype(int)
df['is_two_year'] = (df['Contract'] == 'Two year').astype(int)
df['contract_risk'] = df['Contract'].map({
    'Month-to-month': 1.0, 'One year': 0.5, 'Two year': 0.0
})

# === PAYMENT FEATURES ===
df['is_auto_payment'] = df['PaymentMethod'].str.contains('automatic', case=False).astype(int)
df['is_electronic_check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
df['is_paperless'] = (df['PaperlessBilling'] == 'Yes').astype(int)
df['payment_risk'] = df['PaymentMethod'].map({
    'Electronic check': 1.0, 'Mailed check': 0.7,
    'Bank transfer (automatic)': 0.2, 'Credit card (automatic)': 0.0
})

# === SERVICE FEATURES ===
service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']

df['service_count'] = df[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
df['has_security'] = (df['OnlineSecurity'] == 'Yes').astype(int)
df['has_tech_support'] = (df['TechSupport'] == 'Yes').astype(int)
df['has_internet'] = (df['InternetService'] != 'No').astype(int)
df['is_fiber_optic'] = (df['InternetService'] == 'Fiber optic').astype(int)
df['no_internet'] = (df['InternetService'] == 'No').astype(int)
df['has_phone'] = (df['PhoneService'] == 'Yes').astype(int)
df['has_multiple_lines'] = (df['MultipleLines'] == 'Yes').astype(int)
df['service_adoption_rate'] = df['service_count'] / len(service_cols)
df['no_protection'] = ((df['OnlineSecurity'] == 'No') & (df['DeviceProtection'] == 'No')).astype(int)

# === DEMOGRAPHIC FEATURES ===
df['is_senior'] = df['SeniorCitizen'].astype(int)
df['has_partner'] = (df['Partner'] == 'Yes').astype(int)
df['has_dependents'] = (df['Dependents'] == 'Yes').astype(int)
df['is_single'] = ((df['Partner'] == 'No') & (df['Dependents'] == 'No')).astype(int)
df['family_size'] = df['has_partner'] + df['has_dependents']
df['is_male'] = (df['gender'] == 'Male').astype(int)

# === FINANCIAL FEATURES ===
df['avg_monthly_from_total'] = df['TotalCharges'] / (df['tenure'] + 1)
df['charge_trend'] = df['MonthlyCharges'] - df['avg_monthly_from_total']
df['charge_trend_pct'] = df['charge_trend'] / (df['avg_monthly_from_total'] + 0.01)
df['is_high_charge'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)
df['total_revenue_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)
df['lifetime_value'] = df['MonthlyCharges'] * (df['tenure'] + 12)
df['charge_per_service'] = df['MonthlyCharges'] / (df['service_count'] + 1)
df['mrr_percentile'] = df['MonthlyCharges'].rank(pct=True)

# === INTERACTION FEATURES ===
df['contract_tenure_risk'] = df['contract_risk'] * (1 - df['tenure'] / 72)
df['payment_service_risk'] = df['payment_risk'] * (1 - df['service_adoption_rate'])
df['value_at_risk'] = df['MonthlyCharges'] * df['contract_risk']

# Composite engagement score
df['engagement_score'] = (
    df['service_adoption_rate'] * 0.4 +
    (1 - df['tenure_risk'] / 4) * 0.3 +
    df['is_auto_payment'] * 0.3
)

# Composite churn risk
df['churn_risk_score'] = (
    df['contract_risk'] * 0.35 +
    df['tenure_risk'] / 4 * 0.25 +
    df['payment_risk'] * 0.20 +
    (1 - df['service_adoption_rate']) * 0.10 +
    df['is_single'] * 0.10
)

# === ONE-HOT ENCODE REMAINING CATEGORICALS ===
df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'], drop_first=True)

# Drop non-feature columns
drop_cols = ['customerID', 'Churn', 'gender', 'Partner', 'Dependents',
             'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
             'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
             'PaperlessBilling']
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# Handle any remaining NaN
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

print(f"Final features: {len(df.columns) - 1} (excluding target 'churned')")
print(f"Records: {len(df):,}")
print(f"Churn rate: {df['churned'].mean():.1%}")

# Save
df.to_parquet('data/processed/features.parquet', index=False)
df.to_csv('data/processed/features.csv', index=False)

# Save metadata
metadata = {
    'input_file': 'data/raw/WA_Fn_UseC_Telco_Customer_Churn.csv',
    'output_file': 'data/processed/features.parquet',
    'n_records': len(df),
    'n_features': len(df.columns) - 1,
    'churn_rate': round(df['churned'].mean(), 4),
    'feature_groups': {
        'tenure': 5,
        'contract': 4,
        'payment': 4,
        'service': 10,
        'demographic': 6,
        'financial': 7,
        'interaction': 4,
        'one_hot_encoded': '6+'
    },
    'created_at': datetime.now().isoformat()
}

with open('data/processed/feature_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved to data/processed/features.parquet")
print(f"Feature count: {len(df.columns) - 1}")
print(f"\nTop feature groups:")
print(f"  Tenure: tenure, tenure_years, is_new_customer, tenure_risk, log_tenure")
print(f"  Contract: is_month_to_month, contract_risk, is_one_year, is_two_year")
print(f"  Financial: MonthlyCharges, TotalCharges, charge_trend, lifetime_value")
print(f"  Services: service_count, has_internet, service_adoption_rate")
print(f"  Interactions: contract_tenure_risk, value_at_risk, churn_risk_score")
