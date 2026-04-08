#!/usr/bin/env python3
"""
Process the ACTUAL IBM Telco Customer Churn dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("Loading ACTUAL IBM Telco Customer Churn dataset...")

# Load the real IBM data (from Kaggle)
# Get the signalforge directory (where this script's parent is)
script_dir = Path(__file__).parent
base_dir = script_dir.parent  # This is the signalforge root directory
raw_file = base_dir / 'data' / 'raw' / 'WA_Fn_UseC_Telco_Customer_Churn.csv'

print(f'Reading from: {raw_file}')
print(f'File exists: {raw_file.exists()}')

df = pd.read_csv(raw_file)
print(f"[OK] Loaded {len(df):,} REAL customer records from IBM")

# Clean TotalCharges (some are empty strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'], inplace=True)

# Create processed features
df_processed = df.copy()

# Encode target
df_processed['churned'] = (df['Churn'] == 'Yes').astype(int)

# Create risk scores based on actual patterns
df_processed['contract_risk'] = df['Contract'].map({
    'Month-to-month': 1.0,
    'One year': 0.5,
    'Two year': 0.0
})

df_processed['payment_risk'] = df['PaymentMethod'].map({
    'Electronic check': 1.0,
    'Mailed check': 0.5,
    'Bank transfer (automatic)': 0.0,
    'Credit card (automatic)': 0.0
})

df_processed['tenure_risk'] = 1 - (df['tenure'] / 72)

# Create composite risk score
df_processed['churn_risk_score'] = (
    0.40 * df_processed['contract_risk'] +
    0.25 * df_processed['payment_risk'] +
    0.35 * df_processed['tenure_risk']
)

# Save processed data
processed_dir = base_dir / 'data' / 'processed'
processed_dir.mkdir(parents=True, exist_ok=True)

features_csv = processed_dir / 'features.csv'
features_parquet = processed_dir / 'features.parquet'

print(f'Saving to: {features_parquet}')

df_processed.to_csv(features_csv, index=False)
df_processed.to_parquet(features_parquet, index=False)

print(f"[OK] Created features.parquet with {len(df_processed):,} records")

# Calculate REAL business impact from actual data
total_revenue = df_processed['MonthlyCharges'].sum()
churned_revenue = df_processed[df_processed['churned'] == 1]['MonthlyCharges'].sum()
annual_churned_revenue = churned_revenue * 12
churn_rate = df_processed['churned'].mean()

print(f"\n[SUCCESS] REAL data processed!")
print(f"\nReal Business Impact (from IBM data):")
print(f"  - Total customers: {len(df):,}")
print(f"  - Churn rate: {churn_rate:.1%}")
print(f"  - Total monthly revenue: ${total_revenue:,.2f}")
print(f"  - Monthly revenue at risk: ${churned_revenue:,.2f}")
print(f"  - Annual revenue at risk: ${annual_churned_revenue:,.2f}")
print(f"\nFiles:")
print(f"  - Raw data: {raw_file} ({raw_file.stat().st_size / 1024:.2f} KB)")
print(f"  - Features: {features_parquet} ({features_parquet.stat().st_size / 1024:.2f} KB)")
