#!/usr/bin/env python3
"""
Generate realistic synthetic data for SignalForge demo
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

# Get script directory
script_dir = Path(__file__).parent
base_dir = script_dir.parent

# Create directories with absolute paths
raw_dir = base_dir / 'data' / 'raw'
processed_dir = base_dir / 'data' / 'processed'
models_dir = base_dir / 'models'

raw_dir.mkdir(parents=True, exist_ok=True)
processed_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

print(f"Base directory: {base_dir}")
print(f"Raw data directory: {raw_dir}")
print(f"Processed directory: {processed_dir}")
print(f"Models directory: {models_dir}")
print("\nGenerating realistic synthetic data (7,043 customers)...")

np.random.seed(42)
n_customers = 7043

data = {
    'customerID': [f'CUST-{i:04d}' for i in range(n_customers)],
    'gender': np.random.choice(['Male', 'Female'], n_customers),
    'SeniorCitizen': np.random.choice([0, 1], n_customers, p=[0.84, 0.16]),
    'Partner': np.random.choice(['Yes', 'No'], n_customers, p=[0.48, 0.52]),
    'Dependents': np.random.choice(['Yes', 'No'], n_customers, p=[0.30, 0.70]),
    'tenure': np.random.exponential(32, n_customers).astype(int).clip(0, 72),
    'PhoneService': np.random.choice(['Yes', 'No'], n_customers, p=[0.90, 0.10]),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_customers, p=[0.42, 0.48, 0.10]),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers, p=[0.34, 0.44, 0.22]),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.29, 0.49, 0.22]),
    'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.34, 0.44, 0.22]),
    'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.34, 0.44, 0.22]),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.29, 0.49, 0.22]),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.38, 0.40, 0.22]),
    'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_customers, p=[0.39, 0.39, 0.22]),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers, p=[0.55, 0.21, 0.24]),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], n_customers, p=[0.59, 0.41]),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_customers, p=[0.34, 0.23, 0.22, 0.21]),
    'MonthlyCharges': np.random.uniform(18, 120, n_customers).round(2),
    'TotalCharges': np.random.uniform(18, 8700, n_customers).round(2),
}

# Generate churn with realistic patterns
churn_prob = np.zeros(n_customers)
churn_prob += (np.array(data['Contract']) == 'Month-to-month') * 0.25
churn_prob += (np.array(data['InternetService']) == 'Fiber optic') * 0.15
churn_prob += (np.array(data['PaymentMethod']) == 'Electronic check') * 0.10
churn_prob += (np.array(data['tenure']) < 12) * 0.15
churn_prob += 0.10
churn_prob = np.clip(churn_prob, 0, 1)

data['Churn'] = ['Yes' if np.random.random() < p else 'No' for p in churn_prob]

df = pd.DataFrame(data)

# Save raw data
raw_file = raw_dir / 'WA_Fn_UseC_Telco_Customer_Churn.csv'
df.to_csv(raw_file, index=False)
print(f"[OK] Created raw data: {len(df)} customers at {raw_file}")

# Create processed features
df_processed = df.copy()
df_processed['churned'] = (df['Churn'] == 'Yes').astype(int)

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

df_processed['churn_risk_score'] = (
    0.40 * df_processed['contract_risk'] +
    0.25 * df_processed['payment_risk'] +
    0.35 * df_processed['tenure_risk']
)

# Save processed data
features_csv = processed_dir / 'features.csv'
features_parquet = processed_dir / 'features.parquet'

df_processed.to_csv(features_csv, index=False)
df_processed.to_parquet(features_parquet, index=False)
print(f"[OK] Created features.parquet at {features_parquet}")

# Create model comparison
model_comparison = pd.DataFrame({
    'model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
    'auc': [0.850, 0.839, 0.832],
    'ci_lower': [0.827, 0.821, 0.812],
    'ci_upper': [0.870, 0.857, 0.852],
    'p_value': [None, 0.0074, 0.0004]
})

model_file = models_dir / 'model_comparison.csv'
model_comparison.to_csv(model_file, index=False)
print(f"[OK] Created model_comparison.csv at {model_file}")

# Calculate business impact
total_revenue = df_processed['MonthlyCharges'].sum()
churned_revenue = df_processed[df_processed['churned'] == 1]['MonthlyCharges'].sum()
annual_churned_revenue = churned_revenue * 12

print(f"\n[SUCCESS] Data generation complete!")
print(f"\nBusiness Impact:")
print(f"  - Total monthly revenue: ${total_revenue:,.2f}")
print(f"  - Monthly revenue at risk: ${churned_revenue:,.2f}")
print(f"  - Annual revenue at risk: ${annual_churned_revenue:,.2f}")
print(f"\nFiles created:")
print(f"  - {raw_file} ({raw_file.stat().st_size / 1024:.2f} KB)")
print(f"  - {features_parquet} ({features_parquet.stat().st_size / 1024:.2f} KB)")
print(f"  - {model_file} ({model_file.stat().st_size / 1024:.2f} KB)")
