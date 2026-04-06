#!/usr/bin/env python3
"""
Quick summary of engineered features
"""

import pandas as pd
from pathlib import Path

print("=" * 80)
print("SIGNALFORGE - FEATURE ENGINEERING SUMMARY")
print("=" * 80)

# Load features
try:
    features = pd.read_parquet('data/processed/features.parquet')
    print(f"\n[SUCCESS] Loaded engineered features")
except FileNotFoundError:
    print(f"\n[ERROR] Features not found. Run: python scripts/engineer_features.py")
    exit(1)

print(f"\n[DATASET]")
print(f"   Records: {len(features):,}")
print(f"   Total Features: {len(features.columns)}")
print(f"   Memory: {features.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# Categorize features
numeric_features = features.select_dtypes(include=['int64', 'float64', 'uint8'])
binary_features = numeric_features.loc[:, numeric_features.nunique() <= 2]
categorical_features = features.select_dtypes(include=['object', 'category'])

print(f"\n[FEATURE TYPES]")
print(f"   Numeric Features: {len(numeric_features.columns)}")
print(f"   Binary Features: {len(binary_features.columns)}")
print(f"   Categorical Features: {len(categorical_features.columns)}")
print(f"   Missing Values: {features.isnull().sum().sum()}")

# Show some engineered features
print(f"\n[SAMPLE ENGINEERED FEATURES]")
engineered_features = [
    'churn_risk_score', 'engagement_score', 'tenure_risk',
    'contract_risk_score', 'payment_risk_score', 'service_risk',
    'is_month_to_month', 'is_new_customer', 'is_high_value',
    'service_adoption_rate', 'lifetime_value', 'price_sensitivity'
]

for i, feat in enumerate(engineered_features, 1):
    if feat in features.columns:
        dtype = features[feat].dtype
        mean_val = features[feat].mean()
        print(f"   {i:2}. {feat:25} mean={mean_val:.2f} ({dtype})")

# Target variable
if 'churned' in features.columns:
    print(f"\n[TARGET VARIABLE]")
    churn_rate = features['churned'].mean()
    print(f"   Churn Rate: {churn_rate:.1%}")
    print(f"   Churned: {features['churned'].sum():,}")
    print(f"   Retained: {(1 - features['churned']).sum():,}")

# Correlation with target
if 'churned' in features.columns:
    print(f"\n[TOP CORRELATIONS WITH CHURN]")
    correlations = numeric_features.corr()['churned'].drop('churned')
    top_positive = correlations.nlargest(5)
    top_negative = correlations.nsmallest(5)
    
    print(f"\n   Positive (churn increases with feature):")
    for feat, corr in top_positive.items():
        print(f"      {feat:30} +{corr:.3f}")
    
    print(f"\n   Negative (churn decreases with feature):")
    for feat, corr in top_negative.items():
        print(f"      {feat:30} {corr:.3f}")

print(f"\n[FILES]")
output_dir = Path('data/processed')
files = list(output_dir.glob('features.*'))
for f in files:
    size_mb = f.stat().st_size / 1024**2
    print(f"   {f.name:30} {size_mb:.1f} MB")

print(f"\n[NEXT STEPS]")
print(f"   1. Train model: python scripts/train_model.py")
print(f"   2. Evaluate: python scripts/evaluate_model.py")
print(f"   3. Deploy API: uvicorn src.api.main:app --reload")

print("\n" + "=" * 80)
print("[READY] Features engineered successfully!")
print("=" * 80)
