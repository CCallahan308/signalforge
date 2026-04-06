#!/usr/bin/env python3
"""
Quick EDA - Generate key insights from Telco dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Load data
df = pd.read_csv('data/raw/telco_processed.csv')

print("=" * 80)
print("SIGNALFORGE - TELCO CHURN EDA SUMMARY")
print("=" * 80)

print(f"\n[DATASET OVERVIEW]")
print(f"   Total Customers: {len(df):,}")
print(f"   Features: {len(df.columns)}")
print(f"   Churn Rate: {df['churned'].mean():.1%}")
print(f"   Churned: {df['churned'].sum():,} customers")
print(f"   Retained: {(1 - df['churned']).sum():,} customers")

print(f"\n[REVENUE IMPACT]")
total_mrr = df['mrr'].sum()
mrr_at_risk = df[df['churned'] == 1]['mrr'].sum()
print(f"   Total Monthly Revenue: ${total_mrr:,.2f}")
print(f"   Revenue at Risk (Churned): ${mrr_at_risk:,.2f} ({mrr_at_risk/total_mrr:.1%})")
print(f"   Annual Revenue at Risk: ${mrr_at_risk * 12:,.2f}")

print(f"\n[TOP CHURN DRIVERS]")
print(f"\n1. CONTRACT TYPE (Most Important)")
contract_churn = df.groupby('Contract')['churned'].agg(['mean', 'count'])
for contract in contract_churn.index:
    rate = contract_churn.loc[contract, 'mean']
    count = contract_churn.loc[contract, 'count']
    print(f"   {contract:20} {rate:.1%} churn ({count:,} customers)")

print(f"\n2. TENURE (Account Age)")
df['tenure_bucket'] = pd.cut(df['account_age_months'], 
                             bins=[0, 12, 24, 48, 72],
                             labels=['0-12 mo', '13-24 mo', '25-48 mo', '49-72 mo'])
tenure_churn = df.groupby('tenure_bucket')['churned'].agg(['mean', 'count'])
for bucket in tenure_churn.index:
    rate = tenure_churn.loc[bucket, 'mean']
    count = tenure_churn.loc[bucket, 'count']
    print(f"   {bucket:20} {rate:.1%} churn ({count:,} customers)")

print(f"\n3. PAYMENT METHOD")
payment_churn = df.groupby('PaymentMethod')['churned'].agg(['mean', 'count'])
payment_churn = payment_churn.sort_values('mean', ascending=False)
for payment in payment_churn.index[:3]:
    rate = payment_churn.loc[payment, 'mean']
    count = payment_churn.loc[payment, 'count']
    print(f"   {payment:40} {rate:.1%} churn ({count:,} customers)")

print(f"\n[HIGH-VALUE SEGMENTS]")
hv_threshold = df['mrr'].quantile(0.75)
df['is_high_value'] = (df['mrr'] >= hv_threshold).astype(int)
hv_churn = df.groupby('is_high_value')['churned'].agg(['mean', 'count'])

print(f"   High-Value Threshold: ${hv_threshold:.2f}/month (top 25%)")
print(f"   Standard Customers: {hv_churn.loc[0, 'mean']:.1%} churn ({hv_churn.loc[0, 'count']:,} customers)")
print(f"   High-Value Customers: {hv_churn.loc[1, 'mean']:.1%} churn ({hv_churn.loc[1, 'count']:,} customers)")

hv_mrr_at_risk = df[(df['churned'] == 1) & (df['is_high_value'] == 1)]['mrr'].sum()
print(f"   High-Value Revenue at Risk: ${hv_mrr_at_risk:,.2f}/month")

print(f"\n[DEMOGRAPHICS]")
demo_features = [
    ('gender', 'Gender'),
    ('SeniorCitizen', 'Senior Citizen'),
    ('Partner', 'Has Partner'),
    ('Dependents', 'Has Dependents')
]

for feature, label in demo_features:
    churn_by_demo = df.groupby(feature)['churned'].mean()
    print(f"\n   {label}:")
    for val in churn_by_demo.index:
        rate = churn_by_demo[val]
        print(f"     {val:15} {rate:.1%} churn")

print(f"\n[NUMERICAL FEATURES - Churned vs Retained]")
print(f"\n   Tenure (months):")
print(f"     Retained: {df[df['churned']==0]['account_age_months'].mean():.1f} avg")
print(f"     Churned:  {df[df['churned']==1]['account_age_months'].mean():.1f} avg")

print(f"\n   Monthly Charges:")
print(f"     Retained: ${df[df['churned']==0]['mrr'].mean():.2f} avg")
print(f"     Churned:  ${df[df['churned']==1]['mrr'].mean():.2f} avg")

print(f"\n   Total Charges:")
print(f"     Retained: ${df[df['churned']==0]['total_revenue'].mean():,.2f} avg")
print(f"     Churned:  ${df[df['churned']==1]['total_revenue'].mean():,.2f} avg")

print(f"\n[KEY INSIGHTS FOR MODELING]")
print(f"\n1. TARGET VARIABLE")
print(f"   - Churn rate: 26.5% (moderate class imbalance)")
print(f"   - Use: class_weight='balanced' or SMOTE")

print(f"\n2. TOP PREDICTIVE FEATURES (based on EDA)")
print(f"   - Contract Type (strongest signal)")
print(f"   - Tenure (0-12 months = high risk)")
print(f"   - Payment Method (electronic check = high risk)")
print(f"   - Services (no security/support = high risk)")

print(f"\n3. EXPECTED MODEL PERFORMANCE")
print(f"   - Baseline (always predict majority): 73.5% accuracy")
print(f"   - Target AUC: 0.80-0.85 (achievable)")
print(f"   - Business metric: Revenue saved / intervention cost")

print(f"\n4. RETENTION STRATEGIES")
print(f"   - Target month-to-month customers with annual discounts")
print(f"   - Onboarding focus for 0-12 month customers")
print(f"   - Migrate electronic check to auto-payment")
print(f"   - Bundle security/tech support services")

# Correlation analysis
print(f"\n[CORRELATIONS WITH CHURN]")
numerical_features = ['account_age_months', 'mrr', 'total_revenue', 
                      'MonthlyCharges', 'TotalCharges', 'tenure']
print(f"\n   Top correlations:")
for feature in numerical_features:
    if feature in df.columns:
        corr = df[feature].corr(df['churned'])
        print(f"   {feature:20} {corr:+.3f}")

print("\n" + "=" * 80)
print("[SUCCESS] EDA COMPLETE - Ready for feature engineering and modeling")
print("=" * 80)

# Save findings
import json
findings = {
    'total_customers': len(df),
    'churn_rate': float(df['churned'].mean()),
    'monthly_revenue_at_risk': float(mrr_at_risk),
    'annual_revenue_at_risk': float(mrr_at_risk * 12),
    'high_value_threshold': float(hv_threshold),
    'contract_churn_rates': {
        'month_to_month': float(df[df['Contract'] == 'Month-to-month']['churned'].mean()),
        'one_year': float(df[df['Contract'] == 'One year']['churned'].mean()),
        'two_year': float(df[df['Contract'] == 'Two year']['churned'].mean())
    },
    'tenure_churn_rates': {
        '0_12_months': float(df[df['tenure_bucket'] == '0-12 mo']['churned'].mean()),
        '13_24_months': float(df[df['tenure_bucket'] == '13-24 mo']['churned'].mean()),
        '25_48_months': float(df[df['tenure_bucket'] == '25-48 mo']['churned'].mean()),
        '49_72_months': float(df[df['tenure_bucket'] == '49-72 mo']['churned'].mean())
    }
}

Path('reports').mkdir(exist_ok=True)
with open('reports/eda_findings.json', 'w') as f:
    json.dump(findings, f, indent=2)

print(f"\nSaved: reports/eda_findings.json")
print(f"Notebook: notebooks/01_eda_telco.ipynb")
