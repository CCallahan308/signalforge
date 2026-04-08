#!/usr/bin/env python3
"""
Train SignalForge models on real IBM Telco data and save artifacts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    brier_score_loss, classification_report
)

from scipy import stats

import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SignalForge Model Training")
print("=" * 60)

# Load data
base_dir = Path(__file__).parent.parent
df = pd.read_parquet(base_dir / 'data' / 'processed' / 'features.parquet')
print(f"Loaded {len(df):,} records from IBM Telco dataset")

# Prepare features
drop_cols = ['customerID', 'Churn', 'churned']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Encode categoricals
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
y = df['churned']

feature_names = X.columns.tolist()
print(f"Features: {len(feature_names)}")
print(f"Target: {y.sum()} churners ({y.mean():.1%})")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

# Scale for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create output directory
output_dir = base_dir / 'models' / 'artifacts'
output_dir.mkdir(parents=True, exist_ok=True)

results = {}

# --- Logistic Regression ---
print("\n--- Logistic Regression ---")
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
lr_pred = lr.predict(X_test_scaled)
lr_auc = roc_auc_score(y_test, lr_proba)
lr_f1 = f1_score(y_test, lr_pred)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_cv_scores = cross_val_score(lr, scaler.transform(X), y, cv=cv, scoring='roc_auc')

print(f"AUC: {lr_auc:.3f}")
print(f"F1: {lr_f1:.3f}")
print(f"5-fold CV AUC: {lr_cv_scores.mean():.3f} +/- {lr_cv_scores.std():.3f}")

# Bootstrap CI
n_bootstrap = 1000
boot_aucs = []
np.random.seed(42)
for _ in range(n_bootstrap):
    idx = np.random.choice(len(y_test), len(y_test), replace=True)
    boot_aucs.append(roc_auc_score(y_test.iloc[idx], lr_proba[idx]))
ci_lower = np.percentile(boot_aucs, 2.5)
ci_upper = np.percentile(boot_aucs, 97.5)
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

# Feature importance via Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
feat_weights = pd.DataFrame({
    'feature': feature_names,
    'weight': ridge.coef_,
    'abs_weight': np.abs(ridge.coef_)
}).sort_values('abs_weight', ascending=False)

print("\nTop features (Ridge weights):")
for _, row in feat_weights.head(5).iterrows():
    print(f"  {row['feature']:30} {row['weight']:+.4f}")

results['logistic_regression'] = {
    'auc': round(lr_auc, 4),
    'f1': round(lr_f1, 4),
    'cv_auc_mean': round(lr_cv_scores.mean(), 4),
    'cv_auc_std': round(lr_cv_scores.std(), 4),
    'ci_lower': round(ci_lower, 4),
    'ci_upper': round(ci_upper, 4),
    'precision': round(precision_score(y_test, lr_pred), 4),
    'recall': round(recall_score(y_test, lr_pred), 4),
    'brier_score': round(brier_score_loss(y_test, lr_proba), 4)
}

# Save model and scaler
joblib.dump(lr, output_dir / 'logistic_regression.pkl')
joblib.dump(scaler, output_dir / 'scaler.pkl')

# --- Random Forest ---
print("\n--- Random Forest ---")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

rf_proba = rf.predict_proba(X_test)[:, 1]
rf_pred = rf.predict(X_test)
rf_auc = roc_auc_score(y_test, rf_proba)
rf_f1 = f1_score(y_test, rf_pred)

rf_cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')
print(f"AUC: {rf_auc:.3f}")
print(f"F1: {rf_f1:.3f}")
print(f"5-fold CV AUC: {rf_cv_scores.mean():.3f} +/- {rf_cv_scores.std():.3f}")

# Paired t-test vs LR
lr_cv_all = cross_val_score(lr, scaler.transform(X), y, cv=cv, scoring='roc_auc')
t_stat, p_value = stats.ttest_rel(rf_cv_scores, lr_cv_all)
print(f"Paired t-test vs LR: t={t_stat:.3f}, p={p_value:.4f}")

results['random_forest'] = {
    'auc': round(rf_auc, 4),
    'f1': round(rf_f1, 4),
    'cv_auc_mean': round(rf_cv_scores.mean(), 4),
    'cv_auc_std': round(rf_cv_scores.std(), 4),
    'p_value_vs_lr': round(p_value, 4)
}

joblib.dump(rf, output_dir / 'random_forest.pkl')

# --- Gradient Boosting ---
print("\n--- Gradient Boosting ---")
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)

gb_proba = gb.predict_proba(X_test)[:, 1]
gb_pred = gb.predict(X_test)
gb_auc = roc_auc_score(y_test, gb_proba)
gb_f1 = f1_score(y_test, gb_pred)

gb_cv_scores = cross_val_score(gb, X, y, cv=cv, scoring='roc_auc')
print(f"AUC: {gb_auc:.3f}")
print(f"F1: {gb_f1:.3f}")
print(f"5-fold CV AUC: {gb_cv_scores.mean():.3f} +/- {gb_cv_scores.std():.3f}")

t_stat2, p_value2 = stats.ttest_rel(gb_cv_scores, lr_cv_all)
print(f"Paired t-test vs LR: t={t_stat2:.3f}, p={p_value2:.4f}")

results['gradient_boosting'] = {
    'auc': round(gb_auc, 4),
    'f1': round(gb_f1, 4),
    'cv_auc_mean': round(gb_cv_scores.mean(), 4),
    'cv_auc_std': round(gb_cv_scores.std(), 4),
    'p_value_vs_lr': round(p_value2, 4)
}

joblib.dump(gb, output_dir / 'gradient_boosting.pkl')

# --- Save all artifacts ---
print("\n--- Saving Artifacts ---")

# Model comparison CSV
comparison = pd.DataFrame([
    {
        'model': 'Logistic Regression',
        'auc': results['logistic_regression']['auc'],
        'ci_lower': results['logistic_regression']['ci_lower'],
        'ci_upper': results['logistic_regression']['ci_upper'],
        'p_value': None
    },
    {
        'model': 'Random Forest',
        'auc': results['random_forest']['auc'],
        'ci_lower': results['random_forest']['cv_auc_mean'] - 1.96 * results['random_forest']['cv_auc_std'],
        'ci_upper': results['random_forest']['cv_auc_mean'] + 1.96 * results['random_forest']['cv_auc_std'],
        'p_value': results['random_forest']['p_value_vs_lr']
    },
    {
        'model': 'Gradient Boosting',
        'auc': results['gradient_boosting']['auc'],
        'ci_lower': results['gradient_boosting']['cv_auc_mean'] - 1.96 * results['gradient_boosting']['cv_auc_std'],
        'ci_upper': results['gradient_boosting']['cv_auc_mean'] + 1.96 * results['gradient_boosting']['cv_auc_std'],
        'p_value': results['gradient_boosting']['p_value_vs_lr']
    }
])
comparison.to_csv(base_dir / 'models' / 'model_comparison.csv', index=False)
print("Saved model_comparison.csv")

# Full results JSON
full_results = {
    'training_date': datetime.now().isoformat(),
    'dataset': 'IBM Telco Customer Churn (Kaggle)',
    'n_customers': len(df),
    'n_features': len(feature_names),
    'churn_rate': round(y.mean(), 4),
    'business_impact': {
        'monthly_revenue': round(df['MonthlyCharges'].sum(), 2),
        'churned_monthly_revenue': round(df[df['churned'] == 1]['MonthlyCharges'].sum(), 2),
        'annual_revenue_at_risk': round(df[df['churned'] == 1]['MonthlyCharges'].sum() * 12, 2)
    },
    'models': results,
    'feature_names': feature_names,
    'feature_weights': feat_weights.head(20).to_dict('records'),
    'validation': {
        'method': '5-fold stratified cross-validation',
        'bootstrap_ci_samples': n_bootstrap,
        'significance_test': 'paired t-test'
    }
}

with open(output_dir / 'training_results.json', 'w') as f:
    json.dump(full_results, f, indent=2)
print("Saved training_results.json")

# Feature names for the test suite
with open(output_dir / 'feature_names.json', 'w') as f:
    json.dump(feature_names, f)
print("Saved feature_names.json")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"\nBest model by AUC: {comparison.sort_values('auc', ascending=False).iloc[0]['model']}")
print(f"Files saved to models/artifacts/:")
print(f"  - logistic_regression.pkl")
print(f"  - random_forest.pkl")
print(f"  - gradient_boosting.pkl")
print(f"  - scaler.pkl")
print(f"  - model_comparison.csv")
print(f"  - training_results.json")
print(f"  - feature_names.json")
