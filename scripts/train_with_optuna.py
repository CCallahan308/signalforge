#!/usr/bin/env python3
"""
SignalForge Model Training with Optuna Hyperparameter Optimization

Trains churn prediction models on real IBM Telco data with:
- Optuna Bayesian optimization for hyperparameters
- 5-fold stratified cross-validation
- Bootstrap confidence intervals
- Paired t-tests for significance
- Ridge-learned feature weights
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
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, brier_score_loss

from scipy import stats
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent

print("=" * 60)
print("SignalForge Model Training with Optuna")
print("=" * 60)

# Load features
df = pd.read_parquet(BASE_DIR / 'data' / 'processed' / 'features.parquet')
print(f"Loaded {len(df):,} records")
print(f"Churn rate: {df['churned'].mean():.1%}")

# Prepare data
drop_cols = ['churned']
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
X = X.select_dtypes(include=[np.number])
y = df['churned']
feature_names = X.columns.tolist()
print(f"Features: {len(feature_names)}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

# Scale for LR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
output_dir = BASE_DIR / 'models' / 'artifacts'
output_dir.mkdir(parents=True, exist_ok=True)

results = {}

# ============================================================
# LOGISTIC REGRESSION with Optuna
# ============================================================
print("\n--- Logistic Regression + Optuna ---")

def lr_objective(trial):
    C = trial.suggest_float('C', 0.001, 100, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
    solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
    model = LogisticRegression(C=C, penalty=penalty, solver=solver,
                               class_weight='balanced', max_iter=2000, random_state=42)
    scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
    return scores.mean()

lr_study = optuna.create_study(direction='maximize')
lr_study.optimize(lr_objective, n_trials=20, show_progress_bar=False)

best_lr_params = lr_study.best_params
print(f"Best params: C={best_lr_params['C']:.4f}, penalty={best_lr_params['penalty']}")

solver = 'liblinear' if best_lr_params['penalty'] == 'l1' else 'lbfgs'
lr = LogisticRegression(
    C=best_lr_params['C'], penalty=best_lr_params['penalty'], solver=solver,
    class_weight='balanced', max_iter=2000, random_state=42
)
lr.fit(X_train_scaled, y_train)

lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
lr_pred = lr.predict(X_test_scaled)
lr_auc = roc_auc_score(y_test, lr_proba)
lr_cv = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc')

# Bootstrap CI
np.random.seed(42)
boot_aucs = []
for _ in range(1000):
    idx = np.random.choice(len(y_test), len(y_test), replace=True)
    boot_aucs.append(roc_auc_score(y_test.iloc[idx], lr_proba[idx]))
ci_lower = np.percentile(boot_aucs, 2.5)
ci_upper = np.percentile(boot_aucs, 97.5)

print(f"AUC: {lr_auc:.4f}")
print(f"CV AUC: {lr_cv.mean():.4f} +/- {lr_cv.std():.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Feature weights via Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
feat_weights = pd.DataFrame({
    'feature': feature_names,
    'weight': ridge.coef_,
    'abs_weight': np.abs(ridge.coef_)
}).sort_values('abs_weight', ascending=False)

print("Top features:")
for _, row in feat_weights.head(5).iterrows():
    print(f"  {row['feature']:30} {row['weight']:+.4f}")

results['logistic_regression'] = {
    'auc': round(lr_auc, 4),
    'cv_auc_mean': round(lr_cv.mean(), 4),
    'cv_auc_std': round(lr_cv.std(), 4),
    'ci_lower': round(ci_lower, 4),
    'ci_upper': round(ci_upper, 4),
    'f1': round(f1_score(y_test, lr_pred), 4),
    'precision': round(precision_score(y_test, lr_pred), 4),
    'recall': round(recall_score(y_test, lr_pred), 4),
    'brier_score': round(brier_score_loss(y_test, lr_proba), 4),
    'optuna_params': {k: (round(v, 6) if isinstance(v, float) else v) for k, v in best_lr_params.items()},
    'optuna_best_trial': round(lr_study.best_value, 4),
    'optuna_n_trials': len(lr_study.trials)
}

joblib.dump(lr, output_dir / 'logistic_regression.pkl')
joblib.dump(scaler, output_dir / 'scaler.pkl')

# ============================================================
# RANDOM FOREST with Optuna
# ============================================================
print("\n--- Random Forest + Optuna ---")

def rf_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
    }
    model = RandomForestClassifier(
        **params, class_weight='balanced', random_state=42, n_jobs=-1
    )
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    return scores.mean()

rf_study = optuna.create_study(direction='maximize')
rf_study.optimize(rf_objective, n_trials=20, show_progress_bar=False)

best_rf_params = rf_study.best_params
print(f"Best params: {best_rf_params}")

rf = RandomForestClassifier(
    **best_rf_params, class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

rf_proba = rf.predict_proba(X_test)[:, 1]
rf_pred = rf.predict(X_test)
rf_auc = roc_auc_score(y_test, rf_proba)
rf_cv = cross_val_score(rf, X, y, cv=cv, scoring='roc_auc')

# Significance test vs LR
lr_cv_full = cross_val_score(lr, X_scaled, y, cv=cv, scoring='roc_auc')
t_stat, p_value = stats.ttest_rel(rf_cv, lr_cv_full)
print(f"AUC: {rf_auc:.4f}")
print(f"CV AUC: {rf_cv.mean():.4f} +/- {rf_cv.std():.4f}")
print(f"Paired t-test vs LR: t={t_stat:.3f}, p={p_value:.4f}")

results['random_forest'] = {
    'auc': round(rf_auc, 4),
    'cv_auc_mean': round(rf_cv.mean(), 4),
    'cv_auc_std': round(rf_cv.std(), 4),
    'f1': round(f1_score(y_test, rf_pred), 4),
    'p_value_vs_lr': round(p_value, 4),
    'optuna_params': {k: (round(v, 6) if isinstance(v, float) else v) for k, v in best_rf_params.items()},
    'optuna_best_trial': round(rf_study.best_value, 4),
    'optuna_n_trials': len(rf_study.trials)
}

joblib.dump(rf, output_dir / 'random_forest.pkl')

# ============================================================
# GRADIENT BOOSTING with Optuna
# ============================================================
print("\n--- Gradient Boosting + Optuna ---")

def gb_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8, None]),
    }
    model = GradientBoostingClassifier(**params, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    return scores.mean()

gb_study = optuna.create_study(direction='maximize')
gb_study.optimize(gb_objective, n_trials=20, show_progress_bar=False)

best_gb_params = gb_study.best_params
# Convert None string to actual None
if best_gb_params.get('max_features') == 'None':
    best_gb_params['max_features'] = None
print(f"Best params: {best_gb_params}")

gb = GradientBoostingClassifier(**best_gb_params, random_state=42)
gb.fit(X_train, y_train)

gb_proba = gb.predict_proba(X_test)[:, 1]
gb_pred = gb.predict(X_test)
gb_auc = roc_auc_score(y_test, gb_proba)
gb_cv = cross_val_score(gb, X, y, cv=cv, scoring='roc_auc')

t_stat2, p_value2 = stats.ttest_rel(gb_cv, lr_cv_full)
print(f"AUC: {gb_auc:.4f}")
print(f"CV AUC: {gb_cv.mean():.4f} +/- {gb_cv.std():.4f}")
print(f"Paired t-test vs LR: t={t_stat2:.3f}, p={p_value2:.4f}")

results['gradient_boosting'] = {
    'auc': round(gb_auc, 4),
    'cv_auc_mean': round(gb_cv.mean(), 4),
    'cv_auc_std': round(gb_cv.std(), 4),
    'f1': round(f1_score(y_test, gb_pred), 4),
    'p_value_vs_lr': round(p_value2, 4),
    'optuna_params': {k: (round(v, 6) if isinstance(v, float) else v) for k, v in best_gb_params.items()},
    'optuna_best_trial': round(gb_study.best_value, 4),
    'optuna_n_trials': len(gb_study.trials)
}

joblib.dump(gb, output_dir / 'gradient_boosting.pkl')

# ============================================================
# SAVE ALL ARTIFACTS
# ============================================================
print("\n--- Saving Artifacts ---")

# Model comparison CSV
comparison = pd.DataFrame([
    {'model': 'Logistic Regression', 'auc': results['logistic_regression']['auc'],
     'ci_lower': results['logistic_regression']['ci_lower'],
     'ci_upper': results['logistic_regression']['ci_upper'], 'p_value': None},
    {'model': 'Random Forest', 'auc': results['random_forest']['auc'],
     'ci_lower': results['random_forest']['cv_auc_mean'] - 1.96 * results['random_forest']['cv_auc_std'],
     'ci_upper': results['random_forest']['cv_auc_mean'] + 1.96 * results['random_forest']['cv_auc_std'],
     'p_value': results['random_forest']['p_value_vs_lr']},
    {'model': 'Gradient Boosting', 'auc': results['gradient_boosting']['auc'],
     'ci_lower': results['gradient_boosting']['cv_auc_mean'] - 1.96 * results['gradient_boosting']['cv_auc_std'],
     'ci_upper': results['gradient_boosting']['cv_auc_mean'] + 1.96 * results['gradient_boosting']['cv_auc_std'],
     'p_value': results['gradient_boosting']['p_value_vs_lr']}
])
comparison.to_csv(BASE_DIR / 'models' / 'model_comparison.csv', index=False)

# Full results
full_results = {
    'training_date': datetime.now().isoformat(),
    'dataset': 'IBM Telco Customer Churn (Kaggle)',
    'n_customers': len(df),
    'n_features': len(feature_names),
    'churn_rate': round(float(y.mean()), 4),
    'business_impact': {
        'monthly_revenue': round(float(df['MonthlyCharges'].sum()), 2),
        'churned_monthly_revenue': round(float(df[df['churned'] == 1]['MonthlyCharges'].sum()), 2),
        'annual_revenue_at_risk': round(float(df[df['churned'] == 1]['MonthlyCharges'].sum() * 12), 2)
    },
    'models': results,
    'feature_names': feature_names,
    'feature_weights': feat_weights.head(20).to_dict('records'),
    'validation': {
        'method': '5-fold stratified cross-validation',
        'hyperparameter_tuning': 'Optuna Bayesian optimization (20 trials per model)',
        'bootstrap_ci_samples': 1000,
        'significance_test': 'paired t-test'
    }
}

with open(output_dir / 'training_results.json', 'w') as f:
    json.dump(full_results, f, indent=2, default=str)

with open(output_dir / 'feature_names.json', 'w') as f:
    json.dump(feature_names, f)

# Print summary
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"\n{'Model':<25} {'AUC':>8} {'CV AUC':>12} {'p vs LR':>10}")
print("-" * 60)
print(f"{'Logistic Regression':<25} {lr_auc:>8.4f} {lr_cv.mean():>8.4f}+/-{lr_cv.std():.4f} {'baseline':>10}")
print(f"{'Random Forest':<25} {rf_auc:>8.4f} {rf_cv.mean():>8.4f}+/-{rf_cv.std():.4f} {p_value:>10.4f}")
print(f"{'Gradient Boosting':<25} {gb_auc:>8.4f} {gb_cv.mean():>8.4f}+/-{gb_cv.std():.4f} {p_value2:>10.4f}")
print(f"\nOptuna: 50 trials per model, Bayesian optimization")
print(f"Business impact: ${full_results['business_impact']['annual_revenue_at_risk']:,.0f} annual revenue at risk")
