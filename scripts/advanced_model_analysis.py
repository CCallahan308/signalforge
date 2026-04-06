#!/usr/bin/env python3
"""
SignalForge Advanced Model Analysis

Adds statistical rigor:
- Bootstrap confidence intervals
- Cross-validation
- Statistical tests for model comparison
- Calibration analysis
- Feature weight learning

USAGE:
    python scripts/advanced_model_analysis.py

Created by: Christian G Callahan
Date: April 6, 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import json

from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy.stats import ttest_rel, wilcoxon
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")


class AdvancedModelAnalysis:
    """
    Model analysis with statistical rigor.
    
    Improvements over basic training:
    1. Bootstrap confidence intervals on all metrics
    2. Cross-validation for robust estimates
    3. Statistical tests for model comparison
    4. Calibration analysis
    5. Learn feature weights instead of hard-coding
    """
    
    def __init__(self, features_path: str = 'data/processed/features.parquet'):
        self.features_path = Path(features_path)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.cv_results = {}
        
    def load_features(self):
        """Load engineered features."""
        logger.info(f"Loading features from {self.features_path}")
        
        df = pd.read_parquet(self.features_path)
        
        # Separate features and target
        drop_cols = ['churned', 'account_id', 'customerID', 'tenure_bucket', 
                     'lifecycle_stage', 'value_segment']
        
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])
        y = df['churned']
        
        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])
        
        logger.success(f"Loaded {len(X):,} samples × {len(X.columns)} features")
        
        self.X = X
        self.y = y
        
        return X, y
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train/test sets."""
        logger.info(f"Splitting data: {1-test_size:.0%} train, {test_size:.0%} test")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        logger.success(f"Train: {len(self.X_train):,} samples")
        logger.success(f"Test:  {len(self.X_test):,} samples")
        
        return self
    
    def compute_bootstrap_ci(self, y_true, y_pred_proba, metric_func, 
                             n_bootstrap=1000, alpha=0.05):
        """
        Compute bootstrap confidence interval for any metric.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities
        metric_func : callable
            Function that takes (y_true, y_pred_proba) and returns metric
        n_bootstrap : int
            Number of bootstrap samples
        alpha : float
            Significance level (0.05 for 95% CI)
        
        Returns:
        --------
        dict with 'mean', 'std', 'ci_lower', 'ci_upper', 'all_scores'
        """
        logger.info(f"Computing bootstrap CI with {n_bootstrap} samples...")
        
        scores = []
        n_samples = len(y_true)
        
        for i in range(n_bootstrap):
            # Resample with replacement
            idx = np.random.choice(n_samples, n_samples, replace=True)
            y_true_sample = np.array(y_true)[idx]
            y_pred_sample = np.array(y_pred_proba)[idx]
            
            # Compute metric
            try:
                score = metric_func(y_true_sample, y_pred_sample)
                scores.append(score)
            except Exception as e:
                # Skip if metric computation fails
                continue
            
            # Progress indicator
            if (i + 1) % 200 == 0:
                logger.info(f"  Progress: {i+1}/{n_bootstrap}")
        
        scores = np.array(scores)
        
        result = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'ci_lower': np.percentile(scores, 100 * alpha / 2),
            'ci_upper': np.percentile(scores, 100 * (1 - alpha / 2)),
            'all_scores': scores.tolist()
        }
        
        logger.success(f"Mean: {result['mean']:.3f}, 95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
        
        return result
    
    def cross_validate_model(self, model, X, y, cv=5, scoring='roc_auc'):
        """
        Perform cross-validation with detailed results.
        
        Parameters:
        -----------
        model : estimator
            Sklearn model
        X : DataFrame
            Features
        y : Series
            Target
        cv : int
            Number of folds
        scoring : str
            Scoring metric
        
        Returns:
        --------
        dict with 'mean', 'std', 'scores'
        """
        logger.info(f"Running {cv}-fold cross-validation...")
        
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring, n_jobs=-1)
        
        result = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
        
        logger.success(f"CV Score: {result['mean']:.3f} ± {result['std']:.3f}")
        
        return result
    
    def compare_models_statistically(self, scores_a, scores_b, model_a_name, model_b_name):
        """
        Compare two models using paired t-test and Wilcoxon signed-rank test.
        
        Parameters:
        -----------
        scores_a : array-like
            CV scores from model A
        scores_b : array-like
            CV scores from model B
        model_a_name : str
            Name of model A
        model_b_name : str
            Name of model B
        
        Returns:
        --------
        dict with test results
        """
        logger.info(f"Comparing {model_a_name} vs {model_b_name}...")
        
        # Paired t-test (parametric)
        t_stat, p_value_ttest = ttest_rel(scores_a, scores_b)
        
        # Wilcoxon signed-rank test (non-parametric)
        try:
            stat_wilcoxon, p_value_wilcoxon = wilcoxon(scores_a, scores_b)
        except Exception as e:
            logger.warning(f"Wilcoxon test failed: {e}")
            stat_wilcoxon, p_value_wilcoxon = None, None
        
        result = {
            'model_a': model_a_name,
            'model_b': model_b_name,
            'mean_a': np.mean(scores_a),
            'mean_b': np.mean(scores_b),
            'diff': np.mean(scores_a) - np.mean(scores_b),
            't_statistic': t_stat,
            'p_value_ttest': p_value_ttest,
            'significant_ttest': p_value_ttest < 0.05,
            'p_value_wilcoxon': p_value_wilcoxon,
            'significant_wilcoxon': p_value_wilcoxon < 0.05 if p_value_wilcoxon else None
        }
        
        logger.info(f"  {model_a_name}: {result['mean_a']:.3f}")
        logger.info(f"  {model_b_name}: {result['mean_b']:.3f}")
        logger.info(f"  Difference: {result['diff']:.3f}")
        logger.info(f"  t-test p-value: {p_value_ttest:.4f} {'*' if result['significant_ttest'] else ''}")
        if p_value_wilcoxon:
            logger.info(f"  Wilcoxon p-value: {p_value_wilcoxon:.4f} {'*' if result['significant_wilcoxon'] else ''}")
        
        return result
    
    def analyze_calibration(self, y_true, y_pred_proba, model_name, n_bins=10):
        """
        Analyze model calibration (predicted vs actual probabilities).
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities
        model_name : str
            Name of model
        n_bins : int
            Number of bins for calibration curve
        
        Returns:
        --------
        dict with calibration metrics
        """
        logger.info(f"Analyzing calibration for {model_name}...")
        
        # Brier score (lower is better)
        brier = brier_score_loss(y_true, y_pred_proba)
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
        
        # Expected Calibration Error (ECE)
        # Weighted average of |predicted - actual| per bin
        bin_counts = np.histogram(y_pred_proba, bins=n_bins, range=(0, 1))[0]
        bin_weights = bin_counts / len(y_pred_proba)
        
        # Get bin edges
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Compute calibration error
        calib_error = np.sum(bin_weights * np.abs(prob_true - prob_pred))
        
        result = {
            'model_name': model_name,
            'brier_score': brier,
            'ece': calib_error,  # Expected Calibration Error
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist(),
            'bin_weights': bin_weights.tolist()
        }
        
        logger.success(f"Brier Score: {brier:.3f}, ECE: {calib_error:.3f}")
        
        return result
    
    def learn_feature_weights(self, feature_names, target_feature='churn_risk_score'):
        """
        Learn optimal weights for composite features using regularization.
        
        Instead of hard-coding weights, learn them from data.
        
        Parameters:
        -----------
        feature_names : list
            Names of features to combine
        target_feature : str
            Name of target feature to predict (not the churn target, but a composite)
        
        Returns:
        --------
        dict with learned weights
        """
        logger.info("Learning feature weights with regularization...")
        
        # Select risk features
        available_features = [f for f in feature_names if f in self.X_train.columns]
        
        if len(available_features) < 2:
            logger.warning("Not enough features for weight learning")
            return None
        
        X_risk = self.X_train[available_features]
        
        # Standardize features
        scaler = StandardScaler()
        X_risk_scaled = scaler.fit_transform(X_risk)
        
        # Learn weights with Ridge regression (L2 regularization)
        # Alpha controls regularization strength (higher = more regularization)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_risk_scaled, self.y_train)
        
        weights = dict(zip(available_features, ridge.coef_))
        
        # Sort by absolute value
        sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
        
        logger.success("Learned feature weights:")
        for feat, weight in sorted_weights:
            logger.info(f"  {feat:30} {weight:+.3f}")
        
        return {
            'weights': weights,
            'sorted_weights': sorted_weights,
            'regularization_alpha': 1.0,
            'r2_score': ridge.score(X_risk_scaled, self.y_train)
        }
    
    def run_full_analysis(self):
        """Run complete advanced analysis."""
        logger.info("=" * 80)
        logger.info("SignalForge Advanced Model Analysis")
        logger.info("=" * 80)
        logger.info("Adding statistical rigor: Bootstrap CI, CV, Statistical Tests")
        logger.info("")
        
        # Load data
        X, y = self.load_features()
        self.split_data()
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        # Store results
        all_results = {}
        
        # ========================================
        # 1. CROSS-VALIDATION
        # ========================================
        logger.info("\n" + "=" * 80)
        logger.info("1. CROSS-VALIDATION ANALYSIS")
        logger.info("=" * 80)
        
        cv_results = {}
        
        for name, model in models.items():
            logger.info(f"\n{name}:")
            
            if 'Logistic' in name:
                X_cv = X_train_scaled
            else:
                X_cv = self.X_train
            
            cv_result = self.cross_validate_model(model, X_cv, self.y_train, cv=5)
            cv_results[name] = cv_result
        
        all_results['cross_validation'] = cv_results
        
        # ========================================
        # 2. STATISTICAL COMPARISON
        # ========================================
        logger.info("\n" + "=" * 80)
        logger.info("2. STATISTICAL COMPARISON")
        logger.info("=" * 80)
        
        model_names = list(cv_results.keys())
        comparisons = []
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                comp = self.compare_models_statistically(
                    cv_results[model_names[i]]['scores'],
                    cv_results[model_names[j]]['scores'],
                    model_names[i],
                    model_names[j]
                )
                comparisons.append(comp)
        
        all_results['statistical_comparisons'] = comparisons
        
        # ========================================
        # 3. TRAIN MODELS & BOOTSTRAP CI
        # ========================================
        logger.info("\n" + "=" * 80)
        logger.info("3. BOOTSTRAP CONFIDENCE INTERVALS")
        logger.info("=" * 80)
        
        test_results = {}
        
        for name, model in models.items():
            logger.info(f"\n{name}:")
            
            if 'Logistic' in name:
                model.fit(X_train_scaled, self.y_train)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(self.X_train, self.y_train)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Bootstrap CI for AUC
            logger.info("  Computing AUC bootstrap CI...")
            auc_ci = self.compute_bootstrap_ci(
                self.y_test, y_pred_proba, roc_auc_score, n_bootstrap=1000
            )
            
            # Bootstrap CI for Precision
            logger.info("  Computing Precision bootstrap CI...")
            y_pred = (y_pred_proba >= 0.5).astype(int)
            precision_ci = self.compute_bootstrap_ci(
                self.y_test, y_pred, lambda y, p: precision_score(y, p, zero_division=0),
                n_bootstrap=1000
            )
            
            # Bootstrap CI for Recall
            logger.info("  Computing Recall bootstrap CI...")
            recall_ci = self.compute_bootstrap_ci(
                self.y_test, y_pred, lambda y, p: recall_score(y, p, zero_division=0),
                n_bootstrap=1000
            )
            
            test_results[name] = {
                'auc_ci': auc_ci,
                'precision_ci': precision_ci,
                'recall_ci': recall_ci
            }
        
        all_results['test_with_ci'] = test_results
        
        # ========================================
        # 4. CALIBRATION ANALYSIS
        # ========================================
        logger.info("\n" + "=" * 80)
        logger.info("4. CALIBRATION ANALYSIS")
        logger.info("=" * 80)
        
        calibration_results = {}
        
        for name, model in models.items():
            if 'Logistic' in name:
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            calib = self.analyze_calibration(self.y_test, y_pred_proba, name)
            calibration_results[name] = calib
        
        all_results['calibration'] = calibration_results
        
        # ========================================
        # 5. LEARN FEATURE WEIGHTS
        # ========================================
        logger.info("\n" + "=" * 80)
        logger.info("5. LEARNED FEATURE WEIGHTS")
        logger.info("=" * 80)
        
        risk_features = [
            'contract_risk_score', 'tenure_risk', 'payment_risk_score',
            'service_risk', 'demographic_risk'
        ]
        
        learned_weights = self.learn_feature_weights(risk_features)
        all_results['learned_weights'] = learned_weights
        
        # ========================================
        # SAVE RESULTS
        # ========================================
        logger.info("\n" + "=" * 80)
        logger.info("SAVING RESULTS")
        logger.info("=" * 80)
        
        output_dir = Path('models/artifacts')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / 'advanced_analysis_results.json'
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.success(f"Saved to {output_path}")
        
        # ========================================
        # SUMMARY
        # ========================================
        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 80)
        
        logger.info("\n📊 Cross-Validation Results:")
        for name, result in cv_results.items():
            logger.info(f"   {name:25} {result['mean']:.3f} ± {result['std']:.3f}")
        
        logger.info("\n🔬 Statistical Comparisons:")
        for comp in comparisons:
            sig = "✓ SIGNIFICANT" if comp['significant_ttest'] else "✗ Not significant"
            logger.info(f"   {comp['model_a']} vs {comp['model_b']}: p={comp['p_value_ttest']:.4f} {sig}")
        
        logger.info("\n📈 Test Set with 95% CI:")
        for name, result in test_results.items():
            auc = result['auc_ci']
            logger.info(f"   {name:25} AUC={auc['mean']:.3f} [{auc['ci_lower']:.3f}, {auc['ci_upper']:.3f}]")
        
        logger.info("\n🎯 Calibration (Brier Score, lower is better):")
        for name, result in calibration_results.items():
            logger.info(f"   {name:25} Brier={result['brier_score']:.3f}, ECE={result['ece']:.3f}")
        
        logger.success("\n✅ Advanced analysis complete!")
        logger.info("\nKey improvements:")
        logger.info("   ✓ Bootstrap confidence intervals on all metrics")
        logger.info("   ✓ Cross-validation for robust estimates")
        logger.info("   ✓ Statistical tests for model comparison")
        logger.info("   ✓ Calibration analysis")
        logger.info("   ✓ Learned feature weights (not hard-coded)")
        
        return all_results


def main():
    """Run advanced analysis."""
    analysis = AdvancedModelAnalysis()
    results = analysis.run_full_analysis()


if __name__ == "__main__":
    main()
