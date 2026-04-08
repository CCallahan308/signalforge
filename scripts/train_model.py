#!/usr/bin/env python3
"""
SignalForge Model Training Pipeline

Trains and evaluates churn prediction models with business-focused metrics.

USAGE:
    python scripts/train_model.py

Created by: Christian G Callahan
Date: April 6, 2026

This is my first real production ML model training. Learning as I go!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve
)

import warnings
warnings.filterwarnings('ignore')

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")


class ChurnModelTrainer:
    """
    Train churn prediction models with business-focused evaluation.
    
    Approach:
    1. Start simple (baseline logistic regression)
    2. Add complexity (random forest, XGBoost)
    3. Focus on interpretability + performance
    4. Evaluate with business metrics (not just AUC)
    """
    
    def __init__(self, features_path: str = 'data/processed/features.parquet'):
        self.features_path = Path(features_path)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.feature_names = None
        
    def load_features(self):
        """Load engineered features."""
        logger.info(f"Loading features from {self.features_path}")
        
        df = pd.read_parquet(self.features_path)
        logger.success(f"Loaded {len(df):,} records × {len(df.columns)} columns")
        
        # Separate features and target
        # Drop non-numeric columns that can't be used for modeling
        drop_cols = ['churned', 'account_id', 'customerID', 'tenure_bucket', 
                     'lifecycle_stage', 'value_segment']
        
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])
        y = df['churned']
        
        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train/test sets."""
        logger.info(f"Splitting data: {1-test_size:.0%} train, {test_size:.0%} test")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.success(f"Train: {len(self.X_train):,} samples")
        logger.success(f"Test:  {len(self.X_test):,} samples")
        logger.info(f"Train churn rate: {self.y_train.mean():.1%}")
        logger.info(f"Test churn rate:  {self.y_test.mean():.1%}")
        
        return self
    
    def train_baseline_logistic(self):
        """
        Train baseline logistic regression model.
        
        Why logistic regression first?
        - Interpretable (feature coefficients)
        - Fast to train
        - Good baseline
        - If this doesn't work, complex models won't either
        """
        logger.info("=" * 80)
        logger.info("Training Baseline: Logistic Regression")
        logger.info("=" * 80)
        
        # Scale features (important for logistic regression)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Train with class weights (imbalanced data)
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        
        logger.info("Fitting model...")
        model.fit(X_train_scaled, self.y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        auc = roc_auc_score(self.y_test, y_pred_proba)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        logger.success(f"AUC:       {auc:.3f}")
        logger.success(f"Precision: {precision:.3f}")
        logger.success(f"Recall:    {recall:.3f}")
        logger.success(f"F1 Score:  {f1:.3f}")
        
        # Store results
        self.models['logistic_regression'] = {
            'model': model,
            'scaler': scaler,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Feature importance (coefficients)
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': model.coef_[0],
            'abs_coefficient': np.abs(model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        
        logger.info("\nTop 10 Most Important Features:")
        for i, row in coef_df.head(10).iterrows():
            logger.info(f"  {row['feature']:30} {row['coefficient']:+.3f}")
        
        self.results['logistic_regression'] = {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'top_features': coef_df.head(10).to_dict('records')
        }
        
        return model
    
    def train_random_forest(self):
        """
        Train random forest model.
        
        Why random forest?
        - Handles non-linear relationships
        - Less overfitting than single decision tree
        - Feature importance built-in
        - Good middle ground between simple and complex
        """
        logger.info("\n" + "=" * 80)
        logger.info("Training: Random Forest")
        logger.info("=" * 80)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        logger.info("Fitting model (this might take a minute)...")
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        auc = roc_auc_score(self.y_test, y_pred_proba)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        logger.success(f"AUC:       {auc:.3f}")
        logger.success(f"Precision: {precision:.3f}")
        logger.success(f"Recall:    {recall:.3f}")
        logger.success(f"F1 Score:  {f1:.3f}")
        
        # Store results
        self.models['random_forest'] = {
            'model': model,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Feature importance
        feat_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 Most Important Features:")
        for i, row in feat_importance.head(10).iterrows():
            logger.info(f"  {row['feature']:30} {row['importance']:.3f}")
        
        self.results['random_forest'] = {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'top_features': feat_importance.head(10).to_dict('records')
        }
        
        return model
    
    def train_gradient_boosting(self):
        """
        Train gradient boosting model (sklearn's version - simpler than XGBoost).
        
        Why gradient boosting?
        - Top performance on tabular data
        - Handles non-linear relationships
        - Usually beats random forest
        - Industry standard for many problems
        """
        logger.info("\n" + "=" * 80)
        logger.info("Training: Gradient Boosting")
        logger.info("=" * 80)
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        logger.info("Fitting model (this will take a few minutes)...")
        model.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        auc = roc_auc_score(self.y_test, y_pred_proba)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        logger.success(f"AUC:       {auc:.3f}")
        logger.success(f"Precision: {precision:.3f}")
        logger.success(f"Recall:    {recall:.3f}")
        logger.success(f"F1 Score:  {f1:.3f}")
        
        # Store results
        self.models['gradient_boosting'] = {
            'model': model,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        # Feature importance
        feat_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 Most Important Features:")
        for i, row in feat_importance.head(10).iterrows():
            logger.info(f"  {row['feature']:30} {row['importance']:.3f}")
        
        self.results['gradient_boosting'] = {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'top_features': feat_importance.head(10).to_dict('records')
        }
        
        return model
    
    def compare_models(self):
        """Compare all trained models."""
        logger.info("\n" + "=" * 80)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 80)
        
        comparison = []
        for name, results in self.results.items():
            comparison.append({
                'Model': name,
                'AUC': results['auc'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1': results['f1']
            })
        
        df_comparison = pd.DataFrame(comparison)
        df_comparison = df_comparison.sort_values('AUC', ascending=False)
        
        logger.info("\n" + df_comparison.to_string(index=False))
        
        # Best model
        best_model = df_comparison.iloc[0]
        logger.success(f"\nBest Model: {best_model['Model']}")
        logger.success(f"  AUC: {best_model['AUC']:.3f}")
        
        return df_comparison
    
    def save_results(self):
        """Save training results."""
        logger.info("\nSaving results...")
        
        output_dir = Path('models/artifacts')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison
        comparison = self.compare_models()
        comparison.to_csv(output_dir / 'model_comparison.csv', index=False)
        
        # Save detailed results
        results_path = output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.success(f"Saved to {output_dir}")
        
        return self
    
    def run(self):
        """Run full training pipeline."""
        logger.info("=" * 80)
        logger.info("SignalForge Model Training Pipeline")
        logger.info("=" * 80)
        logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")
        
        # Load and split data
        X, y = self.load_features()
        self.split_data(X, y)
        
        # Train models (simple to complex)
        logger.info("\nTraining Strategy: Start simple, add complexity")
        logger.info("If baseline doesn't work, complex won't either\n")
        
        self.train_baseline_logistic()
        self.train_random_forest()
        self.train_gradient_boosting()
        
        # Compare
        self.compare_models()
        
        # Save
        self.save_results()
        
        logger.info("\n" + "=" * 80)
        logger.success("Training Complete!")
        logger.info("=" * 80)
        logger.info(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("\nNext Steps:")
        logger.info("  1. Analyze feature importance")
        logger.info("  2. Tune hyperparameters for best model")
        logger.info("  3. Build API for predictions")
        logger.info("  4. Add business metrics (revenue at risk)")
        
        return self


def main():
    """Main entry point."""
    trainer = ChurnModelTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
