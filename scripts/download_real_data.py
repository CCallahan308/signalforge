#!/usr/bin/env python3
"""
SignalForge Real Data Ingestion

Downloads and processes real churn datasets from Kaggle, OpenML, and other sources.

Usage:
    python scripts/download_real_data.py --source telco
    python scripts/download_real_data.py --source bank
    python scripts/download_real_data.py --source saas
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")


class RealDataIngestion:
    """Download and process real churn datasets."""
    
    DATASETS = {
        'telco': {
            'name': 'Telco Customer Churn',
            'description': 'IBM Sample Dataset - Telecom customer churn prediction',
            'kaggle': 'blastchar/telco-customer-churn',
            'rows': 7043,
            'features': ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService'],
            'target': 'Churn',
            'notes': 'Classic benchmark dataset with demographic, service, and billing info'
        },
        'bank': {
            'name': 'Bank Customer Churn',
            'description': 'Bank customer churn prediction (2024 Playground Series)',
            'kaggle': 'gauravtopre/bank-customer-churn-dataset',
            'rows': 10000,
            'features': ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance'],
            'target': 'Exited',
            'notes': 'Financial services churn with credit score and balance data'
        },
        'saas': {
            'name': 'SaaS Subscription Churn',
            'description': 'Multi-table SaaS subscription and churn analytics',
            'kaggle': 'rivalytics/saas-subscription-and-churn-analytics-dataset',
            'rows': 10000,
            'features': ['subscription_lifecycle', 'feature_usage', 'support_tickets'],
            'target': 'churn_label',
            'notes': 'Multi-table dataset with subscriptions, usage, and support data'
        },
        'cell2cell': {
            'name': 'Cell2Cell Telecom Churn',
            'description': 'Large telecom churn dataset from Duke University',
            'url': 'https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom',
            'rows': 51047,
            'features': ['MonthsInService', 'CurrentEquipmentDays', 'TotalRecurringCharge'],
            'target': 'Churn',
            'notes': 'Large-scale real telecom data with 51047 customers'
        }
    }
    
    def __init__(self, data_dir: str = 'data/raw'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_kaggle_dataset(self, dataset_key: str) -> pd.DataFrame:
        """
        Download dataset from Kaggle.
        
        Prerequisites:
            1. Install kaggle CLI: pip install kaggle
            2. Get API key from https://www.kaggle.com/settings
            3. Place kaggle.json in ~/.kaggle/
        """
        import subprocess
        
        kaggle_path = self.DATASETS[dataset_key]['kaggle']
        output_dir = self.data_dir / dataset_key
        
        logger.info(f"Downloading {self.DATASETS[dataset_key]['name']} from Kaggle...")
        logger.info(f"Dataset: {kaggle_path}")
        
        # Check if kaggle is installed
        try:
            result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True)
        except FileNotFoundError:
            logger.error("Kaggle CLI not installed!")
            logger.info("Install with: pip install kaggle")
            logger.info("Then get API key from https://www.kaggle.com/settings")
            logger.info("Place kaggle.json in ~/.kaggle/")
            return None
        
        # Download dataset
        try:
            result = subprocess.run([
                'kaggle', 'datasets', 'download',
                '-d', kaggle_path,
                '-p', str(output_dir),
                '--unzip'
            ], capture_output=True, text=True, check=True)
            
            logger.success(f"Downloaded to {output_dir}")
            
            # Find CSV files
            csv_files = list(output_dir.glob('*.csv'))
            
            if not csv_files:
                logger.error("No CSV files found in downloaded dataset")
                return None
            
            # Load main dataset
            main_csv = csv_files[0]
            logger.info(f"Loading {main_csv.name}...")
            
            df = pd.read_csv(main_csv)
            logger.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
            
            return df
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download dataset: {e.stderr}")
            return None
    
    def download_openml_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Download dataset from OpenML."""
        try:
            from sklearn.datasets import fetch_openml
        except ImportError:
            logger.error("scikit-learn not installed. Run: pip install scikit-learn")
            return None
        
        logger.info(f"Downloading {dataset_name} from OpenML...")
        
        try:
            dataset = fetch_openml(name=dataset_name, version=1, as_frame=True)
            df = dataset.frame
            
            logger.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Failed to download from OpenML: {e}")
            return None
    
    def process_telco(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Telco churn dataset to match our schema."""
        logger.info("Processing Telco dataset...")
        
        # Standardize column names
        df = df.rename(columns={
            'customerID': 'account_id',
            'tenure': 'account_age_months',
            'MonthlyCharges': 'mrr',
            'TotalCharges': 'total_revenue',
            'Churn': 'churn_label'
        })
        
        # Convert target to binary
        df['churned'] = (df['churn_label'] == 'Yes').astype(int)
        
        # Feature engineering
        df['avg_monthly_charge'] = df['total_revenue'] / (df['account_age_months'] + 1)
        df['charge_trend'] = df['mrr'] - df['avg_monthly_charge']
        
        # Contract type
        df['contract_type'] = df['Contract'].str.lower().str.replace(' ', '_')
        df['is_month_to_month'] = (df['contract_type'] == 'month-to-month').astype(int)
        
        # Services
        df['has_internet'] = (df['InternetService'] != 'No').astype(int)
        df['has_streaming'] = ((df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')).astype(int)
        df['has_security'] = ((df['OnlineSecurity'] == 'Yes') | (df['DeviceProtection'] == 'Yes')).astype(int)
        
        # Demographics
        df['is_senior'] = df['SeniorCitizen']
        df['has_partner'] = (df['Partner'] == 'Yes').astype(int)
        df['has_dependents'] = (df['Dependents'] == 'Yes').astype(int)
        
        # Payment
        df['is_automatic_payment'] = df['PaymentMethod'].str.contains('automatic', case=False).astype(int)
        df['is_paperless'] = (df['PaperlessBilling'] == 'Yes').astype(int)
        
        logger.success(f"Processed {len(df):,} records")
        logger.info(f"Churn rate: {df['churned'].mean():.1%}")
        
        return df
    
    def process_bank(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process Bank churn dataset to match our schema."""
        logger.info("Processing Bank dataset...")
        
        # Standardize column names
        df = df.rename(columns={
            'CustomerId': 'account_id',
            'CreditScore': 'credit_score',
            'Geography': 'country',
            'Gender': 'gender',
            'Age': 'age',
            'Tenure': 'tenure_years',
            'Balance': 'account_balance',
            'NumOfProducts': 'num_products',
            'HasCrCard': 'has_credit_card',
            'IsActiveMember': 'is_active',
            'EstimatedSalary': 'estimated_salary',
            'Exited': 'churned'
        })
        
        # Feature engineering
        df['balance_salary_ratio'] = df['account_balance'] / (df['estimated_salary'] + 1)
        df['products_per_tenure'] = df['num_products'] / (df['tenure_years'] + 1)
        df['is_high_value'] = (df['account_balance'] > df['account_balance'].median()).astype(int)
        df['is_engaged'] = (df['is_active'] == 1).astype(int)
        
        # Geography encoding
        df['is_germany'] = (df['country'] == 'Germany').astype(int)
        df['is_france'] = (df['country'] == 'France').astype(int)
        df['is_spain'] = (df['country'] == 'Spain').astype(int)
        
        logger.success(f"Processed {len(df):,} records")
        logger.info(f"Churn rate: {df['churned'].mean():.1%}")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, dataset_key: str):
        """Save processed dataset."""
        output_path = self.data_dir / f'{dataset_key}_processed.csv'
        df.to_csv(output_path, index=False)
        logger.success(f"Saved to {output_path}")
        
        # Save metadata
        metadata = {
            'dataset': dataset_key,
            'name': self.DATASETS[dataset_key]['name'],
            'rows': len(df),
            'columns': list(df.columns),
            'churn_rate': df['churned'].mean(),
            'features': self.DATASETS[dataset_key]['features']
        }
        
        import json
        metadata_path = self.data_dir / f'{dataset_key}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved metadata to {metadata_path}")
    
    def list_available_datasets(self):
        """List all available datasets."""
        logger.info("=" * 80)
        logger.info("Available Datasets")
        logger.info("=" * 80)
        
        for key, info in self.DATASETS.items():
            logger.info(f"\n📦 {key.upper()}: {info['name']}")
            logger.info(f"   Description: {info['description']}")
            logger.info(f"   Rows: {info['rows']:,}")
            logger.info(f"   Key Features: {', '.join(info['features'])}")
            logger.info(f"   Target: {info['target']}")
            logger.info(f"   Notes: {info['notes']}")
        
        logger.info("\n" + "=" * 80)
    
    def download_and_process(self, dataset_key: str) -> pd.DataFrame:
        """Download and process a dataset."""
        if dataset_key not in self.DATASETS:
            logger.error(f"Unknown dataset: {dataset_key}")
            self.list_available_datasets()
            return None
        
        logger.info("=" * 80)
        logger.info(f"SignalForge Real Data Ingestion: {self.DATASETS[dataset_key]['name']}")
        logger.info("=" * 80)
        
        # Download
        df = self.download_kaggle_dataset(dataset_key)
        
        if df is None:
            return None
        
        # Process
        if dataset_key == 'telco':
            df = self.process_telco(df)
        elif dataset_key == 'bank':
            df = self.process_bank(df)
        # Add more processors as needed
        
        # Save
        self.save_dataset(df, dataset_key)
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("Dataset Summary")
        logger.info("=" * 80)
        logger.info(f"Rows: {len(df):,}")
        logger.info(f"Columns: {len(df.columns)}")
        logger.info(f"Churn Rate: {df['churned'].mean():.1%}")
        logger.info(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        logger.success("\nIngestion complete!")
        
        return df


def main():
    parser = argparse.ArgumentParser(description="Download real churn datasets")
    parser.add_argument('--source', choices=['telco', 'bank', 'saas', 'cell2cell'],
                       help='Dataset to download')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--output', default='data/raw', help='Output directory')
    
    args = parser.parse_args()
    
    ingestion = RealDataIngestion(data_dir=args.output)
    
    if args.list:
        ingestion.list_available_datasets()
        return
    
    if not args.source:
        logger.error("Please specify --source or use --list")
        ingestion.list_available_datasets()
        return
    
    ingestion.download_and_process(args.source)


if __name__ == "__main__":
    main()
