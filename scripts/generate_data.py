#!/usr/bin/env python3
"""
SignalForge Synthetic Data Generator

Generates realistic SaaS customer data for churn prediction modeling.

Usage:
    python scripts/generate_data.py --accounts 10000 --months 24
"""

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")


class SyntheticDataGenerator:
    """Generate realistic SaaS customer data."""
    
    # Configuration
    INDUSTRIES = [
        'Technology', 'Healthcare', 'Finance', 'E-commerce', 'Education',
        'Manufacturing', 'Retail', 'Media', 'Consulting', 'Real Estate'
    ]
    
    COMPANY_SIZES = ['1-10', '11-50', '51-200', '201-500', '500+']
    
    PLAN_TYPES = ['free', 'starter', 'pro', 'enterprise']
    PLAN_MRR = {'free': 0, 'starter': 49, 'pro': 199, 'enterprise': 999}
    
    EVENTS = {
        'engagement': ['login', 'feature_used', 'api_call', 'export', 'import'],
        'product': ['create_project', 'add_member', 'upgrade_feature', 'use_integration'],
        'billing': ['view_invoice', 'update_payment', 'upgrade_plan', 'downgrade_plan'],
        'support': ['contact_support', 'view_docs', 'watch_tutorial']
    }
    
    def __init__(self, n_accounts: int = 10000, n_months: int = 24, seed: int = 42):
        self.n_accounts = n_accounts
        self.n_months = n_months
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.end_date = datetime.now().replace(day=1)
        self.start_date = self.end_date - timedelta(days=30 * n_months)
        
        logger.info(f"Generating data for {n_accounts:,} accounts over {n_months} months")
    
    def generate_accounts(self) -> pd.DataFrame:
        """Generate account data."""
        logger.info("Generating accounts...")
        
        accounts = []
        for i in range(self.n_accounts):
            signup_date = self.start_date + timedelta(
                days=random.randint(0, int((self.end_date - self.start_date).days * 0.8))
            )
            
            plan = random.choices(
                self.PLAN_TYPES,
                weights=[0.3, 0.35, 0.25, 0.1]
            )[0]
            
            # Enterprise accounts have custom MRR
            if plan == 'enterprise':
                mrr = random.randint(500, 5000)
            else:
                mrr = self.PLAN_MRR[plan]
            
            account = {
                'account_id': f'acc_{i:06d}',
                'email': f'company{i}@example.com',
                'company_name': f'Company {i}',
                'industry': random.choice(self.INDUSTRIES),
                'company_size': random.choices(
                    self.COMPANY_SIZES,
                    weights=[0.2, 0.3, 0.25, 0.15, 0.1]
                )[0],
                'plan_type': plan,
                'billing_frequency': random.choice(['monthly', 'annual']),
                'mrr': mrr,
                'contract_start_date': signup_date.date(),
                'contract_end_date': (signup_date + timedelta(days=365)).date() if random.random() > 0.3 else None,
                'is_active': True,
                'created_at': signup_date
            }
            
            accounts.append(account)
        
        df = pd.DataFrame(accounts)
        logger.success(f"Generated {len(df):,} accounts")
        return df
    
    def generate_churn_labels(self, accounts: pd.DataFrame) -> pd.DataFrame:
        """Generate churn labels based on behavior patterns."""
        logger.info("Generating churn labels...")
        
        # Assign churn probability based on features
        accounts['churn_base_prob'] = np.random.beta(2, 5, len(accounts))  # Skewed towards lower churn
        
        # Adjust based on plan type
        plan_mult = {'free': 2.0, 'starter': 1.5, 'pro': 1.0, 'enterprise': 0.5}
        accounts['churn_base_prob'] *= accounts['plan_type'].map(plan_mult)
        
        # Adjust based on company size
        size_mult = {'1-10': 1.5, '11-50': 1.2, '51-200': 1.0, '201-500': 0.8, '500+': 0.6}
        accounts['churn_base_prob'] *= accounts['company_size'].map(size_mult)
        
        # Adjust based on industry
        industry_mult = {
            'Technology': 1.0, 'Healthcare': 0.8, 'Finance': 0.9, 'E-commerce': 1.2,
            'Education': 1.1, 'Manufacturing': 1.0, 'Retail': 1.3, 'Media': 1.4,
            'Consulting': 0.9, 'Real Estate': 1.1
        }
        accounts['churn_base_prob'] *= accounts['industry'].map(industry_mult)
        
        # Cap probabilities
        accounts['churn_base_prob'] = accounts['churn_base_prob'].clip(0, 1)
        
        # Generate actual churn events
        accounts['will_churn'] = (np.random.random(len(accounts)) < accounts['churn_base_prob'])
        
        # For those who churn, assign churn date
        def get_churn_date(row):
            if not row['will_churn']:
                return None
            signup = row['created_at']
            max_days = (self.end_date - signup).days
            churn_after_days = random.randint(30, max(max_days, 31))
            return (signup + timedelta(days=churn_after_days)).date()
        
        accounts['churn_date'] = accounts.apply(get_churn_date, axis=1)
        
        logger.info(f"Churn rate: {accounts['will_churn'].mean():.1%}")
        return accounts
    
    def generate_events(self, accounts: pd.DataFrame) -> pd.DataFrame:
        """Generate event data for all accounts."""
        logger.info("Generating events (this may take a while)...")
        
        events = []
        event_id = 0
        
        for _, account in accounts.iterrows():
            # Determine activity level for this account
            activity_level = np.random.lognormal(mean=2, sigma=0.5)
            
            # Generate events from signup to churn or end date
            end = account['churn_date'] if account['will_churn'] else self.end_date.date()
            current = account['created_at'].date()
            
            while current < end:
                # Daily event probability (decreases over time, increases before churn)
                days_since_signup = (current - account['created_at'].date()).days
                days_to_churn = (end - current).days if account['will_churn'] else 9999
                
                base_prob = 0.7 * np.exp(-days_since_signup / 365)
                if days_to_churn < 30:
                    base_prob *= 0.5  # Decrease activity before churn
                
                if random.random() < base_prob:
                    # Generate 1-20 events per active day
                    n_events = max(1, int(np.random.poisson(activity_level)))
                    
                    for _ in range(n_events):
                        category = random.choices(
                            list(self.EVENTS.keys()),
                            weights=[0.5, 0.25, 0.15, 0.1]
                        )[0]
                        
                        event_type = random.choice(self.EVENTS[category])
                        
                        event = {
                            'event_id': f'evt_{event_id:08d}',
                            'account_id': account['account_id'],
                            'event_type': event_type,
                            'event_category': category,
                            'properties': '{}',
                            'occurred_at': datetime.combine(current, datetime.min.time()) + timedelta(
                                hours=random.randint(0, 23),
                                minutes=random.randint(0, 59)
                            )
                        }
                        
                        events.append(event)
                        event_id += 1
                
                current += timedelta(days=1)
                
                # Progress indicator
                if event_id % 100000 == 0 and event_id > 0:
                    logger.info(f"  Generated {event_id:,} events...")
        
        df = pd.DataFrame(events)
        logger.success(f"Generated {len(df):,} events")
        return df
    
    def generate_features(self, accounts: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """Generate daily feature snapshots for each account."""
        logger.info("Generating features...")
        
        features = []
        
        # Get event counts per account per day
        events['date'] = events['occurred_at'].dt.date
        event_counts = events.groupby(['account_id', 'date']).agg({
            'event_id': 'count',
            'event_type': lambda x: x.nunique()
        }).reset_index()
        event_counts.columns = ['account_id', 'date', 'total_events', 'unique_event_types']
        
        snapshot_id = 0
        
        for _, account in accounts.iterrows():
            # Get account events
            account_events = event_counts[event_counts['account_id'] == account['account_id']]
            
            end = account['churn_date'] if account['will_churn'] else self.end_date.date()
            current = account['created_at'].date()
            
            # Generate monthly snapshots
            while current <= end:
                if current.day == 1:  # Monthly snapshot
                    # Calculate features
                    account_age_days = (current - account['created_at'].date()).days
                    
                    # Get events from last 30 days
                    recent_events = account_events[
                        (account_events['date'] >= current - timedelta(days=30)) &
                        (account_events['date'] < current)
                    ]
                    
                    feature = {
                        'snapshot_date': current,
                        'account_id': account['account_id'],
                        'account_age_days': account_age_days,
                        'months_active': account_age_days // 30,
                        'daily_active_users': random.randint(1, 10) if len(recent_events) > 0 else 0,
                        'weekly_active_users': random.randint(1, 15) if len(recent_events) > 0 else 0,
                        'monthly_active_users': random.randint(1, 20) if len(recent_events) > 0 else 0,
                        'total_events_7d': recent_events['total_events'].tail(7).sum() if len(recent_events) >= 7 else recent_events['total_events'].sum(),
                        'total_events_30d': recent_events['total_events'].sum() if len(recent_events) > 0 else 0,
                        'avg_events_per_user_7d': random.uniform(5, 50),
                        'login_frequency_7d': random.uniform(0.1, 1.0),
                        'feature_adoption_rate': random.uniform(0.2, 1.0),
                        'core_feature_usage_rate': random.uniform(0.3, 1.0),
                        'session_duration_avg_7d': random.uniform(60, 3600),
                        'open_tickets': random.randint(0, 3),
                        'tickets_30d': random.randint(0, 5),
                        'avg_resolution_time_hours': random.uniform(1, 72),
                        'avg_sentiment_score': random.uniform(-0.5, 0.8),
                        'mrr': account['mrr'],
                        'mrr_change_1m': random.uniform(-100, 100),
                        'mrr_change_3m': random.uniform(-300, 300),
                        'failed_payments_6m': random.randint(0, 2),
                        'days_since_last_payment': random.randint(0, 90),
                        'payment_method_failures': random.randint(0, 2),
                        'nps_score': random.randint(0, 10),
                        'days_since_nps_survey': random.randint(0, 180),
                        'nps_feedback_length': random.randint(0, 500),
                        'days_to_renewal': random.randint(0, 365),
                        'contract_value': account['mrr'] * 12,
                        'is_annual': account['billing_frequency'] == 'annual',
                        'interventions_90d': random.randint(0, 2),
                        'last_intervention_days': random.randint(0, 90),
                        'last_intervention_outcome': random.choice(['saved', 'churned', 'pending', 'no_response', None]),
                        'churned_next_30d': False  # Will be set later
                    }
                    
                    features.append(feature)
                    snapshot_id += 1
                
                current += timedelta(days=1)
        
        df = pd.DataFrame(features)
        
        # Set churn labels
        for _, account in accounts.iterrows():
            if account['will_churn']:
                churn_date = account['churn_date']
                # Mark snapshots where churn happens in next 30 days
                mask = (
                    (df['account_id'] == account['account_id']) &
                    (df['snapshot_date'] >= (churn_date - timedelta(days=30))) &
                    (df['snapshot_date'] < churn_date)
                )
                df.loc[mask, 'churned_next_30d'] = True
        
        logger.success(f"Generated {len(df):,} feature snapshots")
        logger.info(f"Churn rate in features: {df['churned_next_30d'].mean():.1%}")
        return df
    
    def save_to_csv(self, accounts: pd.DataFrame, events: pd.DataFrame, features: pd.DataFrame, output_dir: str):
        """Save generated data to CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving data to {output_path}...")
        
        # Save accounts
        accounts_path = output_path / 'accounts.csv'
        accounts.to_csv(accounts_path, index=False)
        logger.info(f"  Saved {len(accounts):,} accounts to {accounts_path}")
        
        # Save events (sample if too large)
        events_path = output_path / 'events.csv'
        if len(events) > 1000000:
            logger.info(f"  Sampling {len(events):,} events down to 1M for CSV...")
            events_sample = events.sample(n=1000000, random_state=self.seed)
            events_sample.to_csv(events_path, index=False)
            logger.info(f"  Saved {len(events_sample):,} events to {events_path}")
        else:
            events.to_csv(events_path, index=False)
            logger.info(f"  Saved {len(events):,} events to {events_path}")
        
        # Save features
        features_path = output_path / 'features.csv'
        features.to_csv(features_path, index=False)
        logger.info(f"  Saved {len(features):,} feature snapshots to {features_path}")
        
        logger.success("Data saved successfully!")
    
    def run(self, output_dir: str = 'data/raw'):
        """Run full data generation pipeline."""
        logger.info("=" * 60)
        logger.info("SignalForge Synthetic Data Generator")
        logger.info("=" * 60)
        
        accounts = self.generate_accounts()
        accounts = self.generate_churn_labels(accounts)
        events = self.generate_events(accounts)
        features = self.generate_features(accounts, events)
        
        self.save_to_csv(accounts, events, features, output_dir)
        
        logger.success("Data generation complete!")
        
        return accounts, events, features


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic SaaS data")
    parser.add_argument('--accounts', type=int, default=10000, help='Number of accounts to generate')
    parser.add_argument('--months', type=int, default=24, help='Number of months of history')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', default='data/raw', help='Output directory')
    
    args = parser.parse_args()
    
    generator = SyntheticDataGenerator(
        n_accounts=args.accounts,
        n_months=args.months,
        seed=args.seed
    )
    
    generator.run(output_dir=args.output)


if __name__ == "__main__":
    main()
