-- SignalForge PostgreSQL Schema
-- Production-grade SaaS churn intelligence database

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Schemas
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS models;
CREATE SCHEMA IF NOT EXISTS analytics;

-- ============================================
-- RAW LAYER (Ingested data)
-- ============================================

-- Accounts/Customers
CREATE TABLE raw.accounts (
    account_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    company_name VARCHAR(255),
    industry VARCHAR(100),
    company_size VARCHAR(50), -- '1-10', '11-50', '51-200', '201-500', '500+'
    plan_type VARCHAR(50) NOT NULL, -- 'free', 'starter', 'pro', 'enterprise'
    billing_frequency VARCHAR(20), -- 'monthly', 'annual'
    mrr DECIMAL(10, 2) NOT NULL, -- Monthly Recurring Revenue
    contract_start_date DATE NOT NULL,
    contract_end_date DATE,
    account_manager_id UUID,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    INDEX idx_accounts_plan (plan_type),
    INDEX idx_accounts_industry (industry),
    INDEX idx_accounts_created (created_at)
);

-- Users (individual users within accounts)
CREATE TABLE raw.users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES raw.accounts(account_id),
    email VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    role VARCHAR(50), -- 'admin', 'member', 'viewer'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    INDEX idx_users_account (account_id),
    INDEX idx_users_last_login (last_login_at)
);

-- Subscriptions
CREATE TABLE raw.subscriptions (
    subscription_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES raw.accounts(account_id),
    plan_id VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL, -- 'active', 'past_due', 'canceled', 'trialing'
    current_period_start DATE,
    current_period_end DATE,
    canceled_at TIMESTAMP,
    cancelation_reason TEXT,
    mrr DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_subscriptions_account (account_id),
    INDEX idx_subscriptions_status (status),
    INDEX idx_subscriptions_period_end (current_period_end)
);

-- Events (user activity tracking)
CREATE TABLE raw.events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES raw.users(user_id),
    account_id UUID NOT NULL REFERENCES raw.accounts(account_id),
    event_type VARCHAR(100) NOT NULL, -- 'login', 'feature_used', 'api_call', etc.
    event_category VARCHAR(50), -- 'engagement', 'billing', 'support', 'product'
    properties JSONB, -- Flexible event properties
    session_id UUID,
    ip_address INET,
    user_agent TEXT,
    occurred_at TIMESTAMP NOT NULL,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_events_account_time (account_id, occurred_at DESC),
    INDEX idx_events_type (event_type),
    INDEX idx_events_category (event_category),
    PARTITION BY RANGE (occurred_at);
);

-- Create monthly partitions for events (last 24 months)
-- This will be automated by a script
CREATE TABLE raw.events_2024_01 PARTITION OF raw.events
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE raw.events_2024_02 PARTITION OF raw.events
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Payments
CREATE TABLE raw.payments (
    payment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES raw.accounts(account_id),
    subscription_id UUID REFERENCES raw.subscriptions(subscription_id),
    amount DECIMAL(10, 2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    status VARCHAR(20) NOT NULL, -- 'succeeded', 'failed', 'refunded'
    payment_method VARCHAR(50),
    invoice_id VARCHAR(255),
    paid_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_payments_account (account_id),
    INDEX idx_payments_paid_at (paid_at DESC),
    INDEX idx_payments_status (status)
);

-- Support Tickets
CREATE TABLE raw.support_tickets (
    ticket_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES raw.accounts(account_id),
    user_id UUID REFERENCES raw.users(user_id),
    subject TEXT,
    description TEXT,
    category VARCHAR(50),
    priority VARCHAR(20), -- 'low', 'medium', 'high', 'urgent'
    status VARCHAR(20), -- 'open', 'in_progress', 'resolved', 'closed'
    sentiment_score DECIMAL(3, 2), -- -1 to 1
    resolution_time_hours DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    
    INDEX idx_tickets_account (account_id),
    INDEX idx_tickets_created (created_at DESC),
    INDEX idx_tickets_sentiment (sentiment_score)
);

-- NPS Surveys
CREATE TABLE raw.nps_surveys (
    survey_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES raw.accounts(account_id),
    user_id UUID REFERENCES raw.users(user_id),
    score INTEGER CHECK (score >= 0 AND score <= 10),
    feedback TEXT,
    survey_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_nps_account (account_id),
    INDEX idx_nps_score (score)
);

-- Interventions (save attempts)
CREATE TABLE raw.interventions (
    intervention_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES raw.accounts(account_id),
    intervention_type VARCHAR(50) NOT NULL, -- 'email', 'call', 'discount', 'feature_upgrade'
    channel VARCHAR(50), -- 'email', 'phone', 'chat'
    initiated_by UUID, -- Account manager or automated
    initiated_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    outcome VARCHAR(50), -- 'saved', 'churned', 'pending', 'no_response'
    cost_usd DECIMAL(10, 2),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_interventions_account (account_id),
    INDEX idx_interventions_outcome (outcome)
);

-- ============================================
-- FEATURES LAYER (Engineered features)
-- ============================================

-- Account-level features (daily snapshots)
CREATE TABLE features.account_daily (
    snapshot_date DATE NOT NULL,
    account_id UUID NOT NULL REFERENCES raw.accounts(account_id),
    
    -- Tenure features
    account_age_days INTEGER,
    months_active INTEGER,
    
    -- Usage features
    daily_active_users INTEGER,
    weekly_active_users INTEGER,
    monthly_active_users INTEGER,
    total_events_7d BIGINT,
    total_events_30d BIGINT,
    avg_events_per_user_7d DECIMAL(10, 2),
    
    -- Engagement features
    login_frequency_7d DECIMAL(10, 2),
    feature_adoption_rate DECIMAL(5, 4), -- % of features used
    core_feature_usage_rate DECIMAL(5, 4),
    session_duration_avg_7d DECIMAL(10, 2),
    
    -- Support features
    open_tickets INTEGER,
    tickets_30d INTEGER,
    avg_resolution_time_hours DECIMAL(10, 2),
    avg_sentiment_score DECIMAL(4, 3),
    
    -- Billing features
    mrr DECIMAL(10, 2),
    mrr_change_1m DECIMAL(10, 2),
    mrr_change_3m DECIMAL(10, 2),
    failed_payments_6m INTEGER,
    days_since_last_payment INTEGER,
    payment_method_failures INTEGER,
    
    -- NPS features
    nps_score INTEGER,
    days_since_nps_survey INTEGER,
    nps_feedback_length INTEGER,
    
    -- Contract features
    days_to_renewal INTEGER,
    contract_value DECIMAL(12, 2),
    is_annual BOOLEAN,
    
    -- Intervention features
    interventions_90d INTEGER,
    last_intervention_days INTEGER,
    last_intervention_outcome VARCHAR(50),
    
    -- Target
    churned_next_30d BOOLEAN,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (snapshot_date, account_id),
    INDEX idx_features_account_date (account_id, snapshot_date DESC)
);

-- Model features (ready for ML)
CREATE TABLE features.model_features (
    snapshot_date DATE NOT NULL,
    account_id UUID NOT NULL,
    
    -- All features from account_daily plus derived
    feature_vector JSONB NOT NULL, -- Serialized feature vector
    
    -- Metadata
    feature_version VARCHAR(20) DEFAULT '1.0',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (snapshot_date, account_id)
);

-- ============================================
-- MODELS LAYER (Predictions & artifacts)
-- ============================================

-- Model registry
CREATE TABLE models.model_registry (
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- 'churn_risk', 'uplift', 'clv'
    version VARCHAR(20) NOT NULL,
    algorithm VARCHAR(100), -- 'xgboost', 'lightgbm', etc.
    hyperparameters JSONB,
    feature_importance JSONB,
    training_metrics JSONB, -- {auc: 0.66, precision: 0.72, ...}
    validation_metrics JSONB,
    training_data_start DATE,
    training_data_end DATE,
    training_samples INTEGER,
    model_artifact_path TEXT, -- S3/local path
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    
    UNIQUE (model_name, version),
    INDEX idx_models_active (model_name, is_active)
);

-- Predictions
CREATE TABLE models.predictions (
    prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES models.model_registry(model_id),
    account_id UUID NOT NULL REFERENCES raw.accounts(account_id),
    snapshot_date DATE NOT NULL,
    prediction_date TIMESTAMP NOT NULL,
    
    -- Churn predictions
    churn_probability DECIMAL(6, 5),
    churn_risk_tier VARCHAR(20), -- 'low', 'medium', 'high', 'critical'
    
    -- Uplift predictions
    uplift_score DECIMAL(6, 5), -- Intervention effect size
    save_probability DECIMAL(6, 5), -- P(saved | intervention)
    
    -- Value predictions
    expected_value DECIMAL(12, 2), -- Expected account value
    intervention_roi DECIMAL(10, 2), -- ROI of intervening
    
    -- Explanation
    top_risk_factors JSONB, -- [{feature: 'low_usage', impact: 0.15}, ...]
    shap_values JSONB,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_predictions_account_date (account_id, prediction_date DESC),
    INDEX idx_predictions_model (model_id, snapshot_date)
);

-- Model performance tracking
CREATE TABLE models.performance_daily (
    date DATE NOT NULL,
    model_id UUID NOT NULL REFERENCES models.model_registry(model_id),
    
    -- Metrics
    auc_roc DECIMAL(6, 5),
    auc_pr DECIMAL(6, 5),
    precision_score DECIMAL(6, 5),
    recall_score DECIMAL(6, 5),
    f1_score DECIMAL(6, 5),
    
    -- Calibration
    brier_score DECIMAL(6, 5),
    
    -- Uplift metrics
    uplift_correlation DECIMAL(6, 5),
    qini_coefficient DECIMAL(6, 5),
    
    -- Business metrics
    true_positives INTEGER,
    false_positives INTEGER,
    true_negatives INTEGER,
    false_negatives INTEGER,
    
    samples_scored INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (date, model_id)
);

-- ============================================
-- ANALYTICS LAYER (Business metrics)
-- ============================================

-- Intervention effectiveness
CREATE TABLE analytics.intervention_analysis (
    analysis_date DATE NOT NULL,
    intervention_type VARCHAR(50),
    
    accounts_targeted INTEGER,
    accounts_saved INTEGER,
    save_rate DECIMAL(5, 4),
    
    total_cost_usd DECIMAL(12, 2),
    total_revenue_saved_usd DECIMAL(12, 2),
    roi DECIMAL(10, 2),
    
    avg_intervention_cost_usd DECIMAL(10, 2),
    avg_revenue_per_save_usd DECIMAL(10, 2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (analysis_date, intervention_type)
);

-- Churn analysis
CREATE TABLE analytics.churn_cohorts (
    cohort_month DATE NOT NULL,
    plan_type VARCHAR(50),
    industry VARCHAR(100),
    
    accounts_start INTEGER,
    accounts_churned INTEGER,
    churn_rate DECIMAL(5, 4),
    
    mrr_start DECIMAL(12, 2),
    mrr_churned DECIMAL(12, 2),
    
    avg_tenure_days INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    PRIMARY KEY (cohort_month, plan_type, industry)
);

-- ============================================
-- FUNCTIONS & TRIGGERS
-- ============================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to relevant tables
CREATE TRIGGER update_accounts_updated_at
    BEFORE UPDATE ON raw.accounts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_subscriptions_updated_at
    BEFORE UPDATE ON raw.subscriptions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================
-- VIEWS
-- ============================================

-- Customer 360 view
CREATE VIEW analytics.customer_360 AS
SELECT 
    a.account_id,
    a.email,
    a.company_name,
    a.industry,
    a.plan_type,
    a.mrr,
    a.contract_start_date,
    a.contract_end_date,
    
    -- Current churn risk
    p.churn_probability,
    p.churn_risk_tier,
    p.uplift_score,
    
    -- Engagement
    f.daily_active_users,
    f.login_frequency_7d,
    f.feature_adoption_rate,
    
    -- Support health
    f.open_tickets,
    f.avg_sentiment_score,
    
    -- Risk factors
    p.top_risk_factors
    
FROM raw.accounts a
LEFT JOIN LATERAL (
    SELECT * FROM models.predictions 
    WHERE account_id = a.account_id 
    ORDER BY prediction_date DESC 
    LIMIT 1
) p ON true
LEFT JOIN LATERAL (
    SELECT * FROM features.account_daily 
    WHERE account_id = a.account_id 
    ORDER BY snapshot_date DESC 
    LIMIT 1
) f ON true
WHERE a.is_active = TRUE;

-- At-risk accounts
CREATE VIEW analytics.at_risk_accounts AS
SELECT 
    account_id,
    company_name,
    mrr,
    churn_probability,
    churn_risk_tier,
    uplift_score,
    intervention_roi,
    top_risk_factors
FROM analytics.customer_360
WHERE churn_probability >= 0.7
ORDER BY mrr * churn_probability DESC;

-- ============================================

-- Initial indexes for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_events_account_time_opt 
    ON raw.events(account_id, occurred_at DESC);

-- Grant permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA raw TO signalforge_user;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA features TO signalforge_user;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA models TO signalforge_user;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO signalforge_user;
