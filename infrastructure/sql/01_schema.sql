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
    company_size VARCHAR(50),
    plan_type VARCHAR(50) NOT NULL,
    billing_frequency VARCHAR(20),
    mrr DECIMAL(10, 2) NOT NULL,
    contract_start_date DATE NOT NULL,
    contract_end_date DATE,
    account_manager_id UUID,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Users
CREATE TABLE raw.users (
    user_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES raw.accounts(account_id),
    email VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    role VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- Subscriptions
CREATE TABLE raw.subscriptions (
    subscription_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES raw.accounts(account_id),
    plan_id VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    current_period_start DATE,
    current_period_end DATE,
    canceled_at TIMESTAMP,
    cancelation_reason TEXT,
    mrr DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Events (simplified - no partitioning for now)
CREATE TABLE raw.events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES raw.users(user_id),
    account_id UUID NOT NULL REFERENCES raw.accounts(account_id),
    event_type VARCHAR(100) NOT NULL,
    event_category VARCHAR(50),
    properties JSONB,
    session_id UUID,
    occurred_at TIMESTAMP NOT NULL,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Payments
CREATE TABLE raw.payments (
    payment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES raw.accounts(account_id),
    subscription_id UUID REFERENCES raw.subscriptions(subscription_id),
    amount DECIMAL(10, 2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    status VARCHAR(20) NOT NULL,
    payment_method VARCHAR(50),
    invoice_id VARCHAR(255),
    paid_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Support Tickets
CREATE TABLE raw.support_tickets (
    ticket_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES raw.accounts(account_id),
    user_id UUID REFERENCES raw.users(user_id),
    subject TEXT,
    description TEXT,
    category VARCHAR(50),
    priority VARCHAR(20),
    status VARCHAR(20),
    sentiment_score DECIMAL(3, 2),
    resolution_time_hours DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

-- NPS Surveys
CREATE TABLE raw.nps_surveys (
    survey_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES raw.accounts(account_id),
    user_id UUID REFERENCES raw.users(user_id),
    score INTEGER CHECK (score >= 0 AND score <= 10),
    feedback TEXT,
    survey_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Interventions
CREATE TABLE raw.interventions (
    intervention_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id UUID NOT NULL REFERENCES raw.accounts(account_id),
    intervention_type VARCHAR(50) NOT NULL,
    channel VARCHAR(50),
    initiated_by UUID,
    initiated_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    outcome VARCHAR(50),
    cost_usd DECIMAL(10, 2),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- FEATURES LAYER
-- ============================================

-- Account-level features (daily snapshots)
CREATE TABLE features.account_daily (
    snapshot_date DATE NOT NULL,
    account_id UUID NOT NULL REFERENCES raw.accounts(account_id),
    account_age_days INTEGER,
    months_active INTEGER,
    daily_active_users INTEGER,
    weekly_active_users INTEGER,
    monthly_active_users INTEGER,
    total_events_7d BIGINT,
    total_events_30d BIGINT,
    avg_events_per_user_7d DECIMAL(10, 2),
    login_frequency_7d DECIMAL(10, 2),
    feature_adoption_rate DECIMAL(5, 4),
    core_feature_usage_rate DECIMAL(5, 4),
    session_duration_avg_7d DECIMAL(10, 2),
    open_tickets INTEGER,
    tickets_30d INTEGER,
    avg_resolution_time_hours DECIMAL(10, 2),
    avg_sentiment_score DECIMAL(4, 3),
    mrr DECIMAL(10, 2),
    mrr_change_1m DECIMAL(10, 2),
    mrr_change_3m DECIMAL(10, 2),
    failed_payments_6m INTEGER,
    days_since_last_payment INTEGER,
    payment_method_failures INTEGER,
    nps_score INTEGER,
    days_since_nps_survey INTEGER,
    nps_feedback_length INTEGER,
    days_to_renewal INTEGER,
    contract_value DECIMAL(12, 2),
    is_annual BOOLEAN,
    interventions_90d INTEGER,
    last_intervention_days INTEGER,
    last_intervention_outcome VARCHAR(50),
    churned_next_30d BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (snapshot_date, account_id)
);

-- Model features
CREATE TABLE features.model_features (
    snapshot_date DATE NOT NULL,
    account_id UUID NOT NULL,
    feature_vector JSONB NOT NULL,
    feature_version VARCHAR(20) DEFAULT '1.0',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (snapshot_date, account_id)
);

-- ============================================
-- MODELS LAYER
-- ============================================

-- Model registry
CREATE TABLE models.model_registry (
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    algorithm VARCHAR(100),
    hyperparameters JSONB,
    feature_importance JSONB,
    training_metrics JSONB,
    validation_metrics JSONB,
    training_data_start DATE,
    training_data_end DATE,
    training_samples INTEGER,
    model_artifact_path TEXT,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),
    UNIQUE (model_name, version)
);

-- Predictions
CREATE TABLE models.predictions (
    prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES models.model_registry(model_id),
    account_id UUID NOT NULL REFERENCES raw.accounts(account_id),
    snapshot_date DATE NOT NULL,
    prediction_date TIMESTAMP NOT NULL,
    churn_probability DECIMAL(6, 5),
    churn_risk_tier VARCHAR(20),
    uplift_score DECIMAL(6, 5),
    save_probability DECIMAL(6, 5),
    expected_value DECIMAL(12, 2),
    intervention_roi DECIMAL(10, 2),
    top_risk_factors JSONB,
    shap_values JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model performance tracking
CREATE TABLE models.performance_daily (
    date DATE NOT NULL,
    model_id UUID NOT NULL REFERENCES models.model_registry(model_id),
    auc_roc DECIMAL(6, 5),
    auc_pr DECIMAL(6, 5),
    precision_score DECIMAL(6, 5),
    recall_score DECIMAL(6, 5),
    f1_score DECIMAL(6, 5),
    brier_score DECIMAL(6, 5),
    uplift_correlation DECIMAL(6, 5),
    qini_coefficient DECIMAL(6, 5),
    true_positives INTEGER,
    false_positives INTEGER,
    true_negatives INTEGER,
    false_negatives INTEGER,
    samples_scored INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, model_id)
);

-- ============================================
-- ANALYTICS LAYER
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
-- INDEXES
-- ============================================

CREATE INDEX idx_accounts_plan ON raw.accounts(plan_type);
CREATE INDEX idx_accounts_industry ON raw.accounts(industry);
CREATE INDEX idx_accounts_created ON raw.accounts(created_at);

CREATE INDEX idx_users_account ON raw.users(account_id);
CREATE INDEX idx_users_last_login ON raw.users(last_login_at);

CREATE INDEX idx_subscriptions_account ON raw.subscriptions(account_id);
CREATE INDEX idx_subscriptions_status ON raw.subscriptions(status);
CREATE INDEX idx_subscriptions_period_end ON raw.subscriptions(current_period_end);

CREATE INDEX idx_events_account_time ON raw.events(account_id, occurred_at DESC);
CREATE INDEX idx_events_type ON raw.events(event_type);
CREATE INDEX idx_events_category ON raw.events(event_category);

CREATE INDEX idx_payments_account ON raw.payments(account_id);
CREATE INDEX idx_payments_paid_at ON raw.payments(paid_at DESC);
CREATE INDEX idx_payments_status ON raw.payments(status);

CREATE INDEX idx_tickets_account ON raw.support_tickets(account_id);
CREATE INDEX idx_tickets_created ON raw.support_tickets(created_at DESC);
CREATE INDEX idx_tickets_sentiment ON raw.support_tickets(sentiment_score);

CREATE INDEX idx_nps_account ON raw.nps_surveys(account_id);
CREATE INDEX idx_nps_score ON raw.nps_surveys(score);

CREATE INDEX idx_interventions_account ON raw.interventions(account_id);
CREATE INDEX idx_interventions_outcome ON raw.interventions(outcome);

CREATE INDEX idx_features_account_date ON features.account_daily(account_id, snapshot_date DESC);
CREATE INDEX idx_models_active ON models.model_registry(model_name, is_active);
CREATE INDEX idx_predictions_account_date ON models.predictions(account_id, prediction_date DESC);
CREATE INDEX idx_predictions_model ON models.predictions(model_id, snapshot_date);

-- ============================================
-- FUNCTIONS & TRIGGERS
-- ============================================

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

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
    p.churn_probability,
    p.churn_risk_tier,
    p.uplift_score,
    p.intervention_roi,
    f.daily_active_users,
    f.login_frequency_7d,
    f.feature_adoption_rate,
    f.open_tickets,
    f.avg_sentiment_score,
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
