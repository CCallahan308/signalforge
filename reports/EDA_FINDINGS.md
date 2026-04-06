# SignalForge - EDA Findings Report

## Executive Summary

**Dataset:** IBM Telco Customer Churn (7,043 customers)
**Churn Rate:** 26.5%
**Revenue at Risk:** $139,131/month ($1.67M/year)

---

## Key Findings

### 1. Contract Type is the #1 Churn Predictor

| Contract Type | Churn Rate | Customers |
|---------------|------------|-----------|
| Month-to-month | **42.7%** | 3,875 |
| One year | 11.3% | 1,473 |
| Two year | 2.8% | 1,695 |

**Insight:** Month-to-month customers churn **15x more** than two-year contracts.

**Action:** Target month-to-month customers with annual conversion campaign (discount, benefits).

---

### 2. New Customers (0-12 months) Are High Risk

| Tenure | Churn Rate | Customers |
|--------|------------|-----------|
| 0-12 months | **47.4%** | 2,175 |
| 13-24 months | 28.7% | 1,024 |
| 25-48 months | 20.4% | 1,594 |
| 49-72 months | 9.5% | 2,239 |

**Insight:** First-year customers churn **5x more** than customers with 4+ years.

**Action:** Implement onboarding program, proactive check-ins, and early engagement tracking.

---

### 3. Payment Method Matters

| Payment Method | Churn Rate | Customers |
|----------------|------------|-----------|
| Electronic check | **45.3%** | 2,365 |
| Mailed check | 19.1% | 1,612 |
| Bank transfer (auto) | 16.7% | 1,544 |
| Credit card (auto) | 15.2% | 1,522 |

**Insight:** Electronic check users churn **3x more** than automatic payment users.

**Action:** Migrate customers to automatic payment methods with incentives.

---

### 4. Service Bundles Reduce Churn

| Service | Has Service | No Service |
|---------|-------------|------------|
| Online Security | 14.6% | 41.8% |
| Tech Support | 15.2% | 41.6% |
| Online Backup | 21.5% | 39.9% |
| Device Protection | 22.5% | 39.1% |

**Insight:** Customers without security/support services churn **2-3x more**.

**Action:** Bundle services, promote security packages, offer trial periods.

---

### 5. Demographics: Seniors & Singles Churn More

| Demographic | Churn Rate |
|-------------|------------|
| Senior Citizens | 41.7% |
| Non-Seniors | 23.6% |
| No Partner | 33.0% |
| Has Partner | 19.7% |
| No Dependents | 31.3% |
| Has Dependents | 15.5% |

**Insight:** Senior citizens churn **1.8x more**, singles churn **1.7x more**.

**Action:** Targeted support for seniors, family plan promotions.

---

### 6. High-Value Customers Churn More

| Customer Type | Churn Rate | Revenue at Risk |
|---------------|------------|-----------------|
| Standard (<$89.85/mo) | 24.4% | $81,447/month |
| High-Value (≥$89.85/mo) | 32.7% | **$57,683/month** |

**Insight:** High-value customers (top 25%) churn **1.3x more** - representing **41% of at-risk revenue**.

**Action:** Priority intervention for high-value at-risk customers.

---

## Statistical Summary

### Numerical Features

| Feature | Retained | Churned | Difference |
|---------|----------|---------|------------|
| Tenure (months) | 37.6 | 18.0 | **-19.6 months** |
| Monthly Charges | $61.27 | $74.44 | **+$13.17** |
| Total Charges | $2,555 | $1,531 | **-$1,024** |

### Correlations with Churn

| Feature | Correlation |
|---------|-------------|
| Tenure | **-0.352** (strongest) |
| Total Charges | -0.199 |
| Monthly Charges | +0.193 |

---

## Modeling Recommendations

### Target Variable
- **Churn rate:** 26.5% (moderate imbalance)
- **Strategy:** Use `class_weight='balanced'` or SMOTE

### Expected Performance
- **Baseline (majority class):** 73.5% accuracy
- **Target AUC-ROC:** 0.80-0.85
- **Business metric:** Revenue saved / intervention cost

### Top Features for Model
1. **Contract Type** (categorical - strongest signal)
2. **Tenure** (numerical - strongest correlation)
3. **Payment Method** (categorical - strong signal)
4. **Monthly Charges** (numerical)
5. **Service Bundles** (categorical - multiple features)
6. **Demographics** (categorical - weaker signal)

### Feature Engineering Opportunities
1. **Tenure buckets** (0-12, 13-24, 25-48, 49-72 months)
2. **Service count** (total number of services subscribed)
3. **Payment automation** (auto vs manual)
4. **Charge trends** (monthly vs average)
5. **Customer lifecycle stage** (new, growing, mature)
6. **High-value flag** (top 25% by MRR)

---

## Business Impact Analysis

### Revenue at Risk

| Segment | Monthly Revenue | % of Total |
|---------|-----------------|------------|
| Total Revenue | $456,117 | 100% |
| **Churned Revenue** | **$139,131** | **30.5%** |
| **Annual Impact** | **$1,669,570** | - |

### High-Value Customers (Top 25%)

| Metric | Value |
|--------|-------|
| Count | 1,771 customers |
| Churn Rate | 32.7% |
| Revenue at Risk | $57,683/month |
| % of At-Risk Revenue | **41.4%** |

**Critical:** High-value customers represent only 25% of base but 41% of revenue at risk.

---

## Recommended Interventions

### 1. Annual Conversion Campaign (High ROI)
**Target:** Month-to-month customers
**Offer:** 15-20% discount for annual commitment
**Expected Impact:** Reduce churn from 42.7% to ~25%
**ROI:** $2-3 saved per $1 invested

### 2. Onboarding Excellence Program
**Target:** 0-12 month customers
**Actions:**
- Week 1: Welcome call + setup assistance
- Month 1: Feature training
- Month 3: Check-in + feedback
- Month 6: Expansion discussion
**Expected Impact:** Reduce churn from 47.4% to ~35%

### 3. Payment Automation Migration
**Target:** Electronic check users
**Offer:** $10-20 credit for switching to auto-pay
**Expected Impact:** Reduce churn from 45.3% to ~30%
**ROI:** $5-10 saved per $1 invested

### 4. Security Bundle Promotion
**Target:** Customers without security/tech support
**Offer:** 3-month free trial of security bundle
**Expected Impact:** Reduce churn from 40% to ~25%

---

## Model Development Priorities

### Phase 1: Baseline Model
- Logistic Regression with class weights
- Features: Contract, Tenure, PaymentMethod, MonthlyCharges
- Target: AUC 0.78+

### Phase 2: Advanced Model
- XGBoost/LightGBM with feature engineering
- Add service bundles, demographics, charge trends
- Target: AUC 0.82+

### Phase 3: Uplift Modeling
- Identify which customers are **saveable** (not just at-risk)
- Target interventions to customers with high uplift scores
- Optimize intervention ROI

---

## Next Steps

1. **Feature Engineering** (create `scripts/engineer_features.py`)
   - Tenure buckets
   - Service count
   - Payment automation flag
   - High-value flag

2. **Model Training** (create `scripts/train_model.py`)
   - Train/test split (time-based)
   - Baseline logistic regression
   - XGBoost with hyperparameter tuning
   - Cross-validation

3. **Model Evaluation** (create `scripts/evaluate_model.py`)
   - AUC-ROC, AUC-PR
   - Precision/Recall at different thresholds
   - Feature importance
   - SHAP values for interpretability

4. **Business Analysis** (create `scripts/business_impact.py`)
   - Revenue at risk by segment
   - Intervention ROI calculator
   - What-if scenarios

5. **API Development** (create `src/api/main.py`)
   - Prediction endpoint
   - Feature computation
   - Business metrics

---

## Key Metrics to Track

### Model Performance
- AUC-ROC: Target 0.82+
- Precision @ 20%: Target 0.60+
- Recall @ 20%: Target 0.50+

### Business Impact
- Revenue at risk identified
- Customers flagged for intervention
- Estimated revenue saved
- Intervention ROI

### Production Metrics
- Prediction latency: <100ms
- Model drift (monthly)
- Feature drift (weekly)

---

**Status:** EDA Complete - Ready for Feature Engineering
**Next:** Create `scripts/engineer_features.py`
