# SignalForge - Website Update Content

**For:** christiangcallahan.tech/projects/signalforge
**Updated:** April 6, 2026
**Status:** Ready to deploy

---

## UPDATED SIGNALFORGE PAGE

### Hero Section

**SignalForge: Production Churn Intelligence**

A real ML system for predicting customer churn with business impact quantification. Built with production thinking from day one.

**Key Results:**
- **7,043 real customers** (IBM Telco dataset)
- **0.848 AUC** with logistic regression
- **$1.67M annual revenue** at risk identified
- **1.21x - 1.81x ROI** expected from interventions

---

### Project Overview

**What This Is**

This isn't a bootcamp project or tutorial follow-along. I built SignalForge while working full-time as a BI Analyst and pursuing dual Master's degrees (MBA + MS Data Science).

It's a production-grade churn prediction system that demonstrates I understand the **full stack** of data science:

✅ **Data Engineering** - PostgreSQL schema design, ETL pipelines, data quality
✅ **Feature Engineering** - 58 business-informed features
✅ **ML Modeling** - Baseline → Advanced with production thinking
✅ **Business Impact** - ROI quantification, stakeholder-ready insights
✅ **Production Quality** - Docker, monitoring-ready, maintainable code

---

### Technical Implementation

**Stack:**
- **Python 3.11+** - Core language
- **PostgreSQL 18** - Production database (17 tables, 4 schemas)
- **scikit-learn** - ML framework
- **Streamlit** - Interactive dashboard
- **Docker** - Containerization

**Dataset:**
- **Source:** IBM Telco Customer Churn (Kaggle)
- **Size:** 7,043 customers
- **Features:** 34 original → 84 engineered
- **Target:** Churn (26.5% positive rate)

---

### Model Performance

| Model | AUC | Precision | Recall | F1 Score |
|-------|-----|-----------|--------|----------|
| **Logistic Regression** 🏆 | **0.848** | 0.505 | **0.810** | 0.622 |
| Random Forest | 0.843 | 0.564 | 0.711 | 0.629 |
| Gradient Boosting | 0.838 | 0.631 | 0.527 | 0.574 |

**Key Insight:** Simple model (logistic regression) + good features beat complex ensemble methods. This taught me that **feature engineering > algorithm selection**.

---

### Business Impact

**Revenue at Risk Analysis:**

| Metric | Value |
|--------|-------|
| **Total Monthly Revenue** | $456,117 |
| **Revenue at Risk (Churned)** | $139,131/month (30.5%) |
| **Annual Revenue at Risk** | **$1,669,570** |

**Expected ROI:**

**Conservative Scenario (20% intervention success):**
- Revenue saved: $22,539/month
- Intervention cost: $18,650
- **Net ROI: 1.21x**

**Optimistic Scenario (30% intervention success):**
- Revenue saved: $33,809/month
- Intervention cost: $18,650
- **Net ROI: 1.81x**

**Annual Impact:** $270K - $405K saved

---

### Top Churn Drivers

Based on feature importance analysis:

1. **Contract Type** - Month-to-month customers churn 3.8x more than annual
2. **Tenure** - New customers (0-12 months) churn 5x more than established
3. **Payment Method** - Electronic check users churn 3x more than auto-pay
4. **Service Bundles** - Customers without security services churn 2-3x more
5. **Demographics** - Senior citizens and singles churn 1.5-2x more

---

### Retention Strategies

**High-Risk Customers (Top 20%):**
- Profile: Month-to-month, fiber optic, new customers
- Intervention: Personal outreach, annual contract discount (15-20%), free service upgrade
- Expected Impact: 30% save rate

**Medium-Risk Customers (20-50%):**
- Profile: Month-to-month but stable
- Intervention: Email campaign with retention offer, service bundle promotion
- Expected Impact: 15% save rate

**Low-Risk Customers (Bottom 50%):**
- Profile: Annual contracts, high tenure, multiple services
- Intervention: Monitor only

---

### Feature Engineering

**58 Production-Grade Features:**

| Feature Group | Count | Key Features |
|---------------|-------|--------------|
| **Tenure** | 5 | tenure_risk, lifecycle_stage, is_new_customer |
| **Contract** | 4 | contract_risk_score, is_month_to_month |
| **Payment** | 5 | payment_risk_score, is_auto_payment |
| **Service** | 13 | service_adoption_rate, has_security |
| **Demographic** | 7 | demographic_risk, family_size |
| **Financial** | 9 | price_sensitivity, is_high_value |
| **Interaction** | 5 | churn_risk_score, engagement_score |

**Top Predictive Features:**
1. `churn_risk_score` - Composite risk (contract + tenure + payment)
2. `is_month_to_month` - Binary contract flag
3. `tenure_risk` - Tenure-based risk score
4. `is_new_customer` - 0-12 months flag
5. `value_at_risk` - Revenue × risk

---

### Dashboard Features

**Interactive Streamlit Dashboard:**

📊 **Business Impact Overview**
- Total customers, churn rate, revenue at risk
- 4 key business metrics

🎯 **Customer Risk Leaderboard**
- Top 20 at-risk customers
- Risk scores, MRR, contract details

💰 **ROI Calculator**
- Intervention cost vs revenue saved
- Success rate scenarios
- Net ROI visualization

📊 **Model Performance**
- Compare 3 models side-by-side
- AUC, precision, recall metrics
- Feature importance rankings

🔍 **Feature Insights**
- Churn by contract type
- Churn by tenure
- Numerical feature distributions

---

### What I Learned

**Technical Insights:**
1. **Feature Engineering > Algorithm Selection**
   - 3 hours on features > 10 hours on hyperparameters
   - Simple model + good features beats complex model + bad features

2. **Production Thinking from Day 1**
   - Not just notebooks, but modular code
   - Error handling, logging, documentation
   - Monitoring-ready architecture

3. **Business Metrics > Technical Metrics**
   - AUC is nice, but ROI matters more
   - Stakeholders need interpretable models
   - Revenue impact > accuracy score

**Personal Growth:**
- Learning in public builds credibility
- Authentic voice differentiates
- Time constraints = prioritization skills

---

### What Makes This Different

**Not Your Typical Portfolio Project:**

❌ **What This Isn't:**
- A clean Kaggle kernel
- A tutorial follow-along
- A bootcamp capstone
- A theoretical notebook

✅ **What This Is:**
- Real data with real mess
- Production thinking
- Business impact quantified
- Learning documented publicly
- Authentic human voice

**Built While:**
- Working full-time as a BI Analyst
- Pursuing dual Master's degrees
- Time-constrained (only ~6 hours total)

---

### Code Quality

**Production-Ready:**
- ✅ Modular scripts (not monolithic)
- ✅ Error handling
- ✅ Logging
- ✅ Documentation
- ✅ Docker containerization
- ✅ PostgreSQL schema
- ✅ Interactive dashboard

**GitHub Stats:**
- 10+ commits
- 35+ files
- ~8,000 lines of code
- 10+ documentation files

---

### Next Steps

**What's Left to Build:**

⏳ **API Development:**
- FastAPI for real-time predictions
- Batch scoring endpoint
- Intervention recommendations

⏳ **Monitoring:**
- Drift detection (Evidently AI)
- Performance tracking
- Automated retraining

⏳ **Testing:**
- Pytest unit tests
- Data validation
- CI/CD pipeline

---

### See It In Action

**GitHub Repository:**
🔗 [github.com/CCallahan308/signalforge](https://github.com/CCallahan308/signalforge)

**Key Files:**
- 📊 [Dashboard](https://github.com/CCallahan308/signalforge/blob/main/src/app/dashboard.py)
- 📝 [Model Results](https://github.com/CCallahan308/signalforge/blob/main/docs/MODEL_RESULTS.md)
- 📚 [Documentation](https://github.com/CCallahan308/signalforge/tree/main/docs)
- 🎓 [Learning Notes](https://github.com/CCallahan308/signalforge/blob/main/LEARNING.md)

---

### Contact

**Christian G Callahan** (Red)

- 📧 [contact@christiangcallahan.tech](mailto:contact@christiangcallahan.tech)
- 💼 [LinkedIn](https://www.linkedin.com/in/christian--callahan/)
- 🐙 [GitHub](https://github.com/CCallahan308)
- 🌐 [Portfolio](https://www.christiangcallahan.tech/)

---

**Built with 💼 by a grad student learning in public**

*This project isn't perfect. It's a work in progress by someone still learning. But that's the point - showing my process, the wins, the challenges, the iterations.*

*Because that's what real data science looks like.*

---

## SHORT VERSION (For Projects Grid)

**SignalForge: Production Churn Intelligence**

A real ML system predicting customer churn with 0.848 AUC on 7,043 real customers. Identified $1.67M annual revenue at risk with expected 1.21x-1.81x ROI from targeted interventions.

**Tech Stack:** Python • PostgreSQL • scikit-learn • Streamlit • Docker

**Key Results:**
- 0.848 AUC (logistic regression beat ensemble methods)
- $1.67M annual revenue at risk identified
- 58 production-grade features engineered
- Interactive dashboard with ROI calculator

[View Project →](/projects/signalforge) | [GitHub →](https://github.com/CCallahan308/signalforge)

---

## METADATA (For SEO)

**Title:** SignalForge - Production Churn Intelligence System
**Description:** Real ML project predicting customer churn with 0.848 AUC, identifying $1.67M annual revenue at risk. Built by Christian G Callahan - MS Data Science Candidate.
**Keywords:** churn prediction, machine learning, data science, portfolio, production ML, customer retention, ROI optimization
**Author:** Christian G Callahan
**Date:** April 6, 2026
