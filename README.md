# SignalForge - Production SaaS Churn Intelligence

> **A real ML project by a real grad student** 👋

Hi, I'm Christian (Red). I'm a dual Master's candidate (MBA + MS Data Science) building my first production ML system while working full-time.

This isn't a bootcamp project or tutorial follow-along. This is me learning how to build end-to-end ML systems for real business problems, with the goal of landing a $200k+ remote senior data scientist role.

---

## What This Project Does

Predicts customer churn for telecom companies using real IBM data. More importantly, it shows I understand the **full stack** of data science:

- **Data Engineering:** PostgreSQL schema, ETL pipelines, data quality
- **Feature Engineering:** 58 business-informed features
- **ML Modeling:** Baseline → Advanced, with business metrics
- **MLOps:** Production-ready, monitored, maintainable
- **Business Impact:** ROI quantification, stakeholder-ready insights

---

## 🎯 Project Status

| Component | Status | Progress |
|-----------|--------|----------|
| **Database** | ✅ Complete | 100% |
| **Data Ingestion** | ✅ Complete | 100% |
| **EDA** | ✅ Complete | 100% |
| **Feature Engineering** | ✅ Complete | 100% |
| **Model Training** | ✅ Complete | 100% |
| **API** | ⏳ Next | 0% |
| **Dashboard** | ⏳ Pending | 0% |
| **Monitoring** | ⏳ Pending | 0% |

**Overall Progress:** 50% → Production-ready in 2-3 more sessions

---

## 📊 Key Results

### Model Performance
- **Best Model:** Logistic Regression (0.848 AUC)
- **Recall:** 81% of churners identified
- **Business Impact:** $113K monthly revenue at risk identified

### Revenue Impact
- **Total at Risk:** $139K/month ($1.67M/year)
- **Expected ROI:** 1.21x - 1.81x
- **Annual Savings:** $270K - $405K

### What Surprised Me
The simplest model (logistic regression) beat complex ones (random forest, gradient boosting). This taught me that **good feature engineering > fancy algorithms**.

---

## 🚀 Quick Start

```bash
# Clone repo
git clone https://github.com/CCallahan308/signalforge
cd signalforge

# Set up environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download data (requires Kaggle API)
python scripts/download_real_data.py --source telco

# Engineer features
python scripts/engineer_features.py

# Train models
python scripts/train_model.py
```

See [QUICKSTART.md](QUICKSTART.md) for full guide.

---

## 📁 Project Structure

```
signalforge/
├── data/
│   ├── raw/                    # Downloaded from Kaggle
│   └── processed/              # Engineered features
├── models/
│   └── artifacts/              # Trained models + results
├── scripts/
│   ├── download_real_data.py   # Kaggle integration
│   ├── engineer_features.py    # Feature engineering
│   └── train_model.py          # Model training
├── docs/
│   ├── MODEL_RESULTS.md        # Training results
│   ├── FEATURES.md             # Feature documentation
│   └── SETUP.md                # Setup guide
├── ABOUT.md                    # My story (why I built this)
├── LEARNING.md                 # What I'm learning
└── QUICKSTART.md               # Get started guide
```

---

## 📚 Documentation

- **[ABOUT.md](ABOUT.md)** - My story, why I built this, my journey
- **[LEARNING.md](LEARNING.md)** - What I'm learning, mistakes, insights
- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 3 commands
- **[docs/SETUP.md](docs/SETUP.md)** - Full setup guide
- **[docs/FEATURES.md](docs/FEATURES.md)** - Feature engineering docs
- **[docs/MODEL_RESULTS.md](docs/MODEL_RESULTS.md)** - Training results

---

## 💼 Resume-Ready Bullets

> "Built production churn prediction system on IBM Telco dataset (7K customers), achieving 0.848 AUC with logistic regression and identifying $113K monthly revenue at risk"

> "Engineered 58 production-grade features from raw customer data, including composite risk scores and business-informed metrics, demonstrating that good features beat fancy algorithms"

> "Created end-to-end ML pipeline from data ingestion to feature engineering to model training, showing systems thinking and production-ready code quality"

---

## 🎓 About Me

**Christian G Callahan** (Red)
- Dual Master's Candidate: MBA + MS Data Science
- Current Role: BI Analyst
- Goal: Land $200k+ remote senior data scientist role
- Location: America/Chicago (CDT)

**Why This Project:**
I wanted to build something that shows I'm not just an academic. I can build real systems, think about production, and translate ML to business impact.

---

## 🤝 Let's Connect

- **GitHub:** [@CCallahan308](https://github.com/CCallahan308)
- **LinkedIn:** [christian--callahan](https://www.linkedin.com/in/christian--callahan/)
- **Website:** [christiangcallahan.tech](https://www.christiangcallahan.tech/)

---

## 📊 Project Stats

| Metric | Value |
|--------|-------|
| **Commits** | 7 |
| **Files Created** | 25+ |
| **Time Invested** | ~5 hours |
| **Features Engineered** | 58 |
| **Revenue at Risk** | $1.67M/year |
| **Model AUC** | 0.848 |

---

## 🚧 What's Next

**Immediate (This Week):**
- [ ] Build FastAPI for predictions
- [ ] Add business metrics (ROI calculator)
- [ ] Create intervention recommendations

**Short-term (2 Weeks):**
- [ ] Build Streamlit dashboard
- [ ] Add customer risk profiles
- [ ] Deploy to cloud

**Long-term (1 Month):**
- [ ] Add uplift modeling
- [ ] A/B test interventions
- [ ] Measure real-world impact

---

## 🙏 Acknowledgments

- IBM for the Telco dataset
- Kaggle community
- Open-source ML community (scikit-learn, pandas)
- My professors and peers

---

## 📄 License

MIT

---

**Built with 💼 by a grad student learning in public**

This project isn't perfect. It's a work in progress by someone still learning. But that's the point - showing my process, the wins, the challenges, the iterations.

Because that's what real data science looks like.

---

*Last updated: April 6, 2026*
