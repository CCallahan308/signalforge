# SignalForge - Production Churn Intelligence

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**A production ML system for predicting customer churn with business impact quantification**

[Demo](#-demo) • [Quick Start](#-quick-start) • [Results](#-key-results) • [Documentation](#-documentation)

</div>

---

## 👋 About This Project

Hi, I'm **Christian (Red)**. I'm a dual Master's candidate (MBA + MS Data Science) building my first production ML system while working full-time.

This isn't a bootcamp project or tutorial follow-along. This is me learning how to build **end-to-end ML systems** for real business problems, with the goal of landing a **$200k+ remote senior data scientist role**.

### What This Project Demonstrates

✅ **Data Engineering** - PostgreSQL schema design, ETL pipelines, data quality  
✅ **Feature Engineering** - 58 business-informed features that beat fancy algorithms  
✅ **ML Modeling** - Baseline → Advanced with production thinking  
✅ **Business Impact** - ROI quantification, stakeholder-ready insights  
✅ **Production Quality** - Docker, monitoring-ready, maintainable code  

---

## 📊 Key Results

### Model Performance

| Model | AUC | Precision | Recall | F1 Score |
|-------|-----|-----------|--------|----------|
| **Logistic Regression** 🏆 | **0.848** | 0.505 | **0.810** | 0.622 |
| Random Forest | 0.843 | 0.564 | 0.711 | 0.629 |
| Gradient Boosting | 0.838 | 0.631 | 0.527 | 0.574 |

**Key Insight:** Simple model + good features > complex model + bad features

### Business Impact

| Metric | Value |
|--------|-------|
| **Revenue at Risk** | $139K/month ($1.67M/year) |
| **Model Identifies** | $113K/month (81% of churners) |
| **Expected ROI** | 1.21x - 1.81x |
| **Annual Savings** | $270K - $405K |

### Features

- **58 engineered features** from raw customer data
- **Top predictors:** contract type, tenure, payment method
- **Business-informed:** composite risk scores, engagement metrics

---

## 🎯 Project Status

| Component | Status | Progress |
|-----------|--------|----------|
| Database & Schema | ✅ Complete | 100% |
| Data Ingestion | ✅ Complete | 100% |
| EDA & Analysis | ✅ Complete | 100% |
| Feature Engineering | ✅ Complete | 100% |
| Model Training | ✅ Complete | 100% |
| Dashboard | ✅ Complete | 100% |
| Deployment Setup | ✅ Complete | 100% |
| Documentation | ✅ Complete | 100% |
| **API** | ⏳ Planned | 0% |
| **Monitoring** | ⏳ Planned | 0% |

**Overall:** 80% Complete → Production-ready for portfolio

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 18
- Kaggle API key (for data download)

### Installation

```bash
# Clone repository
git clone https://github.com/CCallahan308/signalforge
cd signalforge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up database
python scripts/setup_database.py --user postgres --password YOUR_PASSWORD

# Download data (requires Kaggle API)
python scripts/download_real_data.py --source telco

# Engineer features
python scripts/engineer_features.py

# Train models
python scripts/train_model.py

# Launch dashboard
streamlit run src/app/dashboard.py
```

### Docker Deployment

```bash
# Start all services
docker-compose up -d

# Access dashboard
open http://localhost:8501
```

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for full deployment options.

---

## 📁 Project Structure

```
signalforge/
├── 📊 data/
│   ├── raw/                    # Downloaded from Kaggle
│   └── processed/              # 84 engineered features
├── 🤖 models/
│   └── artifacts/              # Trained models + results
├── 🛠️ scripts/
│   ├── setup_database.py       # PostgreSQL setup
│   ├── download_real_data.py   # Kaggle integration
│   ├── engineer_features.py    # Feature engineering
│   ├── train_model.py          # Model training
│   └── quick_eda.py            # EDA insights
├── 📱 src/
│   ├── app/
│   │   └── dashboard.py        # Streamlit dashboard
│   ├── api/                    # FastAPI (planned)
│   └── monitoring/             # Drift detection (planned)
├── 📚 docs/
│   ├── SETUP.md                # Full setup guide
│   ├── FEATURES.md             # Feature documentation
│   ├── MODEL_RESULTS.md        # Training results
│   ├── DATASETS.md             # Dataset comparison
│   └── DEPLOYMENT.md           # Deployment guide
├── 🐳 infrastructure/
│   └── sql/
│       └── 01_schema.sql       # 17 tables, 4 schemas
├── 📄 README.md                # This file
├── 📖 ABOUT.md                 # My story
├── 🎓 LEARNING.md              # What I'm learning
├── 🔍 REVIEW.md                # Project review
├── 🐋 Dockerfile               # Container config
├── 🐋 docker-compose.yml       # Multi-service orchestration
└── ⚙️ requirements.txt         # Dependencies
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [**ABOUT.md**](ABOUT.md) | My story, why I built this, my journey |
| [**LEARNING.md**](LEARNING.md) | What I'm learning, mistakes, insights |
| [**QUICKSTART.md**](QUICKSTART.md) | Get started in 3 commands |
| [**REVIEW.md**](REVIEW.md) | Self-assessment and progress |
| [**docs/SETUP.md**](docs/SETUP.md) | Full setup guide |
| [**docs/FEATURES.md**](docs/FEATURES.md) | Feature engineering docs |
| [**docs/MODEL_RESULTS.md**](docs/MODEL_RESULTS.md) | Training results |
| [**docs/DATASETS.md**](docs/DATASETS.md) | Dataset comparison |
| [**docs/DEPLOYMENT.md**](docs/DEPLOYMENT.md) | Deployment guide |

---

## 🎓 Key Learnings

### Technical Insights

1. **Feature Engineering > Algorithm Selection**
   - 3 hours on features > 10 hours tuning hyperparameters
   - Simple model (logistic regression) beat complex ensembles
   - Business-informed features captured signal better than raw data

2. **Production Thinking from Day 1**
   - Not just notebooks, but modular code
   - Error handling, logging, documentation
   - Monitoring-ready architecture

3. **Business Metrics > Technical Metrics**
   - ROI matters more than AUC
   - Stakeholders need interpretable models
   - Revenue impact > accuracy score

### Personal Growth

- **Learning in Public** - Documenting process builds credibility
- **Authentic Voice** - Personal projects should feel personal
- **Time Management** - Built while working + studying

See [LEARNING.md](LEARNING.md) for full reflections.

---

## 💼 Resume Bullets

> "Built production churn prediction system on IBM Telco dataset (7K customers), achieving 0.848 AUC with logistic regression and identifying $1.67M annual revenue at risk"

> "Engineered 58 production-grade features from raw customer data, demonstrating that good feature engineering beats complex algorithms - simple model outperformed ensemble methods"

> "Developed end-to-end ML pipeline from data ingestion to feature engineering to interactive dashboard, showing systems thinking and production-ready code quality"

> "Created interactive Streamlit dashboard with ROI calculator, customer risk leaderboard, and retention strategy recommendations for business stakeholders"

---

## 📊 Technology Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.11+ |
| **Database** | PostgreSQL 18 |
| **ML Framework** | scikit-learn, pandas, numpy |
| **Visualization** | Streamlit, Plotly |
| **Containerization** | Docker, Docker Compose |
| **API** | FastAPI (planned) |
| **Monitoring** | Evidently AI (planned) |

---

## 🤝 Connect With Me

**Christian G Callahan** (Red)

- 📧 Email: [contact@christiangcallahan.tech](mailto:contact@christiangcallahan.tech)
- 💼 LinkedIn: [christian--callahan](https://www.linkedin.com/in/christian--callahan/)
- 🐙 GitHub: [@CCallahan308](https://github.com/CCallahan308)
- 🌐 Website: [christiangcallahan.tech](https://www.christiangcallahan.tech/)

---

## 🙏 Acknowledgments

- **IBM** for the Telco Customer Churn dataset
- **Kaggle community** for dataset curation
- **Open-source ML community** (scikit-learn, pandas, XGBoost)
- My professors and peers at grad school

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📈 Project Stats

| Metric | Value |
|--------|-------|
| **Commits** | 10+ |
| **Files Created** | 35+ |
| **Lines of Code** | ~8,000 |
| **Documentation** | 10+ files |
| **Time Invested** | ~6 hours |
| **Features Engineered** | 58 |
| **Revenue at Risk** | $1.67M/year |
| **Model AUC** | 0.848 |

---

<div align="center">

**Built with 💼 by a grad student learning in public**

*This project isn't perfect. It's a work in progress by someone still learning. But that's the point - showing my process, the wins, the challenges, the iterations.*

*Because that's what real data science looks like.*

**[⬆ Back to Top](#signalforge---production-churn-intelligence)**

</div>
