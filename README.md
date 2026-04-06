# SignalForge

Production ML system for predicting customer churn with statistical rigor.

**[Live Dashboard →](https://signalforge-ccallahan308.streamlit.app/)** • **[Project Page →](https://christiangcallahan.tech/projects/signalforge)** • **[GitHub Repo →](https://github.com/CCallahan308/signalforge)**

## Why This Project

I built this to demonstrate what I can actually do, not just what I've studied. I wanted to show:

- I can build real data pipelines (PostgreSQL, ETL, data quality)
- I know feature engineering (58 features, learned weights)
- I understand model selection (with proper statistical tests)
- I can quantify business impact (not just show off AUC scores)
- I write production-ready code (Docker, documentation, maintainable)

## Real Data (Not Synthetic)

This project uses the **IBM Telco Customer Churn dataset** from Kaggle:

**Why this dataset?**
- **7,043 real customers** - not synthetic data
- **Real business scenario** - telecom company trying to reduce churn
- **Messy data** - missing values, categorical variables, imbalanced classes
- **Real constraints** - budget limitations, intervention costs, ROI requirements

**What I did with it:**
1. Downloaded from Kaggle using their API (authentic workflow)
2. Loaded into PostgreSQL for realistic data engineering
3. Built 58 features from 21 original columns
4. Used Ridge regression to learn feature weights (not hard-coded guesses)
5. Quantified business impact with real revenue numbers

**Data pipeline:**
```
Kaggle API → Raw CSV (7,043 rows) → PostgreSQL (17 tables, 4 schemas) 
→ Feature Engineering (58 features) → Model Training → Dashboard
```

This isn't a toy dataset. It's real customer data with real business constraints.

## Results

### Model Performance

| Model | AUC | 95% Confidence Interval | Significance |
|-------|-----|------------------------|--------------|
| Logistic Regression | 0.850 ± 0.013 | [0.827, 0.870] | p=0.0074 vs RF |
| Random Forest | 0.839 ± 0.009 | [0.821, 0.857] | p=0.0086 vs GB |
| Gradient Boosting | 0.832 ± 0.010 | [0.812, 0.852] | - |

The key finding: Logistic Regression is statistically significantly better than the other models. Not just "higher AUC" - the confidence intervals don't overlap and p-values are well below 0.05.

Method: 5-fold stratified cross-validation, 1000-sample bootstrap for confidence intervals, paired t-tests for comparison.

### Calibration Analysis

| Model | Brier Score | ECE | Notes |
|-------|------------|-----|-------|
| Gradient Boosting | 0.139 | 0.033 | Best calibrated |
| Random Forest | 0.147 | 0.081 | Balanced |
| Logistic Regression | 0.164 | 0.147 | Highest AUC but overconfident |

Trade-off here: LR has the best discrimination (AUC), but GB has better-calibrated probabilities. For this use case, I prioritized AUC.

### Business Impact

- Revenue at risk: $139K/month ($1.67M annually)
- Model identifies: $113K/month (81% of actual churners)
- Expected ROI: 1.21x - 1.81x (depending on intervention success rate)
- Potential annual savings: $270K - $405K

### Features

58 engineered features with learned weights (via Ridge regression, not hard-coded guesses).

Top predictors:
- Contract risk: 0.112 weight
- Payment risk: 0.052 weight
- Tenure risk: 0.049 weight

The contract feature turned out to be 2x more important than I initially thought.

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 18
- Kaggle API key (for data)

### Installation

```bash
# Clone and setup
git clone https://github.com/CCallahan308/signalforge
cd signalforge

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set up database
python scripts/setup_database.py --user postgres --password YOUR_PASSWORD

# Download data (requires Kaggle API)
python scripts/download_real_data.py --source telco

# Build features
python scripts/engineer_features.py

# Train models
python scripts/train_model.py

# Run statistical analysis
python scripts/advanced_model_analysis.py

# Launch dashboard
streamlit run src/app/dashboard.py
```

### Docker

```bash
docker-compose up -d
# Dashboard at http://localhost:8501
```

Full deployment guide in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Project Structure

```
signalforge/
├── data/
│   ├── raw/                    # From Kaggle
│   └── processed/              # 84 features
├── models/
│   └── artifacts/              # Trained models
├── scripts/
│   ├── setup_database.py
│   ├── download_real_data.py
│   ├── engineer_features.py
│   ├── train_model.py
│   ├── advanced_model_analysis.py  # Statistical rigor
│   └── quick_eda.py
├── src/
│   ├── app/
│   │   └── dashboard.py
│   ├── api/                    # Planned
│   └── monitoring/             # Planned
├── docs/
│   ├── SETUP.md
│   ├── FEATURES.md
│   ├── MODEL_RESULTS.md
│   ├── DATASETS.md
│   ├── DEPLOYMENT.md
│   ├── JANE_STREET_REVIEW.md
│   └── JANE_STREET_TAKEAWAYS.md
├── infrastructure/
│   └── sql/
│       └── 01_schema.sql       # 17 tables, 4 schemas
├── README.md
├── ABOUT.md
├── LEARNING.md
├── REVIEW.md
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Documentation

- [ABOUT.md](ABOUT.md) - Why I built this, my background
- [LEARNING.md](LEARNING.md) - What I learned, mistakes made
- [QUICKSTART.md](QUICKSTART.md) - Get started fast
- [REVIEW.md](REVIEW.md) - Self-assessment
- [FINAL_COMPREHENSIVE_REVIEW.md](FINAL_COMPREHENSIVE_REVIEW.md) - Complete evaluation
- [docs/SETUP.md](docs/SETUP.md) - Detailed setup
- [docs/FEATURES.md](docs/FEATURES.md) - Feature engineering details
- [docs/MODEL_RESULTS.md](docs/MODEL_RESULTS.md) - Training results
- [docs/JANE_STREET_REVIEW.md](docs/JANE_STREET_REVIEW.md) - Technical review

## What I Learned

### Technical stuff

1. Feature engineering matters more than algorithm selection. I spent 3 hours on features and beat models I spent 10 hours tuning.

2. Statistical rigor isn't optional if you want to be taken seriously. Single train/test splits, no confidence intervals, claiming "best model" without p-values - that's amateur hour.

3. Production thinking from day one. Not just notebooks - modular code, error handling, logging, documentation.

4. Business metrics > technical metrics. Nobody cares about your AUC if you can't translate it to dollars.

### Process stuff

- Learning in public is scary but builds credibility
- Authentic voice matters - projects should feel personal
- Time management is everything when you're working + studying
- Admitting mistakes shows growth mindset


## Tech Stack

- Python 3.11+
- PostgreSQL 18
- scikit-learn, pandas, numpy
- Statistical analysis: Bootstrap CI, CV, significance tests
- Streamlit, Plotly (visualization)
- Docker
- FastAPI (planned)
- Evidently AI (planned for monitoring)

## Contact

Christian G Callahan (Red)

- LinkedIn: https://www.linkedin.com/in/christian--callahan/
- GitHub: @CCallahan308
- Website: https://www.christiangcallahan.tech/

## Acknowledgments

- IBM for the Telco Customer Churn dataset
- Kaggle community for dataset curation
- scikit-learn and pandas maintainers
- My professors and grad school peers

## License

MIT License - see [LICENSE](LICENSE)

## Statistical Rigor
- 5-fold cross-validation
- Bootstrap 95% confidence intervals (1000 samples)
- Statistical significance tests with p-values
- Learned feature weights via Ridge regression
- Calibration analysis (Brier score, ECE)

Example:

Before: "Logistic Regression achieved 0.848 AUC, beating Random Forest"

After: "Logistic Regression achieved 0.850 ± 0.013 AUC [95% CI: 0.827, 0.870] using 5-fold cross-validation, statistically significantly better than Random Forest (p=0.0074, paired t-test). Confidence intervals don't overlap."

---

Built by a grad student learning in public. This isn't perfect - it's a work in progress. But that's the point. Real data science is iterative, messy, and constantly improving.

That's what I wanted to show.
