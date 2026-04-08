# SignalForge

Production churn prediction with statistical rigor.

**[Live Dashboard](https://signalforge-ccallahan308.streamlit.app/)** • **[Project Page](https://christiangcallahan.tech/projects/signalforge)** • **[GitHub](https://github.com/CCallahan308/signalforge)**

## What I Built

A churn prediction system for a telecom company (IBM Telco dataset, 7,043 customers) with one goal: make statistically sound decisions about which customers to save and how much to spend doing it.

## Results

### Model Performance (5-fold CV, Optuna-tuned)

| Model | AUC | 95% CI | vs LR |
|-------|-----|--------|-------|
| Logistic Regression | 0.849 ± 0.012 | [0.828, 0.869] | Baseline |
| Gradient Boosting | 0.846 ± 0.010 | [0.827, 0.866] | p=0.130 |
| Random Forest | 0.844 ± 0.010 | [0.825, 0.863] | p=0.016 |

Logistic Regression won on discrimination. Gradient Boosting was better calibrated (Brier 0.139 vs 0.164). The right model depends on whether you need ranking or probability accuracy.

### Business Impact

- **$1.67M** annual revenue at risk
- **$113K/month** identified by model (81% of actual churners)
- **1.21x–1.81x** expected ROI from retention interventions
- **$270K–$405K** potential annual savings

### Features

58 engineered features with learned weights via Ridge regression. Top predictor: contract type (month-to-month = 3.8x churn), which turned out to be 2x more important than I initially assumed.

## What I Learned

1. **Feature engineering beats tuning.** The biggest AUC gains came from building good risk features, not from Optuna sweeps.
2. **Simple models compete.** Logistic Regression beat two ensemble methods. Interpretability was a free bonus.
3. **Confidence intervals change decisions.** Without CIs, the AUC difference between LR and RF looks meaningful. With them (overlapping), you realize it's borderline.
4. **Calibration matters for business use.** A model that ranks well but gives garbage probabilities leads to bad ROI estimates.

## What I'd Do Differently

- The dataset is a single snapshot. A temporal dataset with monthly cohorts would be more realistic and allow drift detection.
- I'd add cost-sensitive learning upfront instead of post-hoc ROI calculations.
- The dashboard needs pre-computed artifacts, not a live database. I'd restructure for stateless deployment.

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
source venv/bin/bin/activate  # Windows: venv\Scripts\activate
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
│   └── processed/              # 54 features
├── models/
│   └── artifacts/              # Trained models
├── scripts/
│   ├── setup_database.py
│   ├── download_real_data.py
│   ├── engineer_features.py
│   ├── train_model.py
│   ├── advanced_model_analysis.py
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

## Tech Stack

- Python 3.11+ / PostgreSQL 18
- scikit-learn, pandas, numpy
- Optuna (hyperparameter tuning)
- Streamlit, Plotly
- Docker
- Statistical analysis: Bootstrap CI, 5-fold CV, significance testing, calibration

## Statistical Methods

- 5-fold stratified cross-validation
- Bootstrap 95% confidence intervals (1000 samples)
- Paired t-tests for model comparison
- Calibration analysis (Brier score, ECE)
- Optuna Bayesian optimization (20 trials per model)
- Learned feature weights via Ridge regression (L2)

## Contact

Christian Callahan

- [Portfolio](https://christiangcallahan.tech)
- [LinkedIn](https://www.linkedin.com/in/christian--callahan/)
- [GitHub](https://github.com/CCallahan308)

## License

MIT License - see [LICENSE](LICENSE)
