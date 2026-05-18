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

## Architecture Notes

- **Temporal data**: The IBM Telco dataset is a single snapshot. A production system would use monthly cohort data to enable drift detection and retraining triggers.
- **Cost-sensitive learning**: Integrating business cost matrices into model training directly would improve ROI optimization over post-hoc threshold tuning.
- **Stateless deployment**: The dashboard is designed to load pre-computed artifacts, making it suitable for stateless cloud deployment without a live database dependency.

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 16+
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
python scripts/engineer_real_features.py

# Train models (Optuna-tuned, with statistical analysis)
python scripts/train_with_optuna.py

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
│   ├── raw/                    # From Kaggle (not tracked)
│   └── processed/              # 58 engineered features (not tracked)
├── models/
│   └── artifacts/              # Trained models (not tracked)
├── scripts/
│   ├── setup_database.py
│   ├── download_real_data.py
│   ├── engineer_real_features.py
│   └── train_with_optuna.py
├── src/
│   ├── app/
│   │   ├── dashboard.py
│   │   └── dashboard_demo.py
│   ├── api/                    # Planned
│   └── monitoring/             # Planned
├── docs/
│   ├── SETUP.md
│   ├── FEATURES.md
│   ├── MODEL_RESULTS.md
│   ├── DATASETS.md
│   └── DEPLOYMENT.md
├── infrastructure/
│   └── sql/
│       └── 01_schema.sql       # 17 tables, 4 schemas
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Tech Stack

- Python 3.11+ / PostgreSQL 16+
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
- [LinkedIn](https://www.linkedin.com/in/christiangcallahan/)
- [GitHub](https://github.com/CCallahan308)

## License

MIT License - see [LICENSE](LICENSE)
