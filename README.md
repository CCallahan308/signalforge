# SignalForge

Production ML system for predicting customer churn with statistical rigor.

**[Live Dashboard →](https://signalforge-ccallahan308.streamlit.app/)** • **[Project Page →](https://christiangcallahan.tech/projects/signalforge)** • **[GitHub Repo →](https://github.com/CCallahan308/signalforge)**

## Real Data (Not Synthetic)

This project uses the **IBM Telco Customer Churn dataset** from Kaggle:

**Why this dataset?**
- **7,043 real customers** - not synthetic data
- **Real business scenario** - telecom company trying to reduce churn
- **Messy data** - missing values, categorical variables, imbalanced classes
- **Real constraints** - budget limitations, intervention costs, ROI requirements

**Data pipeline:**
```
Kaggle API → Raw CSV (7,043 rows) → PostgreSQL (17 tables, 4 schemas) 
→ Feature Engineering (58 features) → Model Training → Dashboard
```

This is real customer data with real business constraints.

## Results

### Model Performance

| Model | AUC | 95% Confidence Interval | Significance |
|-------|-----|------------------------|--------------|
| Logistic Regression | 0.845 +/- 0.013 | [0.821, 0.863] | Best model |
| Random Forest | 0.843 +/- 0.012 | [0.821, 0.863] | p=0.25 vs LR |
| Gradient Boosting | 0.837 +/- 0.011 | [0.815, 0.859] | p=0.02 vs LR |

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

After: "Logistic Regression achieved 0.845 +/- 0.013 AUC [95% CI: 0.821, 0.863] using 5-fold cross-validation. RF was not significantly different (p=0.25), but GB was (p=0.02, paired t-test). LR wins on interpretability and statistical rigor."

---

Built by a grad student learning in public. Real data science is iterative, messy, and constantly improving.

That's what I wanted to show.
