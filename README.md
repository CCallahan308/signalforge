# SignalForge

Churn prediction on the IBM Telco Customer Churn dataset (7,043 customers), built to make
statistically honest model decisions instead of chasing a single accuracy number.

**~$139K/month in charges (~$1.67M/year) sits with customers who go dark.** The selected
model surfaces ~79% of them with interpretable coefficients, and the dashboard's ROI tab
turns intervention cost and save-rate assumptions into a what-if estimate.

**[Live demo](https://signalforge-ccallahan308.streamlit.app/)** · **[Project page](https://christiangcallahan.tech/projects/signalforge)** · **[GitHub](https://github.com/CCallahan308/signalforge)**

## What this is (and isn't)

A focused, reproducible churn-modeling project: feature engineering → leak-free cross-validated
model selection → an interactive dashboard. It is **not** a deployed production system — there is
no live API, database, or serving layer, and the repo doesn't pretend there is.

## Approach

- **Dataset:** IBM Telco Customer Churn (Kaggle) — a single 7,043-row snapshot, ~26% churn.
- **Features:** 47 engineered features across seven groups (tenure, contract, payment, service,
  demographic, financial, interactions), plus retained raw numeric fields and one-hot encodings —
  58 feature columns in total, of which models use the 51 numeric ones. Every engineered feature
  is row-local (depends only on a single customer's own values, no dataset-wide statistics), so
  nothing leaks across the split. See [docs/FEATURES.md](docs/FEATURES.md).
- **Models:** Logistic Regression, Random Forest, Gradient Boosting, each tuned with Optuna
  (TPE sampler, seeded).
- **Evaluation:**
  - Train/test split made first; the test set is scored exactly once.
  - 5-fold stratified CV on the training split for model selection. For Logistic Regression,
    scaling is refit inside each fold via an sklearn `Pipeline` (no preprocessing leakage).
  - Bootstrap 95% confidence intervals on the test set.
  - Paired t-tests on per-fold CV AUC for model comparison (caveat below).
  - Brier score for calibration.
  - Feature effects read from the standardized logistic-regression coefficients (the selected
    model itself), not a separate surrogate.

## Results

IBM Telco Customer Churn, 7,043 customers (26.5% churn). 5-fold stratified CV on the training split
for model selection; the metrics below are on the held-out 20% test set (scored once). 95% CIs are
bootstrap (1,000 resamples); p-values are paired t-tests on per-fold CV AUC vs. Logistic Regression.

| Model | Test AUC | 95% CI | CV AUC | Brier | Precision | Recall | F1 | p vs LR |
|-------|---------:|--------|-------:|------:|----------:|-------:|---:|--------:|
| Logistic Regression | **0.849** | [0.827, 0.869] | 0.849 ± 0.013 | 0.164 | 0.500 | 0.791 | 0.613 | — |
| Gradient Boosting | 0.847 | [0.825, 0.866] | 0.849 ± 0.012 | **0.135** | 0.650 | 0.516 | 0.575 | 0.46 |
| Random Forest | 0.846 | [0.822, 0.866] | 0.846 ± 0.012 | 0.160 | 0.533 | 0.789 | 0.636 | 0.03 |

*Reproduce: `python scripts/train_with_optuna.py` → `models/artifacts/training_results.json`. Exact
values shift slightly with the seed; the relationships hold.*

The three models land within ~0.003 AUC of each other with heavily overlapping 95% CIs — the
differences are within noise (GB is statistically indistinguishable from LR at p=0.46; RF's p=0.03
comes with fully overlapping CIs). **Logistic Regression** is the primary model: best discrimination,
strong recall (~79% of churners caught), and interpretable coefficients. **Gradient Boosting** is the
best-calibrated (Brier 0.135 vs 0.164), so it's preferable when the predicted probability is consumed
directly — e.g. expected-value ROI math. With good features, the choice here is a
calibration/interpretability decision, not an accuracy contest.

**Business framing (derived from the data, not a model claim):** ~$139K/month in charges come from
customers who churned (~$1.67M/year); surfacing ~79% of them flags ~$110K/month for retention
targeting. The dashboard's ROI tab turns intervention cost and save-rate assumptions into a what-if
estimate.

> These figures are from the corrected, leak-free pipeline (test set held out before tuning; LR
> scaling refit inside each CV fold). An earlier version reported similar AUCs from a pipeline that
> had CV leakage — the practical gap was small, but the corrected numbers are the honest ones.

## Quickstart

Requires Python 3.11.

```bash
git clone https://github.com/CCallahan308/signalforge
cd signalforge
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get the data — real or synthetic:

```bash
# Option A: real IBM Telco data (needs a free Kaggle API token in ~/.kaggle/kaggle.json)
python scripts/download_real_data.py --source telco

# Option B: synthetic sample, no credentials — for trying the pipeline / CI.
# NOT the project's real results.
python scripts/generate_sample_data.py
```

Run the pipeline and dashboard:

```bash
python scripts/engineer_real_features.py   # -> data/processed/features.parquet
python scripts/train_with_optuna.py        # -> models/artifacts/ (models, metrics, comparison)
streamlit run src/app/dashboard.py
```

### Docker

```bash
docker compose up --build   # dashboard at http://localhost:8501 (run the pipeline first so artifacts exist)
```

## Tests & CI

```bash
pytest
```

GitHub Actions lints with ruff, generates the synthetic sample, runs the full pipeline, and runs
the test suite — so CI is green from a clean clone without Kaggle credentials. Because CI runs on
synthetic data, the metrics it produces are not the project's real results.

## Project structure

```
signalforge/
├── scripts/
│   ├── generate_sample_data.py    # synthetic IBM-Telco-schema sample (no credentials)
│   ├── download_real_data.py      # Kaggle download (telco wired end-to-end; bank/saas download only)
│   ├── engineer_real_features.py  # raw CSV -> features.parquet
│   ├── train_with_optuna.py       # leak-free CV, tuning, metrics, artifacts
│   └── feature_summary.py
├── src/
│   ├── config.py                  # paths, seed, hyperparameters
│   └── app/
│       ├── dashboard.py           # renders the real trained artifacts
│       └── dashboard_demo.py      # static showcase (Streamlit Cloud)
├── tests/test_signalforge.py
├── notebooks/01_eda_telco.ipynb
└── docs/
```

## Limitations (honest)

- **Single snapshot dataset.** IBM Telco is one cross-sectional sample — no temporal/out-of-time
  validation or drift monitoring. A production system would need cohort data and retraining triggers.
- **Significance-test caveat.** The paired t-test runs on k-fold CV scores, which aren't
  independent, so it understates variance. It's a rough check; the "within noise" conclusion is
  what's load-bearing, not the exact p-value.
- **Two-stage features.** Feature engineering is a separate step that writes a Parquet file. It's
  leak-free (every feature is row-local; residual NaN/inf is filled with a constant), but it isn't
  a single fitted sklearn artifact — re-running on new data recomputes from scratch.
- **Demo dashboard is static.** `dashboard_demo.py` (the Streamlit Cloud demo) shows representative
  figures because the cloud host has no trained artifacts; `dashboard.py` renders the real ones.
- **Only telco is wired end-to-end.** `download_real_data.py` can fetch other datasets, but only
  the IBM Telco path flows through feature engineering and training.
- **Committed model pickles are version-sensitive.** The raw CSV, `features.parquet`, and trained
  models are committed so a clone is instantly runnable. The `.pkl` files were trained on
  scikit-learn 1.8.0; pickles aren't guaranteed to load across major scikit-learn versions — if you
  hit a load error, just re-run `python scripts/train_with_optuna.py` to regenerate them.

## Contact

Christian Callahan — [Portfolio](https://christiangcallahan.tech) · [GitHub](https://github.com/CCallahan308)

## Related churn work

Three retention projects, three different questions:

- **This repo** — *which model, and is the difference real?* (bootstrap CIs, paired tests, calibration)
- [Churn ROI Simulator](https://github.com/CCallahan308/churn-roi-simulator) — *what is a churn score worth in dollars when the base rate caps lift?* (retention-budget ROI simulator)
- [Ecommerce Retention & Growth](https://github.com/CCallahan308/ecommerce-retention-growth) — *which customers to win back, at what LTV?* (KKBox segmentation)

## License

MIT — see [LICENSE](LICENSE).
