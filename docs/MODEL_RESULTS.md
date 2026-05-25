# Model Results

This file describes **how** results are produced and where to find them. It deliberately does not
hard-code metric values: they depend on the dataset snapshot and the random seed, and are
regenerated every run.

## How to reproduce

```bash
python scripts/generate_sample_data.py     # or: download_real_data.py --source telco
python scripts/engineer_real_features.py
python scripts/train_with_optuna.py
```

Outputs (written to `models/artifacts/`):

- `training_results.json` — per-model test metrics (AUC, precision, recall, F1, Brier), bootstrap
  95% CIs, CV mean/std, paired-t p-values vs. Logistic Regression, tuned Optuna params, and the
  top standardized logistic-regression coefficients.
- `model_comparison.csv` — the three models side by side (`Model, AUC, CI_Lower, CI_Upper,
  Precision, Recall, F1, P_Value`).

## Methodology

- **Split first, score once.** A stratified train/test split is made before any tuning; the test
  set is used only for the final reported metrics.
- **Leak-free CV.** 5-fold stratified CV on the training split drives model selection. Logistic
  Regression scales inside the CV `Pipeline`, so the scaler is refit per fold.
- **Tuning.** Optuna (TPE, seeded) — trial count is configurable via `SIGNALFORGE_N_TRIALS`.
- **Uncertainty.** Bootstrap 95% CIs (1000 resamples) on the test set.
- **Comparison.** Paired t-tests on per-fold CV AUC vs. Logistic Regression. Caveat: k-fold CV
  scores are not independent, so this understates variance — treat it as a rough check, not proof.
- **Calibration.** Brier score, so a model that ranks well but gives poor probabilities is caught.
- **Feature effects.** Standardized logistic-regression coefficients (the selected model), not a
  separately-fit surrogate.

## Results

IBM Telco, 7,043 customers (26.5% churn). Held-out 20% test set, scored once. 95% CIs are bootstrap
(1,000 resamples); p-values are paired t-tests on per-fold CV AUC vs. Logistic Regression.

| Model | Test AUC | 95% CI | CV AUC | Brier | Precision | Recall | F1 | p vs LR |
|-------|---------:|--------|-------:|------:|----------:|-------:|---:|--------:|
| Logistic Regression | **0.849** | [0.827, 0.869] | 0.849 ± 0.013 | 0.164 | 0.500 | 0.791 | 0.613 | — |
| Gradient Boosting | 0.847 | [0.825, 0.866] | 0.849 ± 0.012 | **0.135** | 0.650 | 0.516 | 0.575 | 0.46 |
| Random Forest | 0.846 | [0.822, 0.866] | 0.846 ± 0.012 | 0.160 | 0.533 | 0.789 | 0.636 | 0.03 |

Top standardized logistic-regression coefficients (log-odds; tenure terms are collinear so their
weight splits across `log_tenure` / `tenure_squared` / `tenure`):

| Feature | Coefficient |
|---------|------------:|
| `log_tenure` | −1.43 |
| `tenure_squared` | −1.40 |
| `is_fiber_optic` | +1.01 |
| `contract_tenure_risk` | +0.67 |
| `service_count` | +0.57 |

## Finding

The three models sit within ~0.003 AUC of each other with overlapping 95% CIs — within noise
(GB indistinguishable from LR at p=0.46; RF's p=0.03 comes with overlapping CIs). Logistic
Regression is the primary model (best discrimination, ~79% recall, interpretable). Gradient
Boosting is best-calibrated (Brier 0.135 vs 0.164) and is preferable when the predicted probability
is consumed directly. Model selection here is a calibration/interpretability decision, not an
accuracy contest.
