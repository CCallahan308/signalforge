# Feature Engineering

`scripts/engineer_real_features.py` turns the raw IBM Telco CSV (7,043 rows) into a feature table
written to `data/processed/features.parquet`.

## What it produces

- **47 engineered features** across seven groups (below).
- **4 retained raw numeric fields:** `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`.
- **7 one-hot columns** from `InternetService`, `Contract`, `PaymentMethod` (`drop_first=True`).

That's **58 feature columns** (plus the `churned` target). Models train on the **51 numeric**
columns; the boolean one-hot columns are excluded because they're redundant with explicit
indicator features (e.g. `is_fiber_optic`, `is_month_to_month`, `is_electronic_check`).

Every engineered feature is **row-local** — it depends only on one customer's own values, never on
dataset-wide statistics — so nothing leaks across the train/test split. Residual NaN/inf is filled
with the constant `0` (not a data-derived median). See the module docstring for the leakage notes.

## Groups

### Tenure (8)
`tenure_years`, `is_new_customer` (≤12mo), `is_very_new` (≤6mo), `is_established` (>24mo),
`is_long_term` (>48mo), `tenure_risk` (1 − tenure/72), `tenure_squared`, `log_tenure`.

### Contract (4)
`is_month_to_month`, `is_one_year`, `is_two_year`, `contract_risk` (1.0 / 0.5 / 0.0 by term).

### Payment (5)
`is_auto_payment`, `is_electronic_check`, `is_paperless`, `payment_risk` (by method),
`payment_stability`.

### Service (13)
`service_count`, `has_security`, `has_tech_support`, `has_backup`, `has_protection`,
`has_internet`, `is_fiber_optic`, `no_internet`, `has_phone`, `has_multiple_lines`,
`service_adoption_rate`, `no_protection`, `no_support`.

### Demographic (6)
`is_senior`, `has_partner`, `has_dependents`, `is_single`, `family_size`, `is_male`.

### Financial (6)
`avg_monthly_from_total`, `charge_trend`, `charge_trend_pct`, `total_revenue_per_month`,
`lifetime_value`, `charge_per_service`.

> Earlier versions included `is_high_charge` (global median threshold), `mrr_percentile`
> (global rank), and `price_sensitivity` (derived from `is_high_charge`). These used dataset-wide
> statistics computed before the split and were **removed** because they leaked test information.

### Interaction (5)
`contract_tenure_risk`, `payment_service_risk`, `value_at_risk`, `engagement_score`,
`churn_risk_score` (a weighted composite of contract, tenure, payment, service-adoption, and
single-household risk).

## Run

```bash
python scripts/engineer_real_features.py     # reads data/raw/WA_Fn_UseC_Telco_Customer_Churn.csv
python scripts/feature_summary.py            # prints types, correlations, and a sample
```

Outputs: `data/processed/features.parquet`, `features.csv`, and `feature_metadata.json`
(record/feature counts and the group breakdown).
