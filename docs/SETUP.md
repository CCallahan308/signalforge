# Setup Guide

SignalForge runs as a file-based pipeline: get data → engineer features → train → dashboard.
There is no database or API to configure.

## 1. Environment

Requires Python 3.11.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get data

### Option A — real IBM Telco dataset (recommended for real metrics)

Needs a free Kaggle API token:

1. Go to https://www.kaggle.com/settings → "API" → "Create New API Token" (downloads `kaggle.json`).
2. Place it at `~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\kaggle.json`).
   On Linux/Mac: `chmod 600 ~/.kaggle/kaggle.json`.

```bash
python scripts/download_real_data.py --source telco
```

This writes `data/raw/WA_Fn_UseC_Telco_Customer_Churn.csv`, which the feature step reads.

### Option B — synthetic sample (no credentials)

For trying the pipeline or running CI without Kaggle. The resulting metrics are **not** the
project's real results.

```bash
python scripts/generate_sample_data.py
```

## 3. Run the pipeline

```bash
python scripts/engineer_real_features.py   # -> data/processed/features.parquet
python scripts/train_with_optuna.py        # -> models/artifacts/ (models, metrics, comparison)
streamlit run src/app/dashboard.py         # dashboard at http://localhost:8501
```

`scripts/feature_summary.py` prints a quick summary of the engineered features.

## Datasets

`download_real_data.py` lists several churn datasets (`--list`), but only **telco** is wired
through feature engineering and training end-to-end. `bank`/`saas`/`cell2cell` are download-only.

| Dataset | Source | Rows | Notes |
|---------|--------|------|-------|
| **telco** | Kaggle (IBM) | 7,043 | Classic benchmark; the supported pipeline |
| bank | Kaggle | 10,000 | Download only |
| saas | Kaggle | 10,000 | Download only |
| cell2cell | Kaggle | 51,047 | Download only |

## Troubleshooting

- **`FileNotFoundError` in `engineer_real_features.py`:** you haven't produced the raw CSV yet —
  run Option A or Option B above.
- **Kaggle 403:** accept the dataset's terms on its Kaggle page and regenerate the API token.
- **Parquet errors:** ensure `pyarrow` installed (it's in `requirements.txt`).
