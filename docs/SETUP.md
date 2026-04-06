# SignalForge Setup Guide

## Prerequisites

### 1. Kaggle API Setup

To download real datasets from Kaggle:

```bash
# Install Kaggle CLI
pip install kaggle

# Get API credentials:
# 1. Go to https://www.kaggle.com/settings
# 2. Scroll to "API" section
# 3. Click "Create New API Token"
# 4. This downloads kaggle.json

# Place kaggle.json in the right location
# Windows: C:\Users\<username>\.kaggle\kaggle.json
# Linux/Mac: ~/.kaggle/kaggle.json

# Set permissions (Linux/Mac only)
chmod 600 ~/.kaggle/kaggle.json
```

### 2. PostgreSQL Setup

PostgreSQL should already be installed (we set this up earlier).

**Connection Details:**
- Host: `localhost`
- Port: `5432`
- Database: `signalforge`
- User: `signalforge_user`
- Password: `SignalForge2024!`

## Quick Start

### Step 1: Install Dependencies

```bash
cd C:\Users\Calla\signalforge

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install core dependencies
pip install pandas numpy loguru scikit-learn xgboost
pip install psycopg2-binary sqlalchemy
pip install kaggle  # For downloading datasets
```

### Step 2: Download Real Data

```bash
# List available datasets
python scripts/download_real_data.py --list

# Download Telco churn dataset (recommended for first run)
python scripts/download_real_data.py --source telco

# Or download Bank churn dataset
python scripts/download_real_data.py --source bank

# Or download SaaS churn dataset
python scripts/download_real_data.py --source saas
```

**Available Datasets:**

| Dataset | Source | Rows | Description |
|---------|--------|------|-------------|
| **telco** | Kaggle (IBM) | 7,043 | Telecom customer churn - classic benchmark |
| **bank** | Kaggle | 10,000 | Bank customer churn with credit scores |
| **saas** | Kaggle | 10,000 | Multi-table SaaS subscription data |
| **cell2cell** | Kaggle | 51,047 | Large-scale telecom churn |

### Step 3: Load Data to PostgreSQL

```bash
python scripts/load_to_database.py --source telco
```

### Step 4: Train Model

```bash
python scripts/train_model.py --source telco
```

### Step 5: Start API

```bash
uvicorn src.api.main:app --reload
```

### Step 6: Launch Dashboard

```bash
streamlit run src/app/dashboard.py
```

## Available Datasets

### 1. Telco Customer Churn (RECOMMENDED)

**Source:** IBM Sample Dataset  
**Rows:** 7,043  
**Best for:** First-time setup, benchmarking

**Features:**
- Demographics (age, partner, dependents)
- Services (internet, streaming, security)
- Billing (monthly charges, contract type)
- Tenure and payment methods

**Churn Rate:** ~26.5%

**Why use this:**
- ✅ Well-documented
- ✅ Classic benchmark (shows you know the standards)
- ✅ Good for portfolio interviews
- ✅ Real business patterns

**Download:**
```bash
python scripts/download_real_data.py --source telco
```

### 2. Bank Customer Churn

**Source:** Kaggle Playground Series 2024  
**Rows:** 10,000  
**Best for:** Financial services roles

**Features:**
- Credit score
- Geography (France, Germany, Spain)
- Account balance
- Number of products
- Activity status

**Churn Rate:** ~20.4%

**Why use this:**
- ✅ Recent dataset (2024)
- ✅ Shows fintech domain knowledge
- ✅ Good for banking roles

**Download:**
```bash
python scripts/download_real_data.py --source bank
```

### 3. SaaS Subscription Churn

**Source:** Kaggle  
**Rows:** 10,000  
**Best for:** SaaS company roles

**Features:**
- Subscription lifecycle
- Feature usage patterns
- Support ticket history
- Multi-table relationships

**Why use this:**
- ✅ Multi-table (shows SQL skills)
- ✅ SaaS-specific patterns
- ✅ Real subscription data model

**Download:**
```bash
python scripts/download_real_data.py --source saas
```

### 4. Cell2Cell Telecom Churn

**Source:** Duke University  
**Rows:** 51,047  
**Best for:** Large-scale ML, production systems

**Features:**
- Detailed usage metrics
- Equipment data
- Large customer base
- Real telecom data

**Why use this:**
- ✅ Large dataset (shows scalability)
- ✅ Real academic benchmark
- ✅ Production-ready size

**Download:**
```bash
python scripts/download_real_data.py --source cell2cell
```

## Project Structure After Setup

```
signalforge/
├── data/
│   ├── raw/
│   │   ├── telco/                  # Downloaded from Kaggle
│   │   │   ├── WA_Fn_UseC_Telco_Customer_Churn.csv
│   │   ├── telco_processed.csv     # Cleaned data
│   │   └── telco_metadata.json     # Dataset info
│   └── processed/
│       ├── features.parquet        # Engineered features
│       └── train_test_split/       # CV splits
├── models/
│   └── artifacts/
│       ├── churn_model_v1.pkl
│       └── feature_importance.json
├── logs/
│   └── training.log
└── notebooks/
    └── exploratory_analysis.ipynb
```

## Next Steps

After downloading data:

1. **Exploratory Analysis**
   ```bash
   jupyter notebook notebooks/exploratory_analysis.ipynb
   ```

2. **Feature Engineering**
   ```bash
   python scripts/engineer_features.py --source telco
   ```

3. **Model Training**
   ```bash
   python scripts/train_model.py --source telco --model xgboost
   ```

4. **Model Evaluation**
   ```bash
   python scripts/evaluate_model.py --model models/artifacts/churn_model_v1.pkl
   ```

5. **Deploy API**
   ```bash
   docker-compose up -d
   ```

## Troubleshooting

### Kaggle API Issues

**Error: "kaggle.json not found"**
```bash
# Make sure kaggle.json is in the right place
# Windows: C:\Users\<username>\.kaggle\kaggle.json
# Verify:
dir C:\Users\Calla\.kaggle\kaggle.json
```

**Error: "Permission denied"**
```bash
# Windows: No action needed
# Linux/Mac:
chmod 600 ~/.kaggle/kaggle.json
```

**Error: "403 Forbidden"**
- Go to https://www.kaggle.com/settings
- Accept Kaggle terms of service
- Regenerate API token

### PostgreSQL Issues

**Error: "Connection refused"**
```bash
# Check if PostgreSQL is running
# Windows:
Get-Service -Name postgresql*

# Start if stopped:
Start-Service postgresql-x64-18
```

**Error: "Database signalforge does not exist"**
```bash
# Re-run setup script
python scripts/setup_database.py --user postgres --password "@D@mnati0n123"
```

### Python Dependencies Issues

**Error: "Module not found"**
```bash
# Make sure virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

## Advanced Setup

### Using OpenML Datasets

```python
from sklearn.datasets import fetch_openml

# Download from OpenML
churn = fetch_openml(name='churn', version=1, as_frame=True)
df = churn.frame
```

### Using AWS Open Data

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure

# Download from S3
aws s3 sync s3://aws-open-data/datasets/churn/ data/raw/aws/
```

## What This Shows Employers

✅ **Real Data Engineering:** Downloading from APIs, handling authentication  
✅ **Data Quality:** Real datasets have missing values, outliers  
✅ **Domain Knowledge:** Telco, banking, SaaS industries  
✅ **Best Practices:** Using established benchmarks (IBM, Kaggle)  
✅ **Production Thinking:** Choosing appropriate dataset sizes  

---

**Questions?** Check the main README.md or open an issue on GitHub.
