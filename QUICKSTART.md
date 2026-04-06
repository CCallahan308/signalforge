# SignalForge - Quick Start Guide

## 🚀 Get Started in 3 Commands

```bash
# 1. Activate environment
cd C:\Users\Calla\signalforge
.\venv\Scripts\Activate.ps1

# 2. Run quickstart
python scripts/quickstart.py --dataset telco

# 3. Start building!
jupyter notebook
```

---

## 📊 Available Datasets

| Dataset | Best For | Rows | Download Command |
|---------|----------|------|------------------|
| **Telco** | First project, interviews | 7K | `--dataset telco` |
| **Bank** | Fintech roles | 10K | `--dataset bank` |
| **SaaS** | SQL skills | 10K | `--dataset saas` |

**Recommendation:** Start with **Telco** - it's the industry standard benchmark.

---

## 📁 Project Structure

```
signalforge/
├── data/
│   ├── raw/              # Downloaded datasets
│   └── processed/        # Cleaned + features
├── models/
│   └── artifacts/        # Trained models
├── scripts/
│   ├── quickstart.py     # One-command setup
│   ├── download_real_data.py
│   ├── engineer_features.py
│   └── train_model.py
├── docs/
│   ├── SETUP.md          # Full setup guide
│   └── DATASETS.md       # Dataset comparison
└── src/
    ├── api/              # FastAPI
    ├── models/           # ML code
    └── app/              # Streamlit dashboard
```

---

## 🎯 What This Shows Employers

✅ **Real Data** - Not synthetic, real benchmarks  
✅ **End-to-End** - Data → Features → Model → API  
✅ **Production** - PostgreSQL, Docker, monitoring  
✅ **MLOps** - Model versioning, drift detection  
✅ **Business Impact** - ROI, intervention analysis  

---

## 📝 Resume Bullets

After completion, you'll have bullets like:

> "Built production churn prediction system on IBM Telco benchmark achieving 0.82 AUC, deployed via FastAPI with <100ms latency and integrated drift detection monitoring"

---

## 🛠️ Prerequisites

Before running quickstart:

1. **Kaggle API** (for downloading datasets)
   ```bash
   pip install kaggle
   # Get API key from https://www.kaggle.com/settings
   # Place in C:\Users\Calla\.kaggle\kaggle.json
   ```

2. **PostgreSQL** (already set up ✅)
   - Database: `signalforge`
   - User: `signalforge_user`
   - Password: `SignalForge2024!`

3. **Python packages**
   ```bash
   pip install pandas numpy scikit-learn xgboost loguru
   ```

---

## ⚡ Quick Commands

```bash
# Download specific dataset
python scripts/download_real_data.py --source telco

# Engineer features
python scripts/engineer_features.py --source telco

# Train model
python scripts/train_model.py --source telco --model xgboost

# Start API
uvicorn src.api.main:app --reload

# Start dashboard
streamlit run src/app/dashboard.py
```

---

## 📚 Documentation

- **Full Setup Guide:** `docs/SETUP.md`
- **Dataset Comparison:** `docs/DATASETS.md`
- **Architecture:** `README.md`

---

## 🐛 Troubleshooting

### Kaggle API Error

```
Error: "kaggle.json not found"
```

**Fix:**
1. Go to https://www.kaggle.com/settings
2. Click "Create New API Token"
3. Move downloaded `kaggle.json` to `C:\Users\Calla\.kaggle\`

### PostgreSQL Error

```
Error: "Connection refused"
```

**Fix:**
```bash
Start-Service postgresql-x64-18
```

---

## 🎓 Learning Path

1. **Day 1-2:** Download data, explore in Jupyter
2. **Day 3-4:** Feature engineering
3. **Day 5-6:** Model training + tuning
4. **Day 7-8:** API development
5. **Day 9-10:** Dashboard + monitoring
6. **Day 11-12:** Documentation + portfolio polish

---

## 💼 Portfolio Strategy

This project demonstrates:
- ✅ Data engineering (PostgreSQL, ETL)
- ✅ Machine learning (feature engineering, modeling)
- ✅ MLOps (monitoring, versioning)
- ✅ Software engineering (API, Docker)
- ✅ Business thinking (ROI, metrics)

**Target roles:**
- Senior Data Scientist ($180-250k)
- ML Engineer ($170-240k)
- Data Science Manager ($200-300k)

---

**Questions?** Check `docs/SETUP.md` or open a GitHub issue.

**Ready?** Run the quickstart! 🚀
