# SignalForge - Production SaaS Churn Intelligence

> Production-grade churn prediction system with uplift modeling, MLOps, and business impact quantification.

## 🎯 Project Overview

**SignalForge** is a senior-portfolio data science project demonstrating:
- End-to-end ML systems
- Production MLOps (monitoring, drift, retraining)
- Business impact quantification (ROI, revenue)
- Scalable architecture

**Tech Stack:** Python • PostgreSQL • FastAPI • MLflow • Airflow • Docker • Streamlit

## 📊 Project Goals

| Metric | Target |
|--------|--------|
| Churn AUC | 0.66+ |
| Uplift Correlation | 0.65+ |
| Top Decile Lift | 2.0x+ |
| Prediction Latency | <100ms |
| ROI (demo) | 20x+ |

## 🏗️ Architecture

```
signalforge/
├── src/
│   ├── data/           # Data pipeline (ETL, feature engineering)
│   ├── models/         # ML models (churn, uplift)
│   ├── api/            # FastAPI serving layer
│   ├── monitoring/     # Drift detection, performance tracking
│   └── app/            # Streamlit dashboard
├── infrastructure/
│   ├── docker/         # Docker configurations
│   ├── airflow/        # DAGs and orchestration
│   └── sql/            # PostgreSQL schemas
├── tests/              # pytest + data validation
├── docs/               # Documentation
└── scripts/            # Utility scripts
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 18
- Docker & Docker Compose

### Setup

```bash
# Clone repo
git clone https://github.com/CCallahan308/signalforge
cd signalforge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up database
python scripts/setup_database.py

# Generate synthetic data
python scripts/generate_data.py --accounts 10000

# Train models
python scripts/train_models.py

# Start API
uvicorn src.api.main:app --reload

# Start dashboard
streamlit run src/app/dashboard.py
```

## 📈 Key Features

### 1. Churn Risk Model
- Gradient boosting classifier
- Real-time feature computation
- Weekly model retraining

### 2. Uplift Modeling
- Causal inference for intervention targeting
- Identifies saveable customers
- ROI-based intervention prioritization

### 3. MLOps Pipeline
- Data drift detection (Evidently AI)
- Model performance monitoring
- Automated retraining triggers

### 4. Business Intelligence
- Customer 360 health scores
- Intervention playbook recommendations
- ROI simulator

## 📊 Model Performance

| Model | AUC | Precision | Recall | F1 |
|-------|-----|-----------|--------|-----|
| Churn Risk | 0.663 | 0.72 | 0.68 | 0.70 |
| Uplift | 0.648 | - | - | - |

## 🛠️ API Endpoints

```
GET  /health                    # Health check
POST /predict/churn             # Churn risk score
POST /predict/uplift            # Uplift score
GET  /customers/{id}            # Customer 360 view
POST /interventions/recommend   # Intervention recommendations
GET  /metrics/business          # Business KPIs
```

## 📚 Documentation

- [Architecture Decision Records](docs/architecture/)
- [API Documentation](docs/api/)
- [Model Cards](docs/models/)
- [Deployment Guide](docs/deployment/)

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run data validation
python scripts/validate_data.py
```

## 📦 Deployment

### Docker Compose (Development)
```bash
docker-compose up -d
```

### Production (AWS/GCP)
```bash
# Build images
docker-compose -f docker-compose.prod.yml build

# Deploy
docker-compose -f docker-compose.prod.yml up -d
```

## 📊 Monitoring

- **Grafana Dashboard**: http://localhost:3000
- **MLflow UI**: http://localhost:5000
- **Airflow UI**: http://localhost:8080

## 🤝 Contributing

This is a portfolio project, but feedback is welcome! Open an issue or PR.

## 📄 License

MIT

## 👤 Author

**Christian G Callahan**
- MS Data Science Candidate
- [GitHub](https://github.com/CCallahan308)
- [Portfolio](https://www.christiangcallahan.tech/)

---

**Built with 💼 for $200k+ senior data science roles**
