# Deployment Guide

**SignalForge** can be deployed locally or to cloud platforms. This guide covers both options.

---

## Prerequisites

- Docker & Docker Compose installed
- Python 3.11+ (for local development)
- PostgreSQL 18 (for production database)

---

## Quick Start (Docker)

### 1. Clone and Configure

```bash
git clone https://github.com/CCallahan308/signalforge
cd signalforge

# Copy environment variables
cp .env.example .env
# Edit .env with your settings
```

### 2. Run with Docker Compose

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop services
docker-compose down
```

This will start:
- **PostgreSQL** on port 5432
- **Streamlit Dashboard** on port 8501
- **FastAPI** on port 8000 (when implemented)

### 3. Access Services

- **Dashboard:** http://localhost:8501
- **API:** http://localhost:8000/docs
- **Database:** localhost:5432

---

## Local Development (No Docker)

### 1. Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up Database

```bash
# Create PostgreSQL database
python scripts/setup_database.py --user postgres --password YOUR_PASSWORD
```

### 3. Download Data

```bash
# Requires Kaggle API key
python scripts/download_real_data.py --source telco
```

### 4. Engineer Features

```bash
python scripts/engineer_features.py
```

### 5. Train Models

```bash
python scripts/train_model.py
```

### 6. Run Dashboard

```bash
streamlit run src/app/dashboard.py
```

---

## Cloud Deployment

### Option 1: Render (Easiest)

**Pros:** Free tier, easy setup, auto-deploys from GitHub

**Steps:**
1. Push to GitHub
2. Connect Render to your repo
3. Select "Docker" environment
4. Set environment variables
5. Deploy

**render.yaml** (for automated deployment):
```yaml
services:
  - type: web
    name: signalforge-dashboard
    env: docker
    plan: free
    branch: main
    numReps: 1
    envVars:
      - key: DATABASE_URL
        fromDatabase:
          name: signalforge-db
          property: connectionString

databases:
  - name: signalforge-db
    plan: free
```

### Option 2: Railway

**Pros:** Generous free tier, easy database setup

**Steps:**
1. Connect GitHub to Railway
2. Add PostgreSQL database
3. Deploy from Dockerfile
4. Set environment variables

### Option 3: Fly.io

**Pros:** Global deployment, good performance

**Steps:**
1. Install flyctl
2. `fly launch`
3. Add PostgreSQL: `fly postgres create`
4. Deploy: `fly deploy`

### Option 4: AWS (Production)

**Architecture:**
- **ECS/Fargate** - Container orchestration
- **RDS PostgreSQL** - Managed database
- **ALB** - Load balancer
- **CloudWatch** - Monitoring
- **S3** - Model artifacts

**Cost Estimate:**
- ECS: ~$30-50/month
- RDS: ~$15-30/month
- ALB: ~$20/month
- **Total:** ~$65-100/month for production

---

## Environment Variables

Create `.env` file:

```env
# Database
DATABASE_URL=postgresql://user:password@host:5432/signalforge

# API
API_HOST=0.0.0.0
API_PORT=8000

# Streamlit
STREAMLIT_SERVER_PORT=8501

# Security (for production)
SECRET_KEY=your-secret-key-here
API_KEY_HEADER=X-API-Key
```

---

## Monitoring

### Health Checks

```bash
# Dashboard
curl http://localhost:8501/_stcore/health

# API (when implemented)
curl http://localhost:8000/health
```

### Logs

```bash
# Docker logs
docker-compose logs -f dashboard

# Local logs
# Check logs/ directory
```

### Metrics to Monitor

- **Model Performance:** AUC, precision, recall
- **Business Metrics:** Revenue at risk, interventions, ROI
- **System Metrics:** Response time, error rate, uptime
- **Data Drift:** Feature distribution changes

---

## Scaling Considerations

### Horizontal Scaling

For high-traffic production:
1. Use container orchestration (Kubernetes, ECS)
2. Add load balancer (ALB, nginx)
3. Cache predictions (Redis)
4. Use managed database (RDS)

### Vertical Scaling

For larger datasets:
1. Increase container memory/CPU
2. Optimize feature engineering (batch processing)
3. Use columnar storage (Parquet)
4. Add database indexes

---

## Security Checklist

### Before Deploying

- [ ] Change default passwords
- [ ] Set SECRET_KEY environment variable
- [ ] Enable HTTPS (SSL/TLS)
- [ ] Add authentication to API
- [ ] Restrict database access
- [ ] Remove sensitive data from code
- [ ] Add rate limiting
- [ ] Enable logging
- [ ] Set up error tracking (Sentry)

### Production Security

- Use secrets management (AWS Secrets Manager, HashiCorp Vault)
- Enable VPC for database
- Add WAF (Web Application Firewall)
- Regular security audits
- Keep dependencies updated

---

## CI/CD Pipeline

### GitHub Actions

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build and push Docker image
        run: |
          docker build -t signalforge .
          # Push to registry

      - name: Deploy to production
        run: |
          # Deploy to your platform
```

---

## Troubleshooting

### Common Issues

**Database connection failed:**
```bash
# Check database is running
docker-compose ps db

# Check logs
docker-compose logs db
```

**Dashboard not loading:**
```bash
# Check if dependencies are installed
pip install -r requirements.txt

# Check if data exists
ls data/processed/features.parquet
```

**Model predictions failing:**
```bash
# Check if model is trained
ls models/artifacts/

# Retrain if needed
python scripts/train_model.py
```

---

## Backup & Recovery

### Database Backup

```bash
# Backup
pg_dump -U signalforge_user signalforge > backup.sql

# Restore
psql -U signalforge_user signalforge < backup.sql
```

### Model Artifacts

```bash
# Backup models
tar -czf models-backup.tar.gz models/

# Restore
tar -xzf models-backup.tar.gz
```

---

## Cost Optimization

### Free Tier Options

- **Render:** $0/month (limited resources)
- **Railway:** $5 free credits/month
- **Fly.io:** Free tier available

### Production Optimization

- Use spot instances (50-70% cheaper)
- Right-size containers
- Use reserved instances for database
- Enable auto-scaling
- Optimize model inference time

---

## Next Steps

1. ✅ Local development environment
2. ✅ Docker setup
3. ⏳ Deploy to Render/Railway
4. ⏳ Set up monitoring
5. ⏳ Add authentication
6. ⏳ Set up CI/CD
7. ⏳ Production deployment

---

**Built by Christian G Callahan**  
*This is a learning project - I'm figuring out deployment as I go!*
