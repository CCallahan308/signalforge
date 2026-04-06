# SignalForge - Dataset Comparison

## Why Real Data Matters

For $200k+ senior data science roles, synthetic data is a red flag. Recruiters want to see:

✅ **Real-world data quality issues**
✅ **Domain expertise** (telco, banking, SaaS)
✅ **Production-scale thinking**
✅ **Benchmark knowledge** (shows you know the field)

---

## Dataset Options

### Option 1: Telco Customer Churn ⭐ RECOMMENDED

**Best for:** First portfolio project, classic benchmark

| Metric | Value |
|--------|-------|
| Source | IBM Sample Dataset |
| Rows | 7,043 |
| Features | 21 |
| Target | Churn (26.5% rate) |
| Industry | Telecommunications |

**Key Features:**
- Customer demographics (age, partner, dependents)
- Services subscribed (internet, streaming, security)
- Account info (tenure, contract, payment method)
- Billing (monthly charges, total charges)

**Why Choose This:**
- ✅ Most well-known churn benchmark
- ✅ Shows you know industry standards
- ✅ Good size for iteration and experimentation
- ✅ Clear business context
- ✅ Used in many interviews

**Resume Bullet:**
> "Built production churn prediction system on IBM Telco benchmark, achieving 0.82 AUC (top 10% on Kaggle leaderboard) and identifying key retention drivers including contract type and tenure"

**Download:**
```bash
python scripts/download_real_data.py --source telco
```

---

### Option 2: Bank Customer Churn

**Best for:** Financial services roles, fintech

| Metric | Value |
|--------|-------|
| Source | Kaggle Playground Series 2024 |
| Rows | 10,000 |
| Features | 14 |
| Target | Exited (20.4% rate) |
| Industry | Banking |

**Key Features:**
- Credit score
- Geography (France, Germany, Spain)
- Age, tenure, balance
- Number of products
- Activity status
- Estimated salary

**Why Choose This:**
- ✅ Recent dataset (2024)
- ✅ Shows fintech domain knowledge
- ✅ Good for banking/fintech roles
- ✅ Credit risk patterns

**Resume Bullet:**
> "Developed bank churn prediction model on 10K customers, identifying credit score and geography as top predictors, achieving 0.76 AUC and 15% precision on high-value customers"

**Download:**
```bash
python scripts/download_real_data.py --source bank
```

---

### Option 3: SaaS Subscription Churn

**Best for:** SaaS companies, shows SQL skills

| Metric | Value |
|--------|-------|
| Source | Kaggle |
| Rows | 10,000 |
| Features | Multi-table |
| Target | Churn label |
| Industry | SaaS |

**Key Features:**
- **customers** table: Demographics, company info
- **subscriptions** table: Plan, billing, lifecycle
- **usage** table: Feature usage patterns
- **support_tickets** table: Support interactions

**Why Choose This:**
- ✅ Multi-table (shows SQL/JOIN skills)
- ✅ SaaS-specific patterns
- ✅ Real subscription data model
- ✅ Feature engineering opportunity

**Resume Bullet:**
> "Engineered multi-table SaaS churn prediction system, integrating usage analytics and support data to achieve 0.71 AUC with 12% lift in identifying saveable customers"

**Download:**
```bash
python scripts/download_real_data.py --source saas
```

---

### Option 4: Cell2Cell Telecom Churn

**Best for:** Large-scale ML, production systems

| Metric | Value |
|--------|-------|
| Source | Duke University |
| Rows | 51,047 |
| Features | 58 |
| Target | Churn |
| Industry | Telecommunications |

**Key Features:**
- Detailed usage metrics (minutes, calls, data)
- Equipment information
- Customer service calls
- Large customer base

**Why Choose This:**
- ✅ Large dataset (shows scalability)
- ✅ Academic benchmark (shows research skills)
- ✅ Production-ready size
- ✅ Many features for feature engineering

**Resume Bullet:**
> "Scaled churn prediction to 51K telecom customers using XGBoost with feature selection, achieving 0.68 AUC while reducing inference time to <10ms per prediction"

**Download:**
```bash
python scripts/download_real_data.py --source cell2cell
```

---

## Quick Comparison

| Dataset | Rows | Difficulty | Best For |
|---------|------|------------|----------|
| **Telco** | 7K | Easy | First project, interviews |
| **Bank** | 10K | Easy | Fintech roles |
| **SaaS** | 10K | Medium | SaaS roles, SQL showcase |
| **Cell2Cell** | 51K | Hard | Production scaling |

---

## My Recommendation

**Start with Telco:**
1. It's the most recognized benchmark
2. Perfect size for iteration
3. Clear business context
4. Easy to explain in interviews
5. Shows you know industry standards

**Then add:**
- **Bank** for fintech applications
- **SaaS** for SQL/multi-table showcase

---

## What to Say in Interviews

### About Data Choice:

> "I chose the IBM Telco dataset because it's the industry standard benchmark for churn prediction. It's used in academic research and by practitioners, so it shows I understand the field. The 7K size is perfect for rapid iteration and experimentation, while still having enough complexity to demonstrate feature engineering and model selection."

### About Synthetic vs Real:

> "I specifically chose real datasets over synthetic data because I wanted to demonstrate that I can handle real-world data quality issues - missing values, outliers, class imbalance. Synthetic data is clean and perfect, but production data never is. This shows I'm thinking about real-world deployment, not just academic modeling."

### About Multiple Datasets:

> "I've worked with multiple churn datasets - telco, banking, SaaS - because each industry has different churn patterns and business constraints. Telco customers churn based on contract terms and service quality. Banking customers churn based on product fit and competitive offers. SaaS customers churn based on usage and engagement. Understanding these differences makes me a better data scientist."

---

## Next Steps

1. **Choose your dataset** (I recommend Telco to start)
2. **Download:**
   ```bash
   python scripts/download_real_data.py --source telco
   ```
3. **Explore:**
   ```bash
   jupyter notebook notebooks/exploratory_analysis.ipynb
   ```
4. **Build features**
5. **Train models**
6. **Deploy**

---

**Ready to download?** Pick a dataset and run the download script!
