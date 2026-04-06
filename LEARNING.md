# Learning Notes - Model Training

## What Happened

**Date:** April 6, 2026
**Session:** 3 (Model Training)

---

## The Results (Real Talk)

### What I Expected
I thought gradient boosting or random forest would crush logistic regression. That's what all the blog posts say, right? "Use XGBoost for tabular data."

### What Actually Happened
**Logistic regression won with 0.848 AUC.**

This surprised me at first. But then I remembered something from my ML class: **good features > fancy algorithms**.

---

## Why This Makes Sense

### 1. Feature Engineering Paid Off
I spent 3 hours on feature engineering. It shows:
- **churn_risk_score** - Top predictor in all models
- **contract_risk_score** - Business-informed feature
- **value_at_risk** - Revenue-focused feature

These engineered features captured the signal so well that even a simple model could find it.

### 2. Linear Relationships Dominate
In this dataset, churn is mostly linear:
- Month-to-month → more churn
- New customer → more churn
- Electronic check → more churn

Logistic regression handles linear relationships perfectly. No need for complex non-linear models.

### 3. Class Imbalance Helped
Using `class_weight='balanced'` in logistic regression gave it a boost. It adjusted for the 26.5% churn rate naturally.

---

## What I Learned

### Lesson 1: Start Simple
**My Mistake:** I almost skipped logistic regression because it's "too simple."

**The Truth:** Baseline models are critical. If your baseline is 0.848 AUC, you're doing well. Complex models should beat that, or they're not worth it.

**Next Time:** Always start with logistic regression. It's fast, interpretable, and often surprisingly good.

### Lesson 2: Feature Engineering > Model Selection
- 3 hours on features: 0.848 AUC
- 5 minutes on logistic regression: **Best model**

If I had spent that 3 hours tuning XGBoost hyperparameters instead of engineering features, I'd probably have a worse model.

### Lesson 3: Business Context Matters
The top features make business sense:
- `churn_risk_score` - Composite of contract + tenure + payment
- `is_month_to_month` - High churn (42.7%)
- `value_at_risk` - Revenue × risk

This tells me my feature engineering was on the right track. It's not just random features - they map to business reality.

---

## The Tradeoffs

### Logistic Regression (Best)
**Pros:**
- Highest AUC (0.848)
- Interpretable coefficients
- Fast inference
- Easy to explain to stakeholders

**Cons:**
- Lower precision (0.505) - lots of false positives
- Might miss complex patterns

### Random Forest (2nd)
**Pros:**
- Better balance (F1: 0.629)
- Captures non-linearities
- Robust to outliers

**Cons:**
- Lower AUC than baseline
- Less interpretable
- Slower inference

### Gradient Boosting (3rd)
**Pros:**
- Highest precision (0.631)
- Good for high-stakes predictions

**Cons:**
- Lowest recall (0.527) - misses churners
- Complex to tune
- Overkill for this problem

---

## What I'd Do Differently

### 1. Trust the Process
I almost didn't train logistic regression because I thought it was "too basic." Now I know: **always start with the baseline.**

### 2. Focus on Features, Not Algorithms
The 3 hours I spent on feature engineering gave me more ROI than hours of hyperparameter tuning ever would.

### 3. Use Business Metrics, Not Just AUC
AUC is nice, but what matters is:
- How much revenue can we save?
- How many customers can we intervene on?
- What's the ROI of retention campaigns?

---

## Real-World Implications

### For Stakeholders
**Good News:** We can predict 81% of churners with a simple, interpretable model.

**Bad News:** We'll also flag some non-churners (precision = 50.5%). This means:
- Intervention costs will be higher
- Some customers might get annoyed

**Solution:** Use tiered interventions:
- High-risk (top 20%): High-touch save campaign
- Medium-risk (20-50%): Email + discount
- Low-risk (bottom 50%): Monitor only

### For the Business
**Revenue Impact:**
- Total at risk: $139K/month
- Model identifies: 81% of churners = $113K/month
- Intervention success: 20% (conservative)
- **Revenue saved: $23K/month ($276K/year)**

**ROI:**
- Intervention cost: $50/customer × 373 interventions = $18.7K
- Revenue saved: $23K
- **ROI: 1.23x** (conservative)

---

## Feature Importance Insights

### Top 3 Features (All Models)
1. **churn_risk_score** - My composite risk metric
2. **contract_risk_score** - Contract-based risk
3. **value_at_risk** - Revenue at risk

**What This Means:** My feature engineering worked. The composite features I created are more predictive than raw features.

### What Didn't Work
- `is_male` - Gender doesn't predict churn (good!)
- `has_phone` - Basic service, not predictive
- `entertainment_count` - Less important than core services

---

## Model Deployment Considerations

### Why Logistic Regression is Great for Production
1. **Fast inference:** <1ms per prediction
2. **Interpretable:** Can explain to business stakeholders
3. **Low memory:** Small model size
4. **Stable:** Less sensitive to data drift

### What to Monitor
1. **Feature drift:** Are input distributions changing?
2. **Prediction drift:** Is churn rate changing?
3. **Business metrics:** Are we saving revenue?

---

## Next Steps

### Immediate (This Week)
1. ✅ Train models
2. ⏳ Analyze feature importance (SHAP)
3. ⏳ Build FastAPI for predictions
4. ⏳ Add business metrics to API

### Short-term (Next 2 Weeks)
1. Create Streamlit dashboard
2. Add intervention recommendations
3. Build ROI calculator
4. Test with sample customers

### Long-term (Next Month)
1. Add uplift modeling (who's saveable?)
2. A/B test interventions
3. Deploy to production
4. Monitor performance over time

---

## Personal Takeaways

### What I'm Proud Of
1. **Started simple:** Didn't over-engineer
2. **Feature engineering:** Spent time where it matters
3. **Business focus:** Thought about ROI, not just AUC
4. **Learning in public:** Documenting my process

### What Surprised Me
1. **Logistic regression won** - Shows simplicity works
2. **Feature engineering ROI** - 3 hours → best model
3. **Business alignment** - Features make business sense

### What I'm Still Figuring Out
1. **How to present this to stakeholders?**
2. **What's the right intervention strategy?**
3. **How do I measure real-world impact?**
4. **When do I retrain the model?**

---

## Metrics I Care About

### Model Metrics
- **AUC:** 0.848 ✓ (Target: 0.80+)
- **Recall:** 0.810 ✓ (Find churners)
- **Precision:** 0.505 (Lower than I'd like)

### Business Metrics
- **Revenue at risk identified:** $113K/month
- **Expected revenue saved:** $23K/month
- **ROI:** 1.23x

### Production Metrics
- **Inference time:** <1ms ✓
- **Model size:** <1MB ✓
- **Interpretability:** High ✓

---

## Questions I'm Asking Myself

1. **Should I try to improve precision?**
   - Pro: Fewer false positives → lower intervention costs
   - Con: Might miss churners → lose revenue
   - Answer: **No.** Better to catch more churners and accept some false positives.

2. **Should I ensemble models?**
   - Pro: Might get 0.85-0.86 AUC
   - Con: Complexity, harder to interpret
   - Answer: **Not yet.** Logistic regression is working fine.

3. **Should I use deep learning?**
   - Pro: Cool tech, might get 0.87 AUC
   - Con: Overkill, not interpretable, overkill for this data size
   - Answer: **Definitely no.** This is tabular data with good features.

---

## Advice for Others

### If You're Building Your First ML Model

1. **Start with logistic regression**
   - It's your baseline
   - It's interpretable
   - It might be your best model (like mine was!)

2. **Spend time on features, not algorithms**
   - 3 hours on features > 10 hours tuning XGBoost
   - Domain knowledge beats algorithm complexity

3. **Think about business metrics**
   - AUC is nice, but ROI matters more
   - Can you explain your model to stakeholders?
   - What's the intervention cost?

4. **Learn in public**
   - Document your process
   - Share your failures and successes
   - You'll learn faster

5. **Keep it simple**
   - Complex ≠ better
   - Production systems need to be maintainable
   - Start simple, add complexity only if needed

---

## Final Thoughts

This session taught me that **simple + good features beats complex + bad features.**

I spent 3 hours on feature engineering and 5 minutes on the model. And it worked.

Next time, I'll trust the process even more. Start simple, validate, then iterate.

The best model isn't the most complex one - it's the one that solves the business problem effectively and maintainably.

---

**Status:** Model training complete ✅
**Next:** API development & business metrics
**Feeling:** Pretty good about this! 😊

*Last updated: April 6, 2026*
