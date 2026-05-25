#!/usr/bin/env python3
"""
SignalForge Dashboard Demo - Streamlit Cloud Compatible

A simplified version for live demo deployment.
Uses synthetic data to showcase functionality without database requirements.

Run locally: streamlit run src/app/dashboard_demo.py
Deployed at: https://signalforge-ccallahan308.streamlit.app/
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page config
st.set_page_config(
    page_title="SignalForge - Churn Intelligence Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 SignalForge - Churn Intelligence")
st.markdown("**Live Demo** | [GitHub Repo](https://github.com/CCallahan308/signalforge) | [Project Page](https://christiangcallahan.tech/projects/signalforge)")
st.caption(
    "Static showcase with representative figures from the IBM Telco run. "
    "Reproduce exact numbers with `python scripts/train_with_optuna.py` "
    "(written to models/artifacts/training_results.json)."
)
st.markdown("---")

# Model Performance Section
st.header("🎯 Model Performance (5-Fold CV)")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Logistic Regression AUC",
        value="0.849 ± 0.012",
        delta="95% CI: [0.828, 0.869]"
    )

with col2:
    st.metric(
        label="vs Random Forest",
        value="p = 0.016",
        delta="CIs overlap — borderline",
        delta_color="off"
    )

with col3:
    st.metric(
        label="vs Gradient Boosting",
        value="p = 0.130",
        delta="Not significant",
        delta_color="off"
    )

st.info(
    "💡 The three models are within ~0.005 AUC of each other and their 95% CIs overlap, "
    "so the differences are borderline. Logistic Regression is used in production for its "
    "discrimination **and** interpretability; Gradient Boosting is better calibrated "
    "(Brier 0.139 vs 0.164) and would be preferred when probability accuracy matters most."
)

# Model Comparison Chart
st.subheader("Model Comparison with Confidence Intervals")

models_data = {
    'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
    'AUC': [0.849, 0.844, 0.846],
    'CI_Lower': [0.828, 0.825, 0.827],
    'CI_Upper': [0.869, 0.863, 0.866]
}

df_models = pd.DataFrame(models_data)

fig = go.Figure()

for idx, row in df_models.iterrows():
    color = '#1f77b4' if row['Model'] == 'Logistic Regression' else 'lightgray'
    fig.add_trace(go.Bar(
        name=row['Model'],
        x=[row['Model']],
        y=[row['AUC']],
        error_y=dict(type='data', symmetric=False, array=[row['CI_Upper']-row['AUC']], arrayminus=[row['AUC']-row['CI_Lower']]),
        marker_color=color
    ))

fig.update_layout(
    title='Model AUC with 95% Confidence Intervals',
    yaxis_title='AUC Score',
    yaxis_range=[0.8, 0.9],
    showlegend=False,
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# Business Impact Section
st.header("💰 Business Impact")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Annual Revenue at Risk",
        value="$1.67M",
        delta="$139K/month"
    )

with col2:
    st.metric(
        label="Expected ROI",
        value="1.21x - 1.81x",
        delta="$270K - $405K annual savings"
    )

# ROI Calculator
st.subheader("📈 ROI Calculator")

col1, col2 = st.columns(2)

with col1:
    intervention_budget = st.slider(
        "Monthly Intervention Budget ($K)",
        min_value=10,
        max_value=50,
        value=18,
        step=1
    )

with col2:
    save_rate = st.slider(
        "Expected Save Rate (%)",
        min_value=10,
        max_value=40,
        value=25,
        step=1
    )

# Calculate ROI (annualized, internally consistent)
customers_at_risk = 1409
annual_value_per_customer = 965  # annual revenue retained per saved customer
customers_saved = int(customers_at_risk * (save_rate / 100))
annual_revenue_saved = customers_saved * annual_value_per_customer
annual_cost = intervention_budget * 1000 * 12  # budget slider is $K/month
net_profit = annual_revenue_saved - annual_cost
# Return on spend = revenue per $1 invested; net ROI subtracts cost.
return_on_spend = annual_revenue_saved / annual_cost if annual_cost else 0.0
net_roi = net_profit / annual_cost if annual_cost else 0.0

st.markdown(f"""
**Results (annualized):**
- Customers targeted: **{customers_at_risk:,}**
- Customers saved: **{customers_saved:,}**
- Annual revenue saved: **${annual_revenue_saved:,.0f}**
- Annual intervention cost: **${annual_cost:,.0f}**
- Net profit: **${net_profit:,.0f}**
- Return on spend: **{return_on_spend:.2f}x** (net ROI {net_roi:+.0%})
""")

# Feature Importance
st.header("🔍 Top Churn Drivers")

features_data = {
    'Feature': ['Contract Risk', 'Payment Risk', 'Tenure Risk', 'Demographic Risk', 'Service Risk'],
    'Importance': [0.112, 0.052, 0.049, 0.042, 0.019],
    'Insight': [
        'Month-to-month = 3.8x more churn',
        'Electronic check = 3x more churn',
        'New customers (0-12 mo) = 5x more churn',
        'Seniors/singles = 1.5-2x more churn',
        'No security services = 2-3x more churn'
    ]
}

df_features = pd.DataFrame(features_data)

fig_features = px.bar(
    df_features,
    x='Importance',
    y='Feature',
    orientation='h',
    title='Feature effects (standardized logistic-regression coefficients, |log-odds|)',
    color='Importance',
    color_continuous_scale='Blues'
)

fig_features.update_layout(height=400, showlegend=False)

st.plotly_chart(fig_features, use_container_width=True)

# Feature insights
st.markdown("**Key Insights:**")
for idx, row in df_features.iterrows():
    st.markdown(f"- **{row['Feature']}** (|coef|: {row['Importance']:.3f}): {row['Insight']}")

# Statistical Rigor Section
st.header("🧪 Statistical Rigor")

st.markdown("""
**What Makes This Different:**

✅ **5-fold stratified cross-validation** on the training split — not a single train/test split
✅ **Bootstrap 95% confidence intervals** (1000 samples) — quantifies uncertainty
✅ **Statistical significance testing** (paired t-tests) — quantifies whether model differences are real or noise
✅ **Model coefficients** (standardized logistic regression) — explains the model itself, not a surrogate
✅ **Calibration** (Brier score) — checks probability accuracy, not just ranking

These are the methods this project applies; see the README for honest limitations.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>SignalForge</strong> - churn prediction with statistical model comparison</p>
    <p>Built by <a href='https://christiangcallahan.tech'>Christian Callahan</a>
    | <a href='https://github.com/CCallahan308/signalforge'>GitHub</a></p>
</div>
""", unsafe_allow_html=True)
