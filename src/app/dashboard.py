#!/usr/bin/env python3
"""
SignalForge Dashboard - Streamlit App

Interactive churn prediction dashboard for business stakeholders.

USAGE:
    streamlit run src/app/dashboard.py

Built by Christian G Callahan - April 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Page config
st.set_page_config(
    page_title="SignalForge - Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .risk-high {
        color: #d32f2f;
        font-weight: bold;
    }
    .risk-medium {
        color: #f57c00;
        font-weight: bold;
    }
    .risk-low {
        color: #388e3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load engineered features and model results."""
    try:
        features = pd.read_parquet('data/processed/features.parquet')
        model_comparison = pd.read_csv('models/artifacts/model_comparison.csv')

        # Add derived columns that the dashboard expects
        if 'account_id' not in features.columns:
            features['account_id'] = [f'ACCT-{i:05d}' for i in range(len(features))]
        if 'mrr' not in features.columns:
            features['mrr'] = features['MonthlyCharges']
        if 'account_age_months' not in features.columns:
            features['account_age_months'] = features['tenure']
        if 'Contract' not in features.columns:
            def infer_contract(row):
                if row.get('is_two_year', 0) == 1: return 'Two year'
                elif row.get('is_one_year', 0) == 1: return 'One year'
                else: return 'Month-to-month'
            features['Contract'] = features.apply(infer_contract, axis=1)
        if 'tenure_bucket' not in features.columns:
            features['tenure_bucket'] = pd.cut(features['tenure'], bins=[0, 12, 24, 48, 72], labels=['0-12', '13-24', '25-48', '49-72+'], right=True)

        return features, model_comparison
    except FileNotFoundError:
        st.error("Data not found! Run: python scripts/engineer_features.py && python scripts/train_model.py")
        st.stop()


def main():
    """Main dashboard app."""

    # Header
    st.title("📊 SignalForge - Churn Intelligence Dashboard")
    st.markdown("*Built by Christian G Callahan | [GitHub](https://github.com/CCallahan308/signalforge) | April 2026*")
    st.markdown("---")

    # Load data
    with st.spinner("Loading data..."):
        features, model_comparison = load_data()

    # Sidebar
    st.sidebar.header("🔧 Filters & Settings")
    show_all = st.sidebar.checkbox("Show all customers", value=False)
    risk_threshold = st.sidebar.slider("Risk Score Threshold", 0.0, 1.0, 0.5)

    # Main metrics
    st.header("📈 Business Impact Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_customers = len(features)
        st.metric("Total Customers", f"{total_customers:,}")

    with col2:
        churn_rate = features['churned'].mean()
        st.metric("Churn Rate", f"{churn_rate:.1%}")

    with col3:
        total_revenue = features['mrr'].sum()
        st.metric("Total Monthly Revenue", f"${total_revenue:,.0f}")

    with col4:
        revenue_at_risk = features[features['churned'] == 1]['mrr'].sum()
        st.metric("Revenue at Risk", f"${revenue_at_risk:,.0f}", delta=f"-{revenue_at_risk/total_revenue:.1%}")

    st.markdown("---")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Customer Risk Leaderboard",
        "💰 ROI Calculator",
        "📊 Model Performance",
        "🔍 Feature Insights"
    ])

    # Tab 1: Customer Risk Leaderboard
    with tab1:
        st.header("High-Risk Customers")

        st.markdown("""
        **Top customers at risk of churning.** These are the customers who should be prioritized for intervention campaigns.

        *Risk Score = Probability of churn (0-1 scale)*
        """)

        # Get high-risk customers
        if 'churn_risk_score' in features.columns:
            high_risk = features[features['churn_risk_score'] >= features['churn_risk_score'].quantile(0.75)].copy()
        else:
            # Fallback to churned flag for demo
            high_risk = features[features['churned'] == 1].copy()

        high_risk['risk_label'] = high_risk['churned'].map({1: 'Churned', 0: 'Active'}).fillna('Unknown')

        # Display top 20 at-risk customers
        display_cols = ['account_id', 'mrr', 'churn_risk_score', 'contract_risk_score',
                        'tenure_risk', 'is_month_to_month', 'is_new_customer']

        available_cols = [col for col in display_cols if col in high_risk.columns]

        if len(available_cols) > 0:
            st.dataframe(
                high_risk[available_cols].head(20).style.format({
                    'mrr': '${:.2f}',
                    'churn_risk_score': '{:.2f}',
                    'contract_risk_score': '{:.2f}',
                    'tenure_risk': '{:.2f}'
                }),
                use_container_width=True
            )
        else:
            st.warning("Risk scores not available. Run feature engineering first.")

        # Risk distribution chart
        if 'churn_risk_score' in features.columns:
            fig = px.histogram(
                features,
                x='churn_risk_score',
                color='churned',
                nbins=30,
                title='Churn Risk Score Distribution',
                labels={'churn_risk_score': 'Risk Score', 'churned': 'Churned'},
                color_discrete_map={1: '#d32f2f', 0: '#388e3c'}
            )
            st.plotly_chart(fig, use_container_width=True)

    # Tab 2: ROI Calculator
    with tab2:
        st.header("💰 ROI Calculator")
        st.markdown("Calculate the return on investment for churn intervention campaigns.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Intervention Parameters")

            intervention_cost = st.number_input(
                "Cost per intervention ($)",
                min_value=0,
                max_value=500,
                value=50,
                step=10
            )

            success_rate = st.slider(
                "Expected success rate (%)",
                min_value=0,
                max_value=100,
                value=20,
                step=5
            ) / 100

            target_customers = st.number_input(
                "Number of customers to target",
                min_value=0,
                max_value=len(features),
                value=min(100, int(features['churned'].sum())),
                step=10
            )

            avg_mrr = features['mrr'].mean()
            retention_months = st.slider(
                "Expected retention duration (months)",
                min_value=1,
                max_value=24,
                value=12
            )

        with col2:
            st.subheader("ROI Results")

            # Calculate ROI
            total_cost = intervention_cost * target_customers
            customers_saved = target_customers * success_rate
            revenue_per_customer = avg_mrr * retention_months
            total_revenue_saved = customers_saved * revenue_per_customer
            net_profit = total_revenue_saved - total_cost
            roi = (total_revenue_saved / total_cost - 1) if total_cost > 0 else 0

            # Display metrics
            st.metric("Total Investment", f"${total_cost:,.0f}")
            st.metric("Customers Saved", f"{customers_saved:.0f}")
            st.metric("Revenue Saved", f"${total_revenue_saved:,.0f}")
            st.metric("Net Profit", f"${net_profit:,.0f}", delta=f"{roi:.1%} ROI")

            # ROI visualization
            fig = go.Figure(data=[
                go.Bar(
                    name='Cost',
                    x=['Intervention Cost'],
                    y=[total_cost],
                    marker_color='#d32f2f'
                ),
                go.Bar(
                    name='Revenue Saved',
                    x=['Revenue Saved'],
                    y=[total_revenue_saved],
                    marker_color='#388e3c'
                )
            ])
            fig.update_layout(
                title='Intervention Cost vs Revenue Saved',
                yaxis_title='Amount ($)',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Model Performance
    with tab3:
        st.header("📊 Model Performance")
        st.markdown("Comparing different ML models for churn prediction.")

        # Fill missing precision/recall for models that only have AUC/F1
        for metric in ['AUC', 'Precision', 'Recall', 'F1']:
            if metric in model_comparison.columns:
                model_comparison[metric] = model_comparison[metric].fillna(0)

        # Model comparison table
        st.dataframe(
            model_comparison.style.format({
                'AUC': '{:.3f}',
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1': '{:.3f}'
            }),
            use_container_width=True
        )

        # Model performance visualization
        fig = go.Figure()

        models = model_comparison['Model'].tolist()
        metrics = ['AUC', 'Precision', 'Recall', 'F1']

        for metric in metrics:
            if metric in model_comparison.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=models,
                    y=model_comparison[metric],
                ))

        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig, use_container_width=True)

        # Key insights
        st.subheader("🎯 Key Insights")
        best_model = model_comparison.iloc[0]

        st.markdown(f"""
        **Best Model:** {best_model['Model']}

        - **AUC:** {best_model['AUC']:.3f} (Target: 0.80+)
        - **Precision:** {best_model['Precision']:.3f}
        - **Recall:** {best_model['Recall']:.3f} (Catches {best_model['Recall']:.1%} of churners)

        **What this means:**
        - We can identify **{best_model['Recall']:.1%}** of customers who will churn
        - For every 100 predictions, {best_model['Precision']*100:.0f} will actually churn
        - Business can target interventions to high-risk customers

        **Revenue Impact:**
        - Total at risk: ${revenue_at_risk:,.0f}/month
        - Model identifies: ${revenue_at_risk * best_model['Recall']:,.0f}/month
        - **Annual impact: ${revenue_at_risk * best_model['Recall'] * 12:,.0f}**
        """)

    # Tab 4: Feature Insights
    with tab4:
        st.header("🔍 Feature Insights")
        st.markdown("Understanding what drives customer churn.")

        col1, col2 = st.columns(2)

        with col1:
            # Churn by contract type
            if 'Contract' in features.columns:
                contract_churn = features.groupby('Contract')['churned'].mean().reset_index()
                fig = px.bar(
                    contract_churn,
                    x='Contract',
                    y='churned',
                    title='Churn Rate by Contract Type',
                    labels={'churned': 'Churn Rate'},
                    color='churned',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Churn by tenure
            if 'tenure_bucket' in features.columns:
                tenure_churn = features.groupby('tenure_bucket')['churned'].mean().reset_index()
                fig = px.bar(
                    tenure_churn,
                    x='tenure_bucket',
                    y='churned',
                    title='Churn Rate by Tenure',
                    labels={'churned': 'Churn Rate', 'tenure_bucket': 'Tenure'},
                    color='churned',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig, use_container_width=True)

        # Numerical feature distributions
        st.subheader("Numerical Features")

        numeric_features = ['mrr', 'account_age_months', 'service_count']
        selected_feature = st.selectbox(
            "Select feature to analyze",
            [f for f in numeric_features if f in features.columns]
        )

        if selected_feature:
            fig = px.histogram(
                features,
                x=selected_feature,
                color='churned',
                title=f'{selected_feature} Distribution by Churn',
                labels={'churned': 'Churned'},
                color_discrete_map={1: '#d32f2f', 0: '#388e3c'},
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>SignalForge</strong> - Built with 💼 by Christian G Callahan</p>
        <p><a href='https://github.com/CCallahan308/signalforge'>GitHub</a> | <a href='https://www.christiangcallahan.tech/'>Portfolio</a></p>
        <p><em>This is a learning project - </em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
