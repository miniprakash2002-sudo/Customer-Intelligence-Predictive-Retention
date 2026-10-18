
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_FILE = ROOT / "models" / "best_churn_model.joblib"

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
)

st.title("Customer Intelligence & Predictive Retention Platform")
st.caption("Project 2 | Customer Analytics and Predictive Retention")


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )

    return df


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def get_recommendation(row, risk):
    actions = []

    if risk >= 0.70:
        actions.append("Prioritize customer outreach")
    elif risk >= 0.40:
        actions.append("Send a retention offer")

    if row.get("Contract") == "Month-to-month":
        actions.append("Offer a longer-term contract benefit")

    if row.get("TechSupport") == "No":
        actions.append("Explain available technical support")

    if row.get("tenure", 999) <= 12:
        actions.append("Offer an early-tenure loyalty benefit")

    if row.get("MonthlyCharges", 0) >= 80:
        actions.append("Review plan value and suitable alternatives")

    if not actions:
        actions.append("Continue regular engagement")

    return "; ".join(dict.fromkeys(actions))


if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}")
    st.stop()

if not MODEL_FILE.exists():
    st.error("Trained model not found. Run models/train_model.py first.")
    st.stop()

df = load_data(str(DATA_FILE))
model = load_model(str(MODEL_FILE))

if "Churn" not in df.columns:
    st.error("The dataset must contain a Churn column.")
    st.stop()

# Predict using the same feature columns used during training.
X = df.drop(columns=["Churn", "customerID"], errors="ignore")
risk_scores = model.predict_proba(X)[:, 1]

results = df.copy()
results["Predicted_Churn_Risk"] = risk_scores

results["Risk_Category"] = pd.cut(
    results["Predicted_Churn_Risk"],
    bins=[-0.001, 0.40, 0.70, 1.001],
    labels=["Low", "Medium", "High"],
)

results["Retention_Recommendation"] = results.apply(
    lambda row: get_recommendation(
        row, row["Predicted_Churn_Risk"]
    ),
    axis=1,
)

# Sidebar filters
st.sidebar.header("Dashboard Filters")

risk_options = st.sidebar.multiselect(
    "Risk Category",
    ["High", "Medium", "Low"],
    default=["High", "Medium", "Low"],
)

filtered = results[
    results["Risk_Category"].astype(str).isin(risk_options)
].copy()

search_id = st.sidebar.text_input("Search Customer ID")

if search_id and "customerID" in filtered.columns:
    filtered = filtered[
        filtered["customerID"].astype(str).str.contains(
            search_id, case=False, na=False
        )
    ]

# KPIs
total = len(results)
high_risk = int((results["Risk_Category"] == "High").sum())
avg_risk = results["Predicted_Churn_Risk"].mean() * 100
actual_churn = (
    results["Churn"].astype(str).str.strip().eq("Yes").mean() * 100
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", f"{total:,}")
c2.metric("High-Risk Customers", f"{high_risk:,}")
c3.metric("Average Predicted Risk", f"{avg_risk:.1f}%")
c4.metric("Historical Churn Rate", f"{actual_churn:.1f}%")

st.info(
    "Risk scores are model predictions based on historical data, "
    "not a guarantee that a customer will leave."
)

# Risk distribution
st.subheader("1. Predicted Customer Risk Distribution")

risk_counts = (
    results["Risk_Category"].astype(str)
    .value_counts()
    .reindex(["High", "Medium", "Low"], fill_value=0)
    .rename_axis("Risk Category")
    .reset_index(name="Customers")
)

fig = px.bar(
    risk_counts,
    x="Risk Category",
    y="Customers",
    color="Risk Category",
    title="Customers by Predicted Risk Category",
)
st.plotly_chart(fig, use_container_width=True)

# Contract analysis
if "Contract" in results.columns:
    st.subheader("2. Predicted Risk by Contract")

    contract_summary = (
        results.groupby("Contract", observed=False)
        ["Predicted_Churn_Risk"]
        .mean()
        .mul(100)
        .reset_index(name="Average Predicted Risk (%)")
    )

    fig2 = px.bar(
        contract_summary,
        x="Contract",
        y="Average Predicted Risk (%)",
        title="Average Predicted Risk by Contract",
    )
    st.plotly_chart(fig2, use_container_width=True)

# Customer priority list
st.subheader("3. Customer Retention Priority List")

priority = filtered.sort_values(
    "Predicted_Churn_Risk", ascending=False
).copy()

display_columns = [
    col for col in [
        "customerID",
        "tenure",
        "Contract",
        "MonthlyCharges",
        "Predicted_Churn_Risk",
        "Risk_Category",
        "Retention_Recommendation",
    ]
    if col in priority.columns
]

priority["Predicted_Churn_Risk"] = (
    priority["Predicted_Churn_Risk"] * 100
).round(1)

st.dataframe(
    priority[display_columns].head(100),
    use_container_width=True,
)

st.download_button(
    "Download Customer Risk Report",
    data=priority.to_csv(index=False).encode("utf-8"),
    file_name="customer_retention_priority.csv",
    mime="text/csv",
)

st.caption(
    "Academic project demonstration using the Telco Customer Churn "
    "historical dataset. Recommendations are rule-based examples."
)

# ---- Customer Segmentation Section ----

st.subheader("4. Customer Segmentation")

SEGMENT_FILE = ROOT / "reports" / "segmented_customers.csv"
PROFILE_FILE = ROOT / "reports" / "segment_profiles.csv"

if SEGMENT_FILE.exists() and PROFILE_FILE.exists():
    segments = pd.read_csv(SEGMENT_FILE)
    profiles = pd.read_csv(PROFILE_FILE)

    col1, col2 = st.columns(2)

    col1.metric(
        "Number of Segments",
        segments["Customer_Segment"].nunique()
    )

    col2.metric(
        "Customers Segmented",
        f"{len(segments):,}"
    )

    st.markdown("### Segment Profiles")

    st.dataframe(profiles, use_container_width=True)

    st.markdown("### Customer Distribution by Segment")

    segment_counts = (
        segments["Customer_Segment"]
        .value_counts()
        .sort_index()
        .rename_axis("Segment")
        .reset_index(name="Customers")
    )

    fig_segment = px.bar(
        segment_counts,
        x="Segment",
        y="Customers",
        color="Segment",
        title="Customer Count by Segment",
    )

    st.plotly_chart(
        fig_segment,
        use_container_width=True
    )

    st.markdown("### Explore a Segment")

    selected_segment = st.selectbox(
        "Select Customer Segment",
        sorted(segments["Customer_Segment"].unique())
    )

    segment_customers = segments[
        segments["Customer_Segment"] == selected_segment
    ]

    st.dataframe(
        segment_customers.head(100),
        use_container_width=True
    )

    st.download_button(
        "Download Segment Customers",
        data=segment_customers.to_csv(index=False).encode("utf-8"),
        file_name=f"customer_segment_{selected_segment}.csv",
        mime="text/csv",
    )

else:
    st.warning(
        "Segmentation files not found. "
        "Run python models\\segment_customers.py first."
    )

# ---- Machine Learning Model Evaluation Section ----

MODEL_COMPARISON_FILE = ROOT / "reports" / "model_comparison.csv"

if MODEL_COMPARISON_FILE.exists():
    model_results = pd.read_csv(MODEL_COMPARISON_FILE)

    st.subheader("5. Machine Learning Model Evaluation")
    st.write(
        "Comparison of classification models using "
        "Accuracy, Precision, Recall, F1-score, ROC-AUC and PR-AUC."
    )

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
        "PR_AUC_AP",
    ]

    # Display metrics as percentages
    display_results = model_results.copy()
    display_results[metric_columns] = (
        display_results[metric_columns] * 100
    ).round(2)

    display_results = display_results.rename(
        columns={
            "ROC_AUC": "ROC-AUC",
            "PR_AUC_AP": "PR-AUC (AP)",
            "F1": "F1-Score",
        }
    )

    st.markdown("### Model Comparison Table")
    st.dataframe(
        display_results.sort_values(
            "F1-Score", ascending=False
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Interactive metric comparison chart
    st.markdown("### Performance Comparison Chart")

    chart_data = model_results.melt(
        id_vars="Model",
        value_vars=metric_columns,
        var_name="Metric",
        value_name="Score",
    )

    chart_data["Score"] = chart_data["Score"] * 100

    chart_data["Metric"] = chart_data["Metric"].replace(
        {
            "ROC_AUC": "ROC-AUC",
            "PR_AUC_AP": "PR-AUC (AP)",
            "F1": "F1-Score",
        }
    )

    fig_models = px.bar(
        chart_data,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Machine Learning Model Performance (%)",
        labels={
            "Model": "Machine Learning Model",
            "Score": "Score (%)",
            "Metric": "Evaluation Metric",
        },
        hover_data={"Score": ":.2f"},
    )

    fig_models.update_layout(
        yaxis_range=[0, 100],
        xaxis_tickangle=-15,
        legend_title_text="Evaluation Metric",
    )

    st.plotly_chart(fig_models, use_container_width=True)

    # Explore an individual model
    st.markdown("### Explore Individual Model")

    selected_model = st.selectbox(
        "Select a model",
        model_results["Model"].tolist(),
    )

    selected_row = model_results[
        model_results["Model"] == selected_model
    ].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{selected_row['Accuracy']:.2%}")
    c2.metric("Precision", f"{selected_row['Precision']:.2%}")
    c3.metric("Recall", f"{selected_row['Recall']:.2%}")

    c4, c5, c6 = st.columns(3)
    c4.metric("F1-Score", f"{selected_row['F1']:.2%}")
    c5.metric("ROC-AUC", f"{selected_row['ROC_AUC']:.2%}")
    c6.metric("PR-AUC (AP)", f"{selected_row['PR_AUC_AP']:.2%}")

    st.markdown("### Business Interpretation")

    st.info(
        "Random Forest has the highest F1-score in this comparison. "
        "Gradient Boosting has the highest accuracy and ROC-AUC, "
        "while Decision Tree has the highest recall. "
        "The most suitable model depends on the business objective "
        "and the cost of missing a customer who may churn."
    )

    st.caption(
        "These metrics come from the existing model_comparison.csv. "
        "They reflect the evaluation split used during training."
    )

else:
    st.warning(
        "Model comparison file not found. "
        "Please check reports/model_comparison.csv."
    )
