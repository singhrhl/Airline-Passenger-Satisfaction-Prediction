# frontend/pages/1_EDA_Insights.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://localhost:8000"

st.set_page_config(page_title="EDA & Insights", layout="wide")
st.title("Executive EDA & Statistical Insights")

# ---------------------------------------------------------------------------
# Fetch data from backend
# ---------------------------------------------------------------------------
try:
    response = requests.get(f"{API_URL}/eda-summary", timeout=5)
    response.raise_for_status()
    data = response.json()
except requests.exceptions.RequestException as e:
    st.error(f"Could not reach the backend API at {API_URL}. Is FastAPI running? ({e})")
    st.stop()

# ---------------------------------------------------------------------------
# Chart 1 — Overall satisfaction (KPI cards)
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)
col1.metric("Total Passengers", f"{data['total_passengers']:,}")
col2.metric("Overall Satisfaction Rate", f"{data['overall_satisfaction_rate']:.1%}")

st.divider()

# ---------------------------------------------------------------------------
# Chart 2 — Satisfaction rate by Class
# ---------------------------------------------------------------------------
st.subheader("Satisfaction Rate by Travel Class")
class_df = pd.DataFrame(data["satisfaction_by_class"])
fig_class = px.bar(
    class_df,
    x="travel_class",
    y="satisfaction_rate",
    text_auto=".1%",
    labels={"travel_class": "Class", "satisfaction_rate": "Satisfaction Rate"},
)
fig_class.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_class, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 3 — Satisfaction rate by Travel Type
# ---------------------------------------------------------------------------
st.subheader("Satisfaction Rate by Travel Type")
travel_df = pd.DataFrame(data["satisfaction_by_travel_type"])
fig_travel = px.bar(
    travel_df,
    x="type_of_travel",
    y="satisfaction_rate",
    text_auto=".1%",
    labels={"type_of_travel": "Type of Travel", "satisfaction_rate": "Satisfaction Rate"},
)
fig_travel.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_travel, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Chart 4 — Rating distribution (feature picker)
# ---------------------------------------------------------------------------
st.subheader("Rating Distribution by Feature")
rating_distributions = data["rating_distributions"]
feature_names = [r["feature"] for r in rating_distributions]

selected_feature = st.selectbox("Choose a feature", options=feature_names)
selected_data = next(r for r in rating_distributions if r["feature"] == selected_feature)

value_counts_df = pd.DataFrame(
    list(selected_data["value_counts"].items()), columns=["rating", "count"]
)
value_counts_df["rating"] = value_counts_df["rating"].astype(int)
value_counts_df = value_counts_df.sort_values("rating")

fig_dist = px.bar(
    value_counts_df,
    x="rating",
    y="count",
    labels={"rating": f"{selected_feature} Rating (0-5)", "count": "Number of Passengers"},
)
st.plotly_chart(fig_dist, use_container_width=True)
st.caption(f"Mean rating: {selected_data['mean_rating']:.2f}")

st.divider()

# ---------------------------------------------------------------------------
# Chart 5 — Feature correlation heatmap
# ---------------------------------------------------------------------------
st.subheader("Feature Correlation Heatmap")
corr_matrix = pd.DataFrame(data["feature_correlations"])
fig_heatmap = px.imshow(
    corr_matrix,
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="auto",
    text_auto=".2f",
)
fig_heatmap.update_layout(height=700)
st.plotly_chart(fig_heatmap, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Chart 6 — Departure delay by satisfaction (mean comparison bar chart)
# ---------------------------------------------------------------------------
st.subheader("Mean Departure Delay: Satisfied vs Dissatisfied")
delay_data = data["mean_departure_delay_by_satisfaction"]
delay_df = pd.DataFrame(
    list(delay_data.items()), columns=["satisfaction", "mean_departure_delay"]
)
fig_delay = px.bar(
    delay_df,
    x="satisfaction",
    y="mean_departure_delay",
    text_auto=".1f",
    labels={"satisfaction": "Passenger Group", "mean_departure_delay": "Mean Departure Delay (min)"},
)
st.plotly_chart(fig_delay, use_container_width=True)
st.caption(
    "Note: this shows mean delay only, not the full distribution. "
    "A true boxplot would require raw per-passenger delay values, "
    "which aren't currently in the precomputed EDA summary."
)