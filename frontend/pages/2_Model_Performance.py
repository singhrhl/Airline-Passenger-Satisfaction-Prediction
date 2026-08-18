# frontend/pages/2_Model_Performance.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Model Performance", layout="wide")
st.title("Model Performance & Explainability")

# ---------------------------------------------------------------------------
# Fetch data from backend
# ---------------------------------------------------------------------------
try:
    response = requests.get(f"{API_URL}/model-metrics", timeout=5)
    response.raise_for_status()
    data = response.json()
except requests.exceptions.RequestException as e:
    st.error(f"Could not reach the backend API at {API_URL}. Is FastAPI running? ({e})")
    st.stop()

models = data["models"]
BEST_MODEL_NAME = "Best_XGB"  # matches the key used in train.py's `results`/`models` dict

# ---------------------------------------------------------------------------
# Section 1 — Model comparison (ROC-AUC across all 3)
# ---------------------------------------------------------------------------
st.subheader("Model Comparison")

comparison_rows = [
    {"model": name, "roc_auc": metrics["roc_auc"]}
    for name, metrics in models.items()
]
comparison_df = pd.DataFrame(comparison_rows).sort_values("roc_auc", ascending=False)

fig_comparison = px.bar(
    comparison_df,
    x="model",
    y="roc_auc",
    text_auto=".3f",
    labels={"model": "Model", "roc_auc": "ROC-AUC"},
)
fig_comparison.update_yaxes(range=[0, 1])
st.plotly_chart(fig_comparison, width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Section 2 — Best hyperparameters
# ---------------------------------------------------------------------------
st.subheader(f"Best Hyperparameters — {BEST_MODEL_NAME}")

best_params = data["best_params"]

# strip the "classifier__" prefix GridSearchCV adds, for cleaner display
cleaned_params = {
    key.replace("classifier__", "").replace("_", " ").title(): value
    for key, value in best_params.items()
}

param_cols = st.columns(len(cleaned_params))
for col, (param_name, param_value) in zip(param_cols, cleaned_params.items()):
    col.metric(label=param_name, value=str(param_value))

with st.expander("View raw hyperparameter grid result"):
    st.json(best_params)

# ---------------------------------------------------------------------------
# Section 3 — Confusion matrix (best model)
# ---------------------------------------------------------------------------
st.subheader(f"Confusion Matrix — {BEST_MODEL_NAME}")

cm = models[BEST_MODEL_NAME]["confusion_matrix"]  # [[TN, FP], [FN, TP]], labels=[False, True]
cm_df = pd.DataFrame(
    cm,
    index=["Actual: Satisfied", "Actual: Dissatisfied"],
    columns=["Predicted: Satisfied", "Predicted: Dissatisfied"],
)

fig_cm = px.imshow(
    cm_df,
    text_auto=True,
    color_continuous_scale="Blues",
    aspect="auto",
)
fig_cm.update_layout(height=450)
st.plotly_chart(fig_cm, width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Section 4 — Precision / Recall / F1 (best model)
# ---------------------------------------------------------------------------
st.subheader(f"Precision, Recall, F1 — {BEST_MODEL_NAME}")

report = models[BEST_MODEL_NAME]["classification_report"]
pr_rows = []
for label in ["satisfied", "dissatisfied"]:
    pr_rows.append({
        "class": label,
        "precision": report[label]["precision"],
        "recall": report[label]["recall"],
        "f1-score": report[label]["f1-score"],
    })
pr_df = pd.DataFrame(pr_rows).melt(id_vars="class", var_name="metric", value_name="score")

fig_pr = px.bar(
    pr_df,
    x="class",
    y="score",
    color="metric",
    barmode="group",
    text_auto=".2f",
    labels={"class": "Class", "score": "Score"},
)
fig_pr.update_yaxes(range=[0, 1])
st.plotly_chart(fig_pr, width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# Section 5 — SHAP global feature importance
# ---------------------------------------------------------------------------
st.subheader("Global Feature Importance (SHAP)")

shap_df = pd.DataFrame(data["shap_global_importance"]).sort_values("importance", ascending=True)

fig_shap = px.bar(
    shap_df,
    x="importance",
    y="feature",
    orientation="h",
    labels={"importance": "Mean |SHAP value|", "feature": "Feature"},
)
fig_shap.update_layout(height=600)
st.plotly_chart(fig_shap, width="stretch")