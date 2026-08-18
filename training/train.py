"""
Training script — airline passenger satisfaction.
SHAP section still has TODOs — fill in and ask when stuck.
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df_train = pd.read_csv("./training/train.csv")
df_train = df_train.drop(columns=["Unnamed: 0", "id"])

df_test = pd.read_csv("./training/test.csv")
df_train = df_test.drop(columns=["Unnamed: 0", "id"])

X_train, y_train = df_train.drop(columns=["satisfaction"]), df_train["satisfaction"].apply(lambda x: False if x == "satisfied" else True)
X_test, y_test = df_test.drop(columns=["satisfaction"]), df_test["satisfaction"].apply(lambda x: False if x == "satisfied" else True)


# ---------------------------------------------------------------------------
# 2. Column groups
# ---------------------------------------------------------------------------
CATEGORICAL_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUMERIC_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
RATING_COLS = [col for col in X_train.columns if col not in CATEGORICAL_COLS + NUMERIC_COLS]

# ---------------------------------------------------------------------------
# 3. Train/test split (stratified — preserves the 57/43 class ratio in both sets)
# ---------------------------------------------------------------------------
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=0) 

# ---------------------------------------------------------------------------
# 4. Shared preprocessing
# ---------------------------------------------------------------------------
base_preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ("num_impute", SimpleImputer(strategy="median"), ["Arrival Delay in Minutes"]),
], remainder="passthrough")

# ---------------------------------------------------------------------------
# 5. Three pipelines
# ---------------------------------------------------------------------------
logreg_pipeline = Pipeline(steps=[
    ("base_preprocessor", base_preprocessor),
    ("std_scaler", StandardScaler()),
    ("classifier", LogisticRegression()),
])

xgb_default_pipeline = Pipeline(steps=[
    ("preprocessor", base_preprocessor),
    ("classifier", XGBClassifier(eval_metric="logloss", random_state=42)),
])

grid_params = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 5, 10],
    "classifier__learning_rate": [0.01, 0.02, 0.05, 0.1],
}

best_pipeline = GridSearchCV(
    estimator=xgb_default_pipeline,
    param_grid=grid_params,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    refit="roc_auc",
    cv=5,
    n_jobs=-1,
)

# ---------------------------------------------------------------------------
# 6. Fit + evaluate
# ---------------------------------------------------------------------------
def evaluate_model(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # column for class True (dissatisfied)

    report_str = classification_report(y_test, y_pred, target_names=["satisfied", "dissatisfied"])
    report_dict = classification_report(
        y_test, y_pred, target_names=["satisfied", "dissatisfied"], output_dict=True
    )

    print(f"--- {name} ---")
    print(report_str)
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.4f}")

    cm = confusion_matrix(y_test, y_pred, labels=[False, True])

    return {
        "name": name,
        "y_proba": y_proba,
        "auc": auc,
        "report": report_dict,
        "confusion_matrix": cm.tolist(),
    }


models = {
    "Logistic_Regression_baseline": logreg_pipeline.fit(X_train, y_train),
    "XGBoost_baseline": xgb_default_pipeline.fit(X_train, y_train),
    "Best_XGB": best_pipeline.fit(X_train, y_train),
}

results = []
for name, model in models.items():
    results.append(evaluate_model(name, model, X_test, y_test))

print("Best params:", best_pipeline.best_params_)

# ROC comparison plot
plt.figure()
for r in results:
    fpr, tpr, _ = roc_curve(y_test, r["y_proba"], pos_label=True)
    plt.plot(fpr, tpr, label=f"{r['name']} (AUC={r['auc']:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Comparison")
plt.savefig("./models/roc_comparison.png")

# ---------------------------------------------------------------------------
# 7. Metrics artifact (for /model-metrics — assembled fully once SHAP is added below)
# ---------------------------------------------------------------------------
metrics_artifact = {
    "best_params": best_pipeline.best_params_,
    "models": {
        r["name"]: {
            "roc_auc": r["auc"],
            "classification_report": r["report"],
            "confusion_matrix": r["confusion_matrix"],
        }
        for r in results
    },
}

# ---------------------------------------------------------------------------
# 8. SHAP — global feature importance for the final (Best_XGB) pipeline
# ---------------------------------------------------------------------------
final_pipeline = best_pipeline.best_estimator_
preprocessor = final_pipeline.named_steps["preprocessor"]
xgb_model = final_pipeline.named_steps["classifier"]

X_test_transformed = preprocessor.transform(X_test)

# get feature names post-one-hot-encoding
feature_names = preprocessor.get_feature_names_out()  # TODO

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_transformed)

# Global importance: mean absolute SHAP value per feature, across all test rows
global_importance = np.abs(shap_values).mean(axis=0)

# pair global_importance with feature_names, sort descending, keep top N (decide N: 10? 15?)

TOP_N = 15
importance_pairs = list(zip(feature_names, global_importance))
importance_pairs.sort(key=lambda pair: pair[1], reverse=True)
top_n_pairs = importance_pairs[:TOP_N]

metrics_artifact["shap_global_importance"] = [
    {"feature": name, "importance": float(val)} for name, val in top_n_pairs
]

# save a SHAP summary/beeswarm plot as an image for the dashboard
shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False)
plt.savefig("./models/shap_summary.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 9. Write final metrics.json (after SHAP keys are added above)
# ---------------------------------------------------------------------------
with open("./models/metrics.json", "w") as f:
    json.dump(metrics_artifact, f, indent=2)

# ---------------------------------------------------------------------------
# 10. Save the fitted pipeline artifact
# ---------------------------------------------------------------------------
joblib.dump(final_pipeline, "./models/pipeline.joblib")