"""
compute_eda_summary.py — run once to produce models/eda_summary.json
Matches schemas.EdaSummaryResponse exactly.
"""
import json
import pandas as pd

df = pd.read_csv("train.csv")
df = df.drop(columns=["Unnamed: 0", "id"])

y_bool = df["satisfaction"].apply(lambda x: False if x == "satisfied" else True)  # True = dissatisfied
satisfied_mask = ~y_bool  # True where satisfied

RATING_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
    "Gate location", "Food and drink", "Online boarding", "Seat comfort",
    "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Inflight service", "Cleanliness",
]
NUMERIC_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]

# --- overall ---
total_passengers = len(df)
overall_satisfaction_rate = float(satisfied_mask.mean())

# --- satisfaction by class ---
satisfaction_by_class = [
    {"travel_class": cls, "satisfaction_rate": float(satisfied_mask[df["Class"] == cls].mean())}
    for cls in df["Class"].unique()
]

# --- satisfaction by travel type ---
satisfaction_by_travel_type = [
    {"type_of_travel": tt, "satisfaction_rate": float(satisfied_mask[df["Type of Travel"] == tt].mean())}
    for tt in df["Type of Travel"].unique()
]

# --- rating distributions ---
rating_distributions = []
for col in RATING_COLS:
    value_counts = df[col].value_counts().to_dict()
    value_counts = {str(k): int(v) for k, v in value_counts.items()}  # JSON keys must be strings
    rating_distributions.append({
        "feature": col,
        "mean_rating": float(df[col].mean()),
        "value_counts": value_counts,
    })

# --- feature correlations (numeric + rating columns only) ---
corr_df = df[NUMERIC_COLS + RATING_COLS].corr()
feature_correlations = {
    col: {other_col: float(corr_df.loc[col, other_col]) for other_col in corr_df.columns}
    for col in corr_df.columns
}

# --- mean departure delay by satisfaction ---
mean_departure_delay_by_satisfaction = {
    "satisfied": float(df.loc[satisfied_mask, "Departure Delay in Minutes"].mean()),
    "dissatisfied": float(df.loc[y_bool, "Departure Delay in Minutes"].mean()),
}

eda_summary = {
    "total_passengers": total_passengers,
    "overall_satisfaction_rate": overall_satisfaction_rate,
    "satisfaction_by_class": satisfaction_by_class,
    "satisfaction_by_travel_type": satisfaction_by_travel_type,
    "rating_distributions": rating_distributions,
    "feature_correlations": feature_correlations,
    "mean_departure_delay_by_satisfaction": mean_departure_delay_by_satisfaction,
}

with open("../models/eda_summary.json", "w") as f:
    json.dump(eda_summary, f, indent=2)

print("eda_summary.json written.")