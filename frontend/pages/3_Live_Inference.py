# frontend/pages/3_Live_Inference.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Live Inference", layout="wide")
st.title("Live Passenger Inference & CX Intervention")

# ---------------------------------------------------------------------------
# Maps a raw feature name (as it appears after one-hot encoding, e.g.
# "remainder__Inflight wifi service" or "cat__Class_Business") to a
# human-readable CX intervention suggestion. Simple rule-based lookup —
# not a second model, just a presentation layer over the SHAP drivers.
# ---------------------------------------------------------------------------
INTERVENTION_MAP = {
    "Inflight wifi service": "Prioritize wifi reliability — offer a service credit or free wifi voucher.",
    "Online boarding": "Simplify the online boarding flow; consider a follow-up tutorial email.",
    "Type of Travel": "Segment CX messaging differently for business vs personal travelers.",
    "Customer Type": "Loyalty status strongly affects sentiment — consider a loyalty check-in outreach.",
    "Class": "Cabin class expectations weren't met — consider a class-specific service review.",
    "Baggage handling": "Flag for baggage handling process review at the departure airport.",
    "Checkin service": "Offer expedited check-in or a dedicated check-in agent for this passenger segment.",
    "Inflight service": "Escalate to cabin crew service quality review.",
    "Inflight entertainment": "Check entertainment system functionality on this route.",
    "Age": "Consider age-tailored service messaging (no direct action — demographic signal only).",
    "Seat comfort": "Flag for seat/cabin maintenance review.",
    "Gate location": "Review gate assignment policy for this passenger's route.",
    "Leg room service": "Consider a seat upgrade offer or legroom-related compensation.",
    "Cleanliness": "Escalate to cabin cleaning/turnaround team.",
    "Food and drink": "Review catering quality on this route.",
    "Departure/Arrival time convenient": "Review scheduling — passenger found timing inconvenient.",
    "Ease of Online booking": "Audit the online booking UX for friction points.",
    "On-board service": "Escalate to onboard service quality review.",
}


def get_intervention(feature_name: str) -> str:
    """Strip ColumnTransformer prefixes (cat__/remainder__/num_impute__) and
    any one-hot suffix (e.g. _Business) to match against INTERVENTION_MAP."""
    clean = feature_name.split("__", 1)[-1]  # drop "cat__" / "remainder__" / "num_impute__"
    for known_feature in INTERVENTION_MAP:
        if clean.startswith(known_feature):
            return INTERVENTION_MAP[known_feature]
    return "Review this factor manually — no predefined intervention mapped."


# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.subheader("Passenger Details")

with st.form("passenger_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
        type_of_travel = st.selectbox("Type of Travel", ["Personal Travel", "Business travel"])
        travel_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
        age = st.number_input("Age", min_value=0, max_value=120, value=35)

    with col2:
        flight_distance = st.number_input("Flight Distance", min_value=0, value=1000)
        departure_delay = st.number_input("Departure Delay in Minutes", min_value=0, value=0)
        arrival_delay = st.number_input("Arrival Delay in Minutes", min_value=0, value=0)
        inflight_wifi = st.slider("Inflight wifi service", 0, 5, 3)
        dep_arr_time_convenient = st.slider("Departure/Arrival time convenient", 0, 5, 3)
        ease_online_booking = st.slider("Ease of Online booking", 0, 5, 3)

    with col3:
        gate_location = st.slider("Gate location", 0, 5, 3)
        food_and_drink = st.slider("Food and drink", 0, 5, 3)
        online_boarding = st.slider("Online boarding", 0, 5, 3)
        seat_comfort = st.slider("Seat comfort", 0, 5, 3)
        inflight_entertainment = st.slider("Inflight entertainment", 0, 5, 3)

    col4, col5 = st.columns(2)
    with col4:
        onboard_service = st.slider("On-board service", 0, 5, 3)
        leg_room_service = st.slider("Leg room service", 0, 5, 3)
        baggage_handling = st.slider("Baggage handling", 0, 5, 3)
    with col5:
        checkin_service = st.slider("Checkin service", 0, 5, 3)
        inflight_service = st.slider("Inflight service", 0, 5, 3)
        cleanliness = st.slider("Cleanliness", 0, 5, 3)

    submitted = st.form_submit_button("Predict Satisfaction")

# ---------------------------------------------------------------------------
# On submit: call the API, display results
# ---------------------------------------------------------------------------
if submitted:
    payload = {
        "Gender": gender,
        "Customer Type": customer_type,
        "Type of Travel": type_of_travel,
        "Class": travel_class,
        "Age": age,
        "Flight Distance": flight_distance,
        "Departure Delay in Minutes": departure_delay,
        "Arrival Delay in Minutes": arrival_delay,
        "Inflight wifi service": inflight_wifi,
        "Departure/Arrival time convenient": dep_arr_time_convenient,
        "Ease of Online booking": ease_online_booking,
        "Gate location": gate_location,
        "Food and drink": food_and_drink,
        "Online boarding": online_boarding,
        "Seat comfort": seat_comfort,
        "Inflight entertainment": inflight_entertainment,
        "On-board service": onboard_service,
        "Leg room service": leg_room_service,
        "Baggage handling": baggage_handling,
        "Checkin service": checkin_service,
        "Inflight service": inflight_service,
        "Cleanliness": cleanliness,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Prediction request failed: {e}")
        st.stop()

    st.divider()
    st.subheader("Prediction Result")

    proba = result["dissatisfaction_probability"]
    label = result["predicted_label"]

    result_col1, result_col2 = st.columns(2)
    with result_col1:
        if label == "dissatisfied":
            st.error(f"**{proba:.0%} Probability of Dissatisfaction**")
        else:
            st.success(f"**{proba:.0%} Probability of Dissatisfaction**")
    with result_col2:
        st.metric("Predicted Label", label.capitalize())

    st.divider()
    st.subheader("Key Dissatisfaction Drivers")

    drivers = result["top_dissatisfaction_drivers"]
    drivers_df = pd.DataFrame(drivers)

    fig_drivers = px.bar(
        drivers_df,
        x="shap_value",
        y="feature",
        orientation="h",
        color="direction",
        color_discrete_map={
            "increases_dissatisfaction": "#d62728",
            "decreases_dissatisfaction": "#2ca02c",
        },
        labels={"shap_value": "SHAP Value", "feature": "Feature"},
    )
    st.plotly_chart(fig_drivers, width="stretch")

    st.divider()
    st.subheader("Recommended CX Interventions")

    for driver in drivers:
        if driver["direction"] == "increases_dissatisfaction":
            intervention = get_intervention(driver["feature"])
            st.warning(f"**{driver['feature']}** — {intervention}")

    if not any(d["direction"] == "increases_dissatisfaction" for d in drivers):
        st.info("No strong dissatisfaction drivers detected among the top factors — no intervention needed.")