"""
Pydantic schemas — backend/app/schemas.py
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


# ---------------------------------------------------------------------------
# 1. Input schema for POST /predict
# ---------------------------------------------------------------------------
class PassengerInput(BaseModel):
    gender: Literal["Male", "Female"] = Field(..., alias="Gender")
    customer_type: Literal["Loyal Customer", "disloyal Customer"] = Field(..., alias="Customer Type")
    type_of_travel: Literal["Personal Travel", "Business travel"] = Field(..., alias="Type of Travel")
    travel_class: Literal["Eco Plus", "Business", "Eco"] = Field(..., alias="Class")

    age: int = Field(..., ge=0, le=120, alias="Age")
    flight_distance: int = Field(..., ge=0, alias="Flight Distance")
    departure_delay_minutes: int = Field(..., ge=0, alias="Departure Delay in Minutes")
    arrival_delay_minutes: float | None = Field(None, ge=0, alias="Arrival Delay in Minutes")

    inflight_wifi_service: int = Field(..., ge=0, le=5, alias="Inflight wifi service")
    departure_arrival_time_convenient: int = Field(..., ge=0, le=5, alias="Departure/Arrival time convenient")
    ease_of_online_booking: int = Field(..., ge=0, le=5, alias="Ease of Online booking")
    gate_location: int = Field(..., ge=0, le=5, alias="Gate location")
    food_and_drink: int = Field(..., ge=0, le=5, alias="Food and drink")
    online_boarding: int = Field(..., ge=0, le=5, alias="Online boarding")
    seat_comfort: int = Field(..., ge=0, le=5, alias="Seat comfort")
    inflight_entertainment: int = Field(..., ge=0, le=5, alias="Inflight entertainment")
    onboard_service: int = Field(..., ge=0, le=5, alias="On-board service")
    leg_room_service: int = Field(..., ge=0, le=5, alias="Leg room service")
    baggage_handling: int = Field(..., ge=0, le=5, alias="Baggage handling")
    checkin_service: int = Field(..., ge=0, le=5, alias="Checkin service")
    inflight_service: int = Field(..., ge=0, le=5, alias="Inflight service")
    cleanliness: int = Field(..., ge=0, le=5, alias="Cleanliness")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "Gender": "Male",
                "Customer Type": "Loyal Customer",
                "Age": 35,
                "Type of Travel": "Business travel",
                "Class": "Business",
                "Flight Distance": 1200,
                "Inflight wifi service": 3,
                "Departure Delay in Minutes": 10,
                "Arrival Delay in Minutes": 5,
                "Departure/Arrival time convenient": 3,
                "Ease of Online booking": 2,
                "Gate location": 3,
                "Food and drink": 4,
                "Online boarding": 5,
                "Seat comfort": 1,
                "Inflight entertainment": 1,
                "On-board service": 3,
                "Leg room service": 4,
                "Baggage handling": 2,
                "Checkin service": 4,
                "Inflight service": 2,
                "Cleanliness": 4,
            }
        },
    )


# ---------------------------------------------------------------------------
# 2. Output schema for POST /predict
# ---------------------------------------------------------------------------
class DissatisfactionDriver(BaseModel):
    feature: str
    shap_value: float
    direction: Literal["increases_dissatisfaction", "decreases_dissatisfaction"]


class PredictionResponse(BaseModel):
    dissatisfaction_probability: float = Field(..., ge=0, le=1)
    predicted_label: Literal["satisfied", "dissatisfied"]
    top_dissatisfaction_drivers: list[DissatisfactionDriver]  # model_loader.py returns top 5


# ---------------------------------------------------------------------------
# 3. Output schema for GET /model-metrics
# ---------------------------------------------------------------------------
class ModelMetrics(BaseModel):
    roc_auc: float
    classification_report: dict
    confusion_matrix: list[list[int]]


class ShapGlobalImportance(BaseModel):
    feature: str
    importance: float


class ModelMetricsResponse(BaseModel):
    best_params: dict
    models: dict[str, ModelMetrics]
    shap_global_importance: list[ShapGlobalImportance]


# ---------------------------------------------------------------------------
# 4. Output schema for GET /eda-summary
# ---------------------------------------------------------------------------
class ClassSatisfactionBreakdown(BaseModel):
    travel_class: str
    satisfaction_rate: float


class TravelTypeSatisfactionBreakdown(BaseModel):
    type_of_travel: str
    satisfaction_rate: float


class RatingDistribution(BaseModel):
    feature: str
    mean_rating: float
    value_counts: dict[str, int]


class EdaSummaryResponse(BaseModel):
    total_passengers: int
    overall_satisfaction_rate: float
    satisfaction_by_class: list[ClassSatisfactionBreakdown]
    satisfaction_by_travel_type: list[TravelTypeSatisfactionBreakdown]
    rating_distributions: list[RatingDistribution]
    feature_correlations: dict[str, dict[str, float]]
    mean_departure_delay_by_satisfaction: dict[str, float]


# ---------------------------------------------------------------------------
# 5. Health check
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool