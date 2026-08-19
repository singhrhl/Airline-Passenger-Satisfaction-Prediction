# Airline Passenger Satisfaction — End-to-End ML Web App

A production-style machine learning web application that predicts airline passenger
satisfaction and surfaces the key drivers behind each prediction, built to explore a
full ML product lifecycle: EDA → model training/comparison → explainability (SHAP) →
REST API → interactive dashboard → containerized deployment.

**Stack:** FastAPI (backend) · Streamlit (frontend) · XGBoost · SHAP · Docker Compose

---

## Problem Statement

Using the [Kaggle Airline Passenger Satisfaction dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
(~104k passenger survey records), this project predicts whether a passenger is
**satisfied** or **dissatisfied/neutral**, and explains *why* — down to individual,
per-passenger drivers — so the output can support real CX (customer experience)
intervention decisions, not just a single accuracy number.

---

## Architecture

```
┌─────────────────────┐        REST API (JSON)       ┌────────────────────┐
│  Streamlit Frontend │ ───────────────────────────▶ │   FastAPI Backend  │
│                     │ ◀─────────────────────────── │                    │
│ • EDA Insights      │                              │ GET /health        │
│ • Model Performance │                              │ GET /eda-summary   │
│ • Live Inference    │                              │ GET /model-metrics │
└─────────────────────┘                              │ POST /predict      │
                                                     └────────────────────┘
                                                                │
                                                                ▼
                                                     ┌─────────────────────┐
                                                     │  pipeline.joblib    │
                                                     │  (preprocessing +   │
                                                     │   XGBoost model)    │
                                                     │  metrics.json       │
                                                     │  eda_summary.json   │
                                                     └─────────────────────┘
```

Both services run in separate Docker containers, orchestrated via Docker Compose,
communicating over Docker's internal network.

---

## Model Selection & Results

Three models were trained and compared to validate the choice of algorithm, not just
tune a single one blindly:

| Model                            | ROC-AUC | Notes                                                          |
|----------------------------------|---------|----------------------------------------------------------------|
| Logistic Regression (baseline)   | 0.9256  | Linear boundary — establishes the floor                        |
| XGBoost (default hyperparameters)| 0.9951  | Large jump — confirms non-linear feature interactions matter   |
| XGBoost (tuned, `GridSearchCV`)  | 0.9955  | Marginal gain over defaults (+0.0004)                          |

**Key takeaway:** the jump from linear → tree ensemble was the significant gain in
this problem; hyperparameter tuning on top of a strong default XGBoost model produced
only a negligible improvement. This is a real, useful finding — it means model *family*
choice mattered far more than fine-tuning effort here.

### Top Global Dissatisfaction Drivers (SHAP)

1. Inflight wifi service
2. Type of Travel (Business vs Personal)
3. Online boarding
4. Customer Type (Loyal vs disloyal)
5. Class (Business vs Eco/Eco Plus)

These align with publicly known analyses of this dataset, and were used to sanity-check
the model's local (per-passenger) explanations in the Live Inference page as well.

---

## Project Structure

```
airline_app/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI endpoints
│   │   ├── schemas.py         # Pydantic request/response models
│   │   └── model_loader.py    # Loads pipeline + runs inference/SHAP
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py                 # Streamlit entry point
│   ├── pages/
│   │   ├── 1_EDA_Insights.py
│   │   ├── 2_Model_Performance.py
│   │   └── 3_Live_Inference.py
│   ├── Dockerfile
│   └── requirements.txt
├── models/
│   ├── pipeline.joblib         # Fitted preprocessing + XGBoost pipeline
│   ├── metrics.json            # Model comparison metrics + SHAP global importance
│   └── eda_summary.json        # Precomputed EDA statistics
├── training/
│   ├── train.py                  # Training script (3-model comparison + SHAP)
│   ├── compute_eda_summary.py    # Generates eda_summary.json
│   └── train.csv                 # Kaggle dataset (see Data section)
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Live Demo
- Frontend: https://airline-passenger-satisfaction-prediction-kqdrgopvzcc8u36mm5bk.streamlit.app
- Backend API docs: https://airline-passenger-satisfaction-prediction-2zmh.onrender.com/docs
*(Backend is on a free tier and may take 30–60s to wake up on first request.)*

---

## Running Locally (without Docker)

**1. Backend**
```bash
cd backend
python -m venv .backend-env
source .backend-env/bin/activate     # Windows: .backend-env\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
API available at `http://localhost:8000` (interactive docs at `/docs`).

**2. Frontend** (in a separate terminal)
```bash
cd frontend
python -m venv .frontend-env
source .frontend-env/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Dashboard available at `http://localhost:8501`.

---

## Running with Docker Compose

```bash
docker compose up --build
```
- Frontend: `http://localhost:8501`
- Backend: `http://localhost:8000/docs`

The frontend container reaches the backend via Docker's internal service DNS
(`http://backend:8000`), configured through the `API_URL` environment variable in
`docker-compose.yml`.

---

## Retraining the Model

```bash
cd training
python train.py                  # regenerates models/pipeline.joblib and models/metrics.json
python compute_eda_summary.py    # regenerates models/eda_summary.json
```
Note: `scikit-learn==1.8.0` is pinned in both `backend/requirements.txt` and
`training/requirements.txt` — the pickled pipeline is version-sensitive, so keep
these in sync if you retrain with a different scikit-learn version.

---

## Screenshots

*(Add screenshots of the 3 Streamlit pages here — EDA Insights, Model Performance, Live Inference)*

```
![EDA Insights](docs/screenshot-eda.png)
![Model Performance](docs/screenshot-metrics.png)
![Live Inference](docs/screenshot-inference.png)
```

---

## Dataset

[Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
on Kaggle, adapted from a dataset originally shared by John D.

---

## Possible Extensions

- Add authentication for the `/predict` endpoint
- Persist prediction logs for monitoring/drift detection
- CI pipeline to auto-retrain and validate on new data
- Swap the SHAP feature-to-intervention mapping for a config-driven or LLM-generated version
