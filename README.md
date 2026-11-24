# Credit Default Risk Prediction with Cost-Sensitive Decisions

Stack: Python (FastAPI, scikit-learn, XGBoost) for the backend, plain HTML/CSS/JavaScript (Chart.js) for the dashboard.

## Layout

    data/creditrisk_x_enhanced.csv   dataset (30,000 rows, has a train/validation/test split column)
    backend/
      features.py   rebuilds the 20 engineered features from the 23 raw ones
      costs.py      cost matrix, cost-vs-threshold curve, best threshold
      train.py      trains Logistic Regression, SVM, Decision Tree, Random Forest, XGBoost
      service.py    analysis + scoring logic used by the API
      main.py       FastAPI routes (also serves the frontend)
    frontend/       index.html, style.css, app.js

## Run it

    cd backend
    python -m venv venv
    venv\Scripts\activate          # Windows   (source venv/bin/activate on Linux/Mac)
    pip install -r requirements.txt
    python train.py                # one-time, creates backend/artifacts/
    uvicorn main:app --reload

Open http://127.0.0.1:8000 for the dashboard and http://127.0.0.1:8000/docs for the API docs.

## How the pieces map to the problem statement

| Requirement | Where |
|---|---|
| Compare LR, SVM, Decision Tree, Random Forest | `train.py`, table in dashboard section 2 |
| Explicit cost matrix | `costs.py` (`DEFAULT_COSTS`) + editable matrix in section 1 |
| Threshold that minimises cost | `best_threshold()` in `costs.py`, chart in section 3 |
| Precision-recall curves | `/api/pr-curves`, chart in section 4 |
| Risk-scoring dashboard | section 5 (`/api/predict`) |
| Transparent decision rules | section 6: policy, tree rules, LR coefficients, per-customer XGBoost reasons |
