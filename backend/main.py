# To run:   uvicorn main:app --reload
# Then open http://127.0.0.1:8000/docs to test it in the browser.
import glob
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBClassifier
# Kaggle model handle = owner/model/framework/variation/version
KAGGLE_HANDLE = "manivardhangundoju/credit-default-risk-prediction-model-xgboost/scikitLearn/default/1"
MODEL_FILE = "xgboost.json"
THRESHOLD = 0.5
FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "utilization_latest", "utilization_mean_6m", "utilization_max_6m",
    "utilization_std_6m", "utilization_trend_6m",
    "avg_bill_6m", "max_bill_6m", "bill_volatility_6m", "bill_trend_slope_6m",
    "total_payment_6m", "avg_payment_6m", "payment_volatility_6m",
    "payment_trend_slope_6m", "payment_active_months_6m",
    "late_month_count_6m", "severe_late_month_count_6m",
    "recent_late_month_count_3m", "max_delay_severity_6m",
    "avg_delay_severity_6m", "consecutive_recent_late_months",
]

def find_model_file():
    """Look for xgboost.json: first on my computer, then download it.
    # 1) is the file already in this folder?
    if os.path.isfile(MODEL_FILE):
        return MODEL_FILE
    """
    import kagglehub
    folder = kagglehub.model_download(KAGGLE_HANDLE)
    found = glob.glob(os.path.join(folder, "**", MODEL_FILE), recursive=True)
    if not found:
        raise RuntimeError("Downloaded the model but could not find " + MODEL_FILE)
    return found[0]
model = XGBClassifier()
model.load_model(find_model_file())
print("Model loaded!")
# ---------------------------------------------------------------
# Step 3: Describe the input data with Pydantic
# FastAPI uses these classes to check the data automatically.
# If something is missing or has the wrong type, it sends back a 422 error.
# ---------------------------------------------------------------

class RawInput(BaseModel):
    """The 23 original columns of the dataset."""
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


class FullInput(RawInput):
    """RawInput + the 20 engineered columns (inheritance saves me from retyping)."""
    utilization_latest: float
    utilization_mean_6m: float
    utilization_max_6m: float
    utilization_std_6m: float
    utilization_trend_6m: float
    avg_bill_6m: float
    max_bill_6m: float
    bill_volatility_6m: float
    bill_trend_slope_6m: float
    total_payment_6m: float
    avg_payment_6m: float
    payment_volatility_6m: float
    payment_trend_slope_6m: float
    payment_active_months_6m: int
    late_month_count_6m: int
    severe_late_month_count_6m: int
    recent_late_month_count_3m: int
    max_delay_severity_6m: int
    avg_delay_severity_6m: float
    consecutive_recent_late_months: int


# ---------------------------------------------------------------
# Step 4: Feature engineering
# The model needs 43 columns, but a user will usually only know the first 23,
# so this function calculates the other 20.
#
# NOTE: I guessed these formulas from the column names. They MUST match
# what I did in my training notebook, otherwise the predictions will be wrong.
# ---------------------------------------------------------------

def slope(values):
    """How fast something is going up or down (a straight line fitted to 6 points)."""
    x = np.arange(len(values))
    return float(np.polyfit(x, values, 1)[0])


def make_features(data):
    # In the dataset, month 1 is the NEWEST and month 6 is the OLDEST.
    # For trends I want oldest -> newest, so I reverse the order here.
    bills = np.array([data.BILL_AMT6, data.BILL_AMT5, data.BILL_AMT4, data.BILL_AMT3, data.BILL_AMT2, data.BILL_AMT1])
    payments = np.array([data.PAY_AMT6, data.PAY_AMT5, data.PAY_AMT4, data.PAY_AMT3, data.PAY_AMT2, data.PAY_AMT1])
    # PAY_ columns: newest -> oldest. Positive number = months late.
    status = np.array([data.PAY_0, data.PAY_2, data.PAY_3,
                       data.PAY_4, data.PAY_5, data.PAY_6])

    # how much of the credit limit is used each month (avoid dividing by 0)
    utilization = bills / max(data.LIMIT_BAL, 1)

    # -1 and 0 mean "paid on time", so treat them as 0 delay
    delay = np.clip(status, 0, None)
    is_late = status > 0

    # count how many months in a row (starting from the latest) were late
    consecutive_late = 0
    for late in is_late:
        if not late:
            break
        consecutive_late += 1

    return {
        "utilization_latest": float(utilization[-1]),
        "utilization_mean_6m": float(utilization.mean()),
        "utilization_max_6m": float(utilization.max()),
        "utilization_std_6m": float(utilization.std()),
        "utilization_trend_6m": slope(utilization),
        "avg_bill_6m": float(bills.mean()),
        "max_bill_6m": float(bills.max()),
        "bill_volatility_6m": float(bills.std()),
        "bill_trend_slope_6m": slope(bills),
        "total_payment_6m": float(payments.sum()),
        "avg_payment_6m": float(payments.mean()),
        "payment_volatility_6m": float(payments.std()),
        "payment_trend_slope_6m": slope(payments),
        "payment_active_months_6m": int((payments > 0).sum()),
        "late_month_count_6m": int(is_late.sum()),
        "severe_late_month_count_6m": int((status >= 2).sum()),
        "recent_late_month_count_3m": int(is_late[:3].sum()), 
        "max_delay_severity_6m": int(delay.max()),
        "avg_delay_severity_6m": float(delay.mean()),
        "consecutive_recent_late_months": consecutive_late,
    }

def run_model(row):
    df = pd.DataFrame([row])[FEATURES]            # one row, columns in the right order
    probability = float(model.predict_proba(df)[0][1])   # [1] = chance of default
    return {
        "default_probability": round(probability, 4),
        "prediction": int(probability >= THRESHOLD),     # 1 = default, 0 = no default
        "threshold": THRESHOLD,
    }

app = FastAPI(title="Credit Default Prediction API")

@app.get("/")
def home():
    return {"message": "API is running. Go to /docs to try it out."}

@app.get("/health")
def health():
    return {"status": "ok", "features_expected": len(FEATURES)}

@app.post("/predict")
def predict(data: FullInput):
    return run_model(data.model_dump())

@app.post("/predict/raw")
def predict_raw(data: RawInput):
    row = data.model_dump()
    row.update(make_features(data))
    return run_model(row)
