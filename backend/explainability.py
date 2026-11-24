"""Explainability Module for CreditRisk-X.

Calculates individual feature contributions using TreeSHAP on the XGBoost model.
Returns friendly feature labels, directional impacts, and values.
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xgboost

from model import get_model
from prediction import validate_and_prepare_features

FRIENDLY_NAMES = {
    "PAY_0": "Repayment status, last month",
    "PAY_2": "Repayment status, 2 months ago",
    "PAY_3": "Repayment status, 3 months ago",
    "PAY_4": "Repayment status, 4 months ago",
    "PAY_5": "Repayment status, 5 months ago",
    "PAY_6": "Repayment status, 6 months ago",
    "LIMIT_BAL": "Credit limit",
    "AGE": "Age",
    "SEX": "Gender",
    "EDUCATION": "Education level",
    "MARRIAGE": "Marital status",
    "utilization_latest": "Credit utilization (latest)",
    "utilization_mean_6m": "Credit utilization (6-month avg)",
    "utilization_max_6m": "Credit utilization (6-month peak)",
    "utilization_std_6m": "Credit utilization volatility",
    "utilization_trend_6m": "Utilization slope (6 months)",
    "late_month_count_6m": "Late months count (last 6 months)",
    "severe_late_month_count_6m": "Severely late months (2+ months behind)",
    "recent_late_month_count_3m": "Late months in recent 3 months",
    "max_delay_severity_6m": "Worst repayment delay in last 6 months",
    "consecutive_recent_late_months": "Consecutive recent late months",
    "avg_delay_severity_6m": "Average repayment delay severity",
    "payment_active_months_6m": "Active payment months (last 6 months)",
    "total_payment_6m": "Total payments made (6 months)",
    "avg_payment_6m": "Average monthly payment",
    "payment_volatility_6m": "Payment volatility (std)",
    "payment_trend_slope_6m": "Payment trend slope",
    "avg_bill_6m": "Average monthly bill",
    "max_bill_6m": "Peak monthly bill statement",
    "bill_volatility_6m": "Bill statement volatility",
    "bill_trend_slope_6m": "Bill balance trend slope",
    "BILL_AMT1": "Latest bill statement amount",
    "BILL_AMT2": "Bill statement 2 months ago",
    "BILL_AMT3": "Bill statement 3 months ago",
    "BILL_AMT4": "Bill statement 4 months ago",
    "BILL_AMT5": "Bill statement 5 months ago",
    "BILL_AMT6": "Bill statement 6 months ago",
    "PAY_AMT1": "Latest payment amount",
    "PAY_AMT2": "Payment 2 months ago",
    "PAY_AMT3": "Payment 3 months ago",
    "PAY_AMT4": "Payment 4 months ago",
    "PAY_AMT5": "Payment 5 months ago",
    "PAY_AMT6": "Payment 6 months ago",
}

DISCLAIMER = (
    "Note: SHAP values represent statistical feature contributions to this specific XGBoost model prediction. "
    "They reflect mathematical model attribution and correlation within the training data, and do not prove "
    "individual real-world causality, moral fault, or regulatory fairness compliance."
)


def explain_prediction(
    features: Union[Dict[str, Any], pd.DataFrame],
    top_n: int = 6,
    model_name: str = "xgboost"
) -> Dict[str, Any]:
    """
    Generate TreeSHAP local explanations for an individual borrower record.
    Returns top contributors raising and lowering default risk.
    """
    if isinstance(features, pd.DataFrame):
        X = features
    else:
        X = validate_and_prepare_features(features)

    model = get_model(model_name)
    booster = model.get_booster()

    # TreeSHAP via XGBoost pred_contribs=True
    # Output shape: [1, n_features + 1] where the last element is the base margin / intercept
    dmat = xgboost.DMatrix(X)
    contribs_raw = booster.predict(dmat, pred_contribs=True)
    contribs = contribs_raw[0][:-1]
    base_value = float(contribs_raw[0][-1])

    # Rank by absolute magnitude
    abs_order = np.argsort(-np.abs(contribs))[:top_n]

    reasons: List[Dict[str, Any]] = []
    for idx in abs_order:
        feat_name = X.columns[idx]
        val = float(X.iloc[0, idx])
        contrib = float(contribs[idx])
        direction = "raises risk" if contrib > 0 else "lowers risk"

        reasons.append({
            "feature": feat_name,
            "label": FRIENDLY_NAMES.get(feat_name, feat_name),
            "value": round(val, 4),
            "contribution": round(contrib, 4),
            "direction": direction,
        })

    return {
        "model_name": model_name,
        "base_value": round(base_value, 4),
        "top_features": reasons,
        "disclaimer": DISCLAIMER,
    }
