"""All the scoring / analysis logic lives here so main.py can stay thin."""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from costs import THRESHOLDS, best_threshold, cost_curve, evaluate_at, confusion_counts
from features import add_engineered_features, clean_categories

ARTIFACT_DIR = Path(__file__).parent / "artifacts"

MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "svm": "SVM (linear, calibrated)",
    "tree": "Decision Tree",
    "forest": "Random Forest",
    "xgboost": "XGBoost (primary)",
}
PRIMARY = "xgboost"

FRIENDLY = {
    "PAY_0": "Repayment status, last month",
    "PAY_2": "Repayment status, 2 months ago",
    "PAY_3": "Repayment status, 3 months ago",
    "LIMIT_BAL": "Credit limit",
    "AGE": "Age",
    "utilization_latest": "Credit utilization (latest)",
    "utilization_mean_6m": "Credit utilization (6-month avg)",
    "utilization_max_6m": "Credit utilization (6-month peak)",
    "late_month_count_6m": "Late months in last 6",
    "severe_late_month_count_6m": "Severely late months (2+ behind)",
    "recent_late_month_count_3m": "Late months in last 3",
    "max_delay_severity_6m": "Worst delay in last 6 months",
    "consecutive_recent_late_months": "Current run of late months",
    "avg_delay_severity_6m": "Average delay severity",
    "payment_active_months_6m": "Months with a payment made",
    "total_payment_6m": "Total paid in 6 months",
    "avg_payment_6m": "Average monthly payment",
    "avg_bill_6m": "Average monthly bill",
}


def _load():
    if not (ARTIFACT_DIR / "models.joblib").exists():
        raise RuntimeError("No trained models found. Run `python train.py` first.")
    models = joblib.load(ARTIFACT_DIR / "models.joblib")
    scores = np.load(ARTIFACT_DIR / "scores.npz")
    meta = json.loads((ARTIFACT_DIR / "meta.json").read_text())
    return models, scores, meta


MODELS, SCORES, META = _load()
Y_VAL, Y_TEST = SCORES["y_val"], SCORES["y_test"]
VAL = {name: SCORES[f"val_{name}"] for name in MODELS}
TEST = {name: SCORES[f"test_{name}"] for name in MODELS}


def _cost_key(costs):
    return (costs["tn"], costs["fp"], costs["fn"], costs["tp"])


def _costs_from_key(key):
    return dict(zip(("tn", "fp", "fn", "tp"), key))


@lru_cache(maxsize=256)
def _threshold_cached(model_name, key):
    thr, _ = best_threshold(Y_VAL, VAL[model_name], _costs_from_key(key))
    return thr


def threshold_for(model_name, costs):
    return _threshold_cached(model_name, _cost_key(costs))


@lru_cache(maxsize=1)
def _ranking_metrics():
    return {
        name: {
            "roc_auc": float(roc_auc_score(Y_TEST, TEST[name])),
            "pr_auc": float(average_precision_score(Y_TEST, TEST[name])),
        }
        for name in MODELS
    }


def comparison(costs):
    rows = []
    for name in MODELS:
        thr = threshold_for(name, costs)
        at_best = evaluate_at(Y_TEST, TEST[name], thr, costs)
        at_half = evaluate_at(Y_TEST, TEST[name], 0.5, costs)
        saving = 0.0
        if at_half["total_cost"] > 0:
            saving = 100 * (at_half["total_cost"] - at_best["total_cost"]) / at_half["total_cost"]
        rows.append({
            "model": name,
            "label": MODEL_LABELS[name],
            **_ranking_metrics()[name],
            "best_threshold": thr,
            "at_best": at_best,
            "cost_at_half": at_half["total_cost"],
            "recall_at_half": at_half["recall"],
            "saving_pct": round(saving, 1),
        })

    # reference points: what if we skip the model entirely?
    n_pos = int((Y_TEST == 1).sum())
    n_neg = int((Y_TEST == 0).sum())
    return {
        "costs": costs,
        "models": rows,
        "baselines": {
            "approve_everyone": n_pos * costs["fn"] + n_neg * costs["tn"],
            "flag_everyone": n_neg * costs["fp"] + n_pos * costs["tp"],
        },
        "test_size": int(len(Y_TEST)),
    }


def cost_curves(costs):
    return {
        "thresholds": THRESHOLDS.tolist(),
        "curves": {
            name: {
                "label": MODEL_LABELS[name],
                "cost": cost_curve(Y_VAL, VAL[name], costs).round(2).tolist(),
                "best_threshold": threshold_for(name, costs),
            }
            for name in MODELS
        },
    }


@lru_cache(maxsize=1)
def pr_curves():
    out = {}
    for name in MODELS:
        precision, recall, _ = precision_recall_curve(Y_TEST, TEST[name])
        # ~150 points is plenty for a chart
        idx = np.unique(np.linspace(0, len(recall) - 1, 150).astype(int))
        out[name] = {
            "label": MODEL_LABELS[name],
            "average_precision": round(float(average_precision_score(Y_TEST, TEST[name])), 4),
            "points": [{"x": round(float(recall[i]), 4), "y": round(float(precision[i]), 4)} for i in idx],
        }
    return {"base_rate": round(float(Y_TEST.mean()), 4), "curves": out}


def rules():
    return {
        "tree_rules": META["tree_rules"],
        "logreg_top": META["logreg_top"],
        "xgb_importance": META["xgb_importance"],
    }


def risk_band(proba, threshold):
    """The decision policy shown on the dashboard. Keep it in sync with the text there."""
    if proba >= threshold:
        return "HIGH", "Flag: decline or send to manual review"
    if proba >= 0.5 * threshold:
        return "MEDIUM", "Approve with a lower limit and monitoring"
    return "LOW", "Approve"


def score_customer(raw, costs):
    frame = clean_categories(pd.DataFrame([raw]))
    X = add_engineered_features(frame)

    model = MODELS[PRIMARY]
    proba = float(model.predict_proba(X)[0, 1])
    threshold = threshold_for(PRIMARY, costs)
    band, decision = risk_band(proba, threshold)

    # xgboost can give per-feature contributions (log-odds) for each prediction
    contribs = model.get_booster().predict(xgboost.DMatrix(X), pred_contribs=True)[0][:-1]
    order = np.argsort(-np.abs(contribs))[:6]
    reasons = []
    for i in order:
        feature = X.columns[i]
        reasons.append({
            "feature": feature,
            "label": FRIENDLY.get(feature, feature),
            "value": round(float(X.iloc[0, i]), 4),
            "contribution": round(float(contribs[i]), 3),
            "direction": "raises risk" if contribs[i] > 0 else "lowers risk",
        })

    return {
        "probability": round(proba, 4),
        "threshold": threshold,
        "band": band,
        "decision": decision,
        "reasons": reasons,
        "model": MODEL_LABELS[PRIMARY],
    }


def sample_customer(kind):
    return META["samples"].get(kind)
