"""Threshold Optimization Engine for CreditRisk-X.

Finds the cost-minimizing decision threshold using validation set data,
evaluating precision, recall, F1-score, and financial loss curves.
Never optimizes on held-out test data.
"""
from typing import Any, Dict, List, Optional

import numpy as np

import config
from costs import THRESHOLDS, confusion_counts, evaluate_at
from model import get_scores


def optimize_threshold(
    model_name: str = "xgboost",
    fp_cost: Optional[float] = None,
    fn_cost: Optional[float] = None,
    min_recall: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Find cost-optimal threshold on validation set (Y_VAL, VAL_PROBA).
    Evaluates every threshold from 0.01 to 0.99.
    """
    if fp_cost is None:
        fp_cost = config.DEFAULT_FP_COST
    if fn_cost is None:
        fn_cost = config.DEFAULT_FN_COST

    scores = get_scores()
    y_val = scores["y_val"]
    val_key = f"val_{model_name}"
    if val_key not in scores:
        raise KeyError(f"Validation scores for model '{model_name}' not available.")
    val_proba = scores[val_key]

    cost_dict = {"tn": 0.0, "fp": float(fp_cost), "fn": float(fn_cost), "tp": 0.0}

    curve_points = []
    best_thr = 0.5
    best_cost = float("inf")
    best_eval = None

    for t in THRESHOLDS:
        ev = evaluate_at(y_val, val_proba, t, cost_dict)
        rec = ev["recall"]
        tot_cost = ev["total_cost"]

        curve_points.append({
            "threshold": round(float(t), 2),
            "total_cost": round(tot_cost, 2),
            "precision": round(ev["precision"], 4),
            "recall": round(rec, 4),
            "f1": round(ev["f1"], 4),
            "fp": ev["confusion"]["fp"],
            "fn": ev["confusion"]["fn"],
        })

        if min_recall is not None and rec < min_recall:
            continue

        if tot_cost < best_cost:
            best_cost = tot_cost
            best_thr = float(t)
            best_eval = ev

    # Compare against standard 0.5 threshold
    eval_at_half = evaluate_at(y_val, val_proba, 0.5, cost_dict)
    saving = eval_at_half["total_cost"] - best_cost
    saving_pct = (
        (saving / eval_at_half["total_cost"] * 100.0)
        if eval_at_half["total_cost"] > 0
        else 0.0
    )

    return {
        "model_name": model_name,
        "dataset_used": "validation",
        "sample_size": len(y_val),
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
        "optimal_threshold": round(best_thr, 2),
        "minimum_cost": round(best_cost, 2),
        "cost_at_half": round(eval_at_half["total_cost"], 2),
        "savings_amount": round(saving, 2),
        "savings_percentage": round(saving_pct, 2),
        "evaluation_at_optimal": best_eval,
        "evaluation_at_half": eval_at_half,
        "threshold_curve": curve_points,
    }


def compare_thresholds(
    thresholds: List[float],
    model_name: str = "xgboost",
    fp_cost: Optional[float] = None,
    fn_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """Compare performance and cost across a custom list of thresholds on validation data."""
    if fp_cost is None:
        fp_cost = config.DEFAULT_FP_COST
    if fn_cost is None:
        fn_cost = config.DEFAULT_FN_COST

    scores = get_scores()
    y_val = scores["y_val"]
    val_proba = scores[f"val_{model_name}"]
    cost_dict = {"tn": 0.0, "fp": float(fp_cost), "fn": float(fn_cost), "tp": 0.0}

    results = []
    for t in thresholds:
        ev = evaluate_at(y_val, val_proba, t, cost_dict)
        results.append({
            "threshold": round(float(t), 4),
            "precision": round(ev["precision"], 4),
            "recall": round(ev["recall"], 4),
            "f1": round(ev["f1"], 4),
            "fp": ev["confusion"]["fp"],
            "fn": ev["confusion"]["fn"],
            "total_cost": round(ev["total_cost"], 2),
        })

    return {
        "model_name": model_name,
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
        "comparisons": results,
    }
