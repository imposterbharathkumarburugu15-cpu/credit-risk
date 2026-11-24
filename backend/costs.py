"""Cost matrix, evaluation, and cost-efficiency helpers for CreditRisk-X.

Convention: 1 = default (positive class). A "flag" means we predict default.
                      predicted no-default | predicted default
    actual no-default        TN cost       |      FP cost   (false alarm)
    actual default           FN cost       |      TP cost   (missed default = FN)

Standard project assumptions:
    FP cost = $500 (lost customer / processing overhead)
    FN cost = $5,000 (charged-off principal / collection loss)
    In normalized units: fp = 1.0, fn = 5.0 (or 10.0 depending on scenario).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Default cost configuration
DEFAULT_COSTS = {"tn": 0.0, "fp": 1.0, "fn": 5.0, "tp": 0.0}
DEFAULT_DOLLAR_COSTS = {"fp": 500.0, "fn": 5000.0, "tp": 0.0, "tn": 0.0}

THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)


def confusion_counts(y_true, y_pred):
    """Calculate true negatives, false positives, false negatives, true positives."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def total_cost(counts, costs=None):
    """Compute total financial or unit cost given confusion counts and cost dictionary."""
    if costs is None:
        costs = DEFAULT_COSTS
    return sum(counts[k] * costs.get(k, 0.0) for k in ("tn", "fp", "fn", "tp"))


def cost_curve(y_true, proba, costs=None, thresholds=THRESHOLDS):
    """Total cost at every threshold across the specified range."""
    if costs is None:
        costs = DEFAULT_COSTS
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    out = []
    for t in thresholds:
        counts = confusion_counts(y_true, (proba >= t).astype(int))
        out.append(total_cost(counts, costs))
    return np.array(out)


def best_threshold(y_true, proba, costs=None, thresholds=THRESHOLDS):
    """Threshold with the lowest total cost (first one wins on ties)."""
    if costs is None:
        costs = DEFAULT_COSTS
    curve = cost_curve(y_true, proba, costs, thresholds=thresholds)
    idx = int(np.argmin(curve))
    return float(thresholds[idx]), float(curve[idx])


def evaluate_at(y_true, proba, threshold, costs=None):
    """Confusion matrix, precision/recall/F1 and cost at one threshold."""
    if costs is None:
        costs = DEFAULT_COSTS
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    counts = confusion_counts(y_true, pred)
    n = max(len(y_true), 1)
    tot = total_cost(counts, costs)
    return {
        "threshold": float(threshold),
        "confusion": counts,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "flag_rate": float(pred.mean()),
        "total_cost": float(tot),
        "cost_per_customer": float(tot / n),
    }


def calculate_cost(fp_count: int, fn_count: int, fp_cost: float = 500.0, fn_cost: float = 5000.0):
    """Calculate modeled classification cost under project assumptions."""
    total_fp_cost = fp_count * fp_cost
    total_fn_cost = fn_count * fn_cost
    total = total_fp_cost + total_fn_cost
    return {
        "fp_count": int(fp_count),
        "fn_count": int(fn_count),
        "fp_cost_per_unit": float(fp_cost),
        "fn_cost_per_unit": float(fn_cost),
        "total_fp_cost": float(total_fp_cost),
        "total_fn_cost": float(total_fn_cost),
        "total_cost": float(total),
    }


def compare_costs(
    baseline_fp: int,
    baseline_fn: int,
    optimized_fp: int,
    optimized_fn: int,
    fp_cost: float = 500.0,
    fn_cost: float = 5000.0,
):
    """Compare baseline strategy (e.g. 0.5 threshold) vs optimized strategy."""
    base = calculate_cost(baseline_fp, baseline_fn, fp_cost, fn_cost)
    opt = calculate_cost(optimized_fp, optimized_fn, fp_cost, fn_cost)
    reduction_amount = base["total_cost"] - opt["total_cost"]
    reduction_pct = (
        (reduction_amount / base["total_cost"] * 100.0) if base["total_cost"] > 0 else 0.0
    )
    return {
        "baseline": base,
        "optimized": opt,
        "cost_reduction_amount": round(reduction_amount, 2),
        "cost_reduction_percentage": round(reduction_pct, 2),
    }
