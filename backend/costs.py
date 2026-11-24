"""Cost matrix + threshold helpers.

Convention: 1 = default (positive class). A "flag" means we predict default.

                      predicted no-default | predicted default
    actual no-default        TN cost       |      FP cost   (false alarm)
    actual default           FN cost       |      TP cost   (missed default = FN)
"""
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Business assumptions, in "units" of one false alarm. A missed default loses
# principal, a false alarm only loses some interest income, so FN >> FP.
DEFAULT_COSTS = {"tn": 0.0, "fp": 1.0, "fn": 5.0, "tp": 0.0}

THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)


def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def total_cost(counts, costs):
    return sum(counts[k] * costs[k] for k in ("tn", "fp", "fn", "tp"))


def cost_curve(y_true, proba, costs, thresholds=THRESHOLDS):
    """Total cost at every threshold."""
    out = []
    for t in thresholds:
        counts = confusion_counts(y_true, (proba >= t).astype(int))
        out.append(total_cost(counts, costs))
    return np.array(out)


def best_threshold(y_true, proba, costs):
    """Threshold with the lowest total cost (first one wins on ties)."""
    curve = cost_curve(y_true, proba, costs)
    idx = int(np.argmin(curve))
    return float(THRESHOLDS[idx]), float(curve[idx])


def evaluate_at(y_true, proba, threshold, costs):
    """Confusion matrix, precision/recall/F1 and cost at one threshold."""
    pred = (proba >= threshold).astype(int)
    counts = confusion_counts(y_true, pred)
    n = len(y_true)
    return {
        "threshold": float(threshold),
        "confusion": counts,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "flag_rate": float(pred.mean()),
        "total_cost": float(total_cost(counts, costs)),
        "cost_per_customer": float(total_cost(counts, costs) / n),
    }
