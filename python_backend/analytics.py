"""Analytics and Portfolio Evaluation Engine for CreditRisk-X.

Computes ROC-AUC, PR-AUC, Brier score, log loss, and portfolio risk distributions.
Only computes evaluation metrics when true labels and probabilities are provided.
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

import config
from model import get_scores


def evaluate_model_performance(
    y_true: Union[List[int], np.ndarray],
    y_proba: Union[List[float], np.ndarray],
) -> Dict[str, Any]:
    """
    Compute rigorous binary classification metrics given ground truth labels and predicted probabilities.
    Requires at least one positive and one negative sample.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain both positive (1) and negative (0) labels.")

    # ROC-AUC & PR-AUC
    roc_auc = float(roc_auc_score(y_true, y_proba))
    pr_auc = float(average_precision_score(y_true, y_proba))

    # Brier Score = Mean Squared Error of probability predictions
    brier = float(np.mean((y_proba - y_true) ** 2))

    # Log loss (binary cross-entropy) with probability clipping to avoid log(0)
    p_clipped = np.clip(y_proba, 1e-15, 1.0 - 1e-15)
    loss = float(log_loss(y_true, p_clipped))

    return {
        "sample_size": len(y_true),
        "default_rate": round(float(y_true.mean()), 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "log_loss": round(loss, 4),
    }


def analyze_portfolio(
    probabilities: Union[List[float], np.ndarray],
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Analyze portfolio-level risk metrics across a batch of loan applications.
    """
    if threshold is None:
        threshold = config.DEFAULT_THRESHOLD

    p = np.asarray(probabilities, dtype=float)
    n = max(len(p), 1)

    high_mask = p >= threshold
    medium_mask = (p >= 0.5 * threshold) & (~high_mask)
    low_mask = p < 0.5 * threshold

    high_count = int(high_mask.sum())
    medium_count = int(medium_mask.sum())
    low_count = int(low_mask.sum())

    return {
        "total_applications": len(p),
        "threshold_applied": round(float(threshold), 4),
        "average_default_probability": round(float(p.mean()), 4),
        "median_default_probability": round(float(np.median(p)), 4),
        "high_risk_count": high_count,
        "high_risk_percentage": round((high_count / n) * 100.0, 2),
        "medium_risk_count": medium_count,
        "medium_risk_percentage": round((medium_count / n) * 100.0, 2),
        "low_risk_count": low_count,
        "low_risk_percentage": round((low_count / n) * 100.0, 2),
    }


def get_stored_test_metrics() -> Dict[str, Any]:
    """Retrieve pre-computed evaluation metrics on the held-out test set."""
    scores = get_scores()
    y_test = scores["y_test"]
    metrics = {}
    for k in scores.keys():
        if k.startswith("test_"):
            m_name = k.replace("test_", "")
            metrics[m_name] = evaluate_model_performance(y_test, scores[k])
    return metrics
