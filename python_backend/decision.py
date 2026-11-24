"""Decision Engine for CreditRisk-X.

Applies the cost-sensitive risk threshold to separate borrowers into risk tiers
(LOW, MEDIUM, HIGH) and generates operational recommendations without automatically
declining loans or overriding credit policy.
"""
from typing import Any, Dict, Optional

import config


def make_decision(
    default_probability: float,
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Evaluate predicted default probability against policy threshold.

    Banding Policy:
      - probability >= threshold       -> HIGH risk (flag for manual review or decline)
      - threshold/2 <= probability < t -> MEDIUM risk (approve with reduced limit & monitoring)
      - probability < threshold/2      -> LOW risk (standard approval)
    """
    if threshold is None:
        threshold = config.DEFAULT_THRESHOLD

    p = float(default_probability)
    t = float(threshold)

    if p >= t:
        risk_classification = "HIGH"
        suggested_action = "Flag: decline or send to manual review"
        review_recommendation = (
            "Requires senior underwriter manual inspection and additional income verification before decision."
        )
        flagged = True
    elif p >= 0.5 * t:
        risk_classification = "MEDIUM"
        suggested_action = "Approve with a lower limit and monitoring"
        review_recommendation = (
            "Approve with a reduced credit limit, mandatory automatic payments, and 60-day utilization monitoring."
        )
        flagged = False
    else:
        risk_classification = "LOW"
        suggested_action = "Approve"
        review_recommendation = "Standard automated approval on standard credit terms."
        flagged = False

    return {
        "default_probability": round(p, 4),
        "threshold": round(t, 4),
        "risk_classification": risk_classification,
        "suggested_action": suggested_action,
        "review_recommendation": review_recommendation,
        "flagged": flagged,
    }
