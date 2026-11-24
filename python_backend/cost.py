"""Cost Efficiency Engine for CreditRisk-X.

Calculates false-positive cost, false-negative cost, total modeled classification cost,
and compares cost profiles across decision policies.
"""
from typing import Any, Dict, Optional

import config
from costs import (
    DEFAULT_COSTS,
    DEFAULT_DOLLAR_COSTS,
    THRESHOLDS,
    best_threshold,
    calculate_cost,
    compare_costs,
    confusion_counts,
    cost_curve,
    evaluate_at,
    total_cost,
)


def compute_classification_cost(
    fp_count: int,
    fn_count: int,
    fp_cost: Optional[float] = None,
    fn_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute total financial cost under specified or default project assumptions.
    Defaults: FP cost = $500, FN cost = $5,000.
    """
    if fp_cost is None:
        fp_cost = config.DEFAULT_FP_COST
    if fn_cost is None:
        fn_cost = config.DEFAULT_FN_COST

    return calculate_cost(
        fp_count=fp_count,
        fn_count=fn_count,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
    )


def compare_policy_costs(
    baseline_fp: int,
    baseline_fn: int,
    optimized_fp: int,
    optimized_fn: int,
    fp_cost: Optional[float] = None,
    fn_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compare baseline classification policy against optimized cost-sensitive policy.
    Returns dollar savings amount and percentage reduction.
    """
    if fp_cost is None:
        fp_cost = config.DEFAULT_FP_COST
    if fn_cost is None:
        fn_cost = config.DEFAULT_FN_COST

    return compare_costs(
        baseline_fp=baseline_fp,
        baseline_fn=baseline_fn,
        optimized_fp=optimized_fp,
        optimized_fn=optimized_fn,
        fp_cost=fp_cost,
        fn_cost=fn_cost,
    )
