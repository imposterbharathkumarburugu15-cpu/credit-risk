"""
cost_engine.py  -  Cost Efficiency Engine for the bank customer monitoring app

What it does (and which real-world fault each part fixes):

  1. Exposure-based expected cost   -> costs depend on each customer's exposure, not a flat number
  2. Expected approval value        -> counts profit from good customers AND loss from defaulters
  3. Opportunity cost of an action  -> for existing customers the choice is an ACTION, not approve/reject
  4. Comparison with sensitivity    -> shows if the best decision survives different cost assumptions
  5. Recommendation + review zone   -> unstable, high-stakes decisions go to a human, with reason codes
  6. Calibrated break-even threshold-> probabilities are calibrated first; threshold comes from costs
  7. Decision support + monitoring  -> fairness report, backtest, versioned assumptions for the audit trail

Two modes:
  * SIMPLE mode   : fixed cost matrix + one global threshold (what the assignment asks for)
  * ADVANCED mode : per-customer costs + an action menu (what a real bank would use)

IMPORTANT: every default number below is ILLUSTRATIVE. Replace them with your bank's own
figures (loss rate, profit per customer, cost of a phone call...) before trusting the output.
The engine only needs numpy and pandas. scikit-learn is used for calibration and PR curves,
and xgboost for reason codes (both are imported only when those functions are called).
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd

# Columns that credit laws in many countries restrict. We do not hide them: if the model
# really uses them, the reason codes must say so, and the fairness report must be checked.
PROTECTED_ATTRIBUTES = {"SEX", "MARRIAGE", "EDUCATION", "AGE"}


# =====================================================================================
# 1. Assumptions (versioned, so every decision can be traced back to the numbers used)
# =====================================================================================

@dataclass(frozen=True)
class Action:
    """One thing the bank can do about a customer."""
    name: str
    label: str
    headroom_factor: float   # share of the unused limit still available (1 = unchanged, 0 = frozen)
    pd_multiplier: float     # effect on default probability (0.8 = 20% lower)
    profit_retained: float   # share of normal profit kept (cutting a limit earns less)
    churn_prob: float        # chance the customer leaves because of this action
    fixed_cost: float        # cost of carrying it out (staff time, letters, calls)


# ILLUSTRATIVE numbers. How well each action really works must come from your bank's own data.
DEFAULT_ACTIONS = (
    Action("keep",           "Keep as is",          1.0, 1.00, 1.00, 0.00,   0.0),
    Action("early_outreach", "Early outreach call", 1.0, 0.90, 0.97, 0.01,  30.0),
    Action("payment_plan",   "Offer payment plan",  0.5, 0.75, 0.80, 0.05, 100.0),
    Action("reduce_limit",   "Reduce credit limit", 0.4, 0.85, 0.75, 0.08,  10.0),
    Action("block",          "Block new spending",  0.0, 0.95, 0.15, 0.25,  10.0),
)


@dataclass(frozen=True)
class CostAssumptions:
    version: str = "v1"

    # --- advanced mode ---
    lgd: float = 0.45                  # loss given default: share of the exposure that is never recovered
    ccf: float = 0.30                  # share of the unused limit a customer draws before defaulting
    annual_profit_rate: float = 0.15   # yearly profit as a share of the customer's usual balance
    profit_horizon_months: int = 12    # how long the profit of a good customer is counted
    min_balance_share: float = 0.10    # usual balance is at least this share of the limit

    # --- simple mode (explicit cost matrix, same currency as the data) ---
    cost_missed_default: float = 60000.0   # approved a customer who then defaulted
    cost_false_alarm: float = 5000.0       # rejected a customer who would have paid

    # --- sensitivity analysis and review zone ---
    lgd_scales: tuple = (0.8, 1.0, 1.2)
    profit_scales: tuple = (0.8, 1.0, 1.2)
    pd_scales: tuple = (0.85, 1.0, 1.15)   # uncertainty in the probability itself
    review_robustness: float = 0.70        # below this share of agreeing scenarios -> unstable
    review_min_regret: float = 50.0        # and the cost of being wrong is at least this much

    def with_changes(self, **changes) -> "CostAssumptions":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def fingerprint(self) -> str:
        """Short hash of all the numbers: same numbers = same fingerprint."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:12]


# =====================================================================================
# 2. Calibration: make the model's scores behave like real probabilities
# =====================================================================================

def _clip(p, lo=1e-4, hi=1 - 1e-4):
    return np.clip(np.atleast_1d(np.asarray(p, dtype=float)), lo, hi)


class ProbabilityCalibrator:
    """
    Fit on data the model was NOT trained on (a validation set), then use transform().
      method = "isotonic" (flexible, needs a few thousand rows) or "platt" (smooth, 2 parameters)
    """

    def __init__(self, method: str = "isotonic"):
        if method not in ("isotonic", "platt"):
            raise ValueError("method must be 'isotonic' or 'platt'")
        self.method = method
        self._model = None

    def fit(self, raw_probability, y_true):
        p, y = _clip(raw_probability), np.asarray(y_true).astype(int)
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            self._model = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p, y)
        else:
            from sklearn.linear_model import LogisticRegression
            self._model = LogisticRegression(C=1e6).fit(np.log(p / (1 - p)).reshape(-1, 1), y)
        return self

    def transform(self, raw_probability):
        if self._model is None:
            raise RuntimeError("Call fit() first")
        p = _clip(raw_probability)
        if self.method == "isotonic":
            out = self._model.predict(p)
        else:
            out = self._model.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]
        return _clip(out, 0.001, 0.999)


def adjust_for_base_rate(probability, train_default_rate: float, target_default_rate: float):
    """
    Prior-shift correction. Example: the UCI data has ~22% defaults, your real portfolio has 3%.
    Use this so the probabilities reflect the portfolio the engine will actually run on.
    """
    p = _clip(probability)
    a = target_default_rate / train_default_rate
    b = (1 - target_default_rate) / (1 - train_default_rate)
    return p * a / (p * a + (1 - p) * b)


def reliability_curve(y_true, probability, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed default rate per bin. A calibrated model hugs the diagonal."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(probability)})
    df["bin"] = pd.qcut(df["p"], q=bins, duplicates="drop")
    return (df.groupby("bin", observed=True)
              .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
              .reset_index(drop=True))


def expected_calibration_error(y_true, probability, bins: int = 10) -> float:
    r = reliability_curve(y_true, probability, bins)
    return float((r["n"] * (r["mean_predicted"] - r["observed_rate"]).abs()).sum() / r["n"].sum())


# =====================================================================================
# 3. Core economics (vectorised: works for one customer or all 30,000 at once)
# =====================================================================================

def exposure_and_profit(limit_bal, latest_bill, avg_bill, a: CostAssumptions):
    """
    balance  = what the customer owes now
    headroom = unused credit limit
    profit   = profit the bank expects from this customer if they do NOT default
    """
    limit, bill, avg = (np.atleast_1d(np.asarray(x, dtype=float)) for x in (limit_bal, latest_bill, avg_bill))
    balance = np.maximum(bill, 0.0)
    headroom = np.maximum(limit - balance, 0.0)
    usual_balance = np.maximum(np.maximum(avg, 0.0), a.min_balance_share * limit)
    profit = a.annual_profit_rate * usual_balance * a.profit_horizon_months / 12.0
    return balance, headroom, profit


def action_value(prob, balance, headroom, profit, a: CostAssumptions, action: Action):
    """
    Expected net value (profit minus loss minus cost of the action) of one action.
        value = (1 - PD') x profit'  -  PD' x LGD x EAD'  -  fixed cost
    where PD', profit' and EAD' are changed by the action.
    """
    pd_a = np.clip(prob * action.pd_multiplier, 0.0, 1.0)
    ead = balance + a.ccf * action.headroom_factor * headroom
    profit_a = profit * action.profit_retained * (1.0 - action.churn_prob)
    return (1.0 - pd_a) * profit_a - pd_a * a.lgd * ead - action.fixed_cost


def binary_costs(prob, balance, headroom, profit, a: CostAssumptions):
    """
    The classic approve/reject comparison, per customer.
      approve cost = PD x L                 (expected loss if we approve)
      reject cost  = (1 - PD) x B           (expected profit we give up if we reject)
      break-even PD = B / (B + L)           (approve below it, reject above it)
    """
    loss_if_default = a.lgd * (balance + a.ccf * headroom)
    denom = profit + loss_if_default
    breakeven = np.divide(profit, denom, out=np.zeros_like(denom), where=denom > 0)
    return prob * loss_if_default, (1.0 - prob) * profit, breakeven, loss_if_default


def breakeven_lgd(prob, balance, headroom, profit, a: CostAssumptions):
    """The loss rate at which approving and rejecting cost the same (sensitivity helper)."""
    exposure = balance + a.ccf * headroom
    return np.divide((1 - prob) * profit, prob * exposure,
                     out=np.full_like(exposure, np.inf), where=(prob * exposure) > 0)


# =====================================================================================
# 4. Evaluate a whole portfolio: best action, sensitivity, review zone
# =====================================================================================

REQUIRED_COLUMNS = ["default_probability", "limit_bal", "latest_bill", "avg_bill"]


def evaluate_portfolio(df: pd.DataFrame,
                       assumptions: Optional[CostAssumptions] = None,
                       actions=DEFAULT_ACTIONS) -> pd.DataFrame:
    """
    df needs: default_probability (CALIBRATED), limit_bal, latest_bill (BILL_AMT1), avg_bill
    (average of BILL_AMT1..6). An optional customer_id column is carried through.
    """
    a = assumptions or CostAssumptions()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    prob = df["default_probability"].to_numpy(dtype=float)
    balance, headroom, profit = exposure_and_profit(df["limit_bal"], df["latest_bill"], df["avg_bill"], a)
    names = [act.name for act in actions]

    values = np.column_stack([action_value(prob, balance, headroom, profit, a, act) for act in actions])
    best = values.argmax(axis=1)
    order = np.argsort(values, axis=1)
    runner_up = order[:, -2]
    rows = np.arange(len(df))
    best_value = values[rows, best]
    margin = best_value - values[rows, runner_up]

    # --- sensitivity: does the same action stay best when the assumptions are a bit off? ---
    agree = np.zeros(len(df))
    worst_regret = np.zeros(len(df))
    scenarios = list(itertools.product(a.lgd_scales, a.profit_scales, a.pd_scales))
    for lgd_s, profit_s, pd_s in scenarios:
        a_s = replace(a, lgd=a.lgd * lgd_s)
        p_s = np.clip(prob * pd_s, 0.0, 1.0)
        v_s = np.column_stack([action_value(p_s, balance, headroom, profit * profit_s, a_s, act)
                               for act in actions])
        agree += (v_s.argmax(axis=1) == best)
        worst_regret = np.maximum(worst_regret, v_s.max(axis=1) - v_s[rows, best])
    robustness = agree / len(scenarios)

    # review zone: unstable AND being wrong would be expensive
    needs_review = (robustness < a.review_robustness) & (worst_regret >= a.review_min_regret)

    approve_cost, reject_cost, breakeven, loss_if_default = binary_costs(prob, balance, headroom, profit, a)

    out = pd.DataFrame(index=df.index)
    if "customer_id" in df.columns:
        out["customer_id"] = df["customer_id"]
    out["default_probability"] = prob
    out["exposure_at_default"] = balance + a.ccf * headroom
    out["profit_if_good"] = profit
    out["approve_cost"] = approve_cost
    out["reject_cost"] = reject_cost
    out["breakeven_pd"] = breakeven
    out["binary_decision"] = np.where(prob < breakeven, "approve", "reject")
    out["risk_band"] = np.select([prob < 0.5 * breakeven, prob < breakeven], ["low", "medium"], "high")
    out["best_action"] = [names[i] for i in best]
    out["runner_up_action"] = [names[i] for i in runner_up]
    out["best_value"] = best_value
    out["margin"] = margin
    out["robustness"] = robustness
    out["worst_case_regret"] = worst_regret
    out["needs_review"] = needs_review
    out["expected_loss_if_kept"] = approve_cost
    for j, name in enumerate(names):
        out[f"value_{name}"] = values[:, j]
        # "cost of choosing this action" = how much worse it is than the best one
        out[f"regret_{name}"] = best_value - values[:, j]
    return out


def evaluate_customer(prob, limit_bal, latest_bill, avg_bill,
                      assumptions: Optional[CostAssumptions] = None, actions=DEFAULT_ACTIONS) -> dict:
    """One customer in, one dictionary out (handy for an API endpoint)."""
    df = pd.DataFrame({"default_probability": [prob], "limit_bal": [limit_bal],
                       "latest_bill": [latest_bill], "avg_bill": [avg_bill]})
    row = evaluate_portfolio(df, assumptions, actions).iloc[0].to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}


def portfolio_summary(result: pd.DataFrame, fixed_threshold: float = 0.5) -> dict:
    """
    Totals for the dashboard, including the headline 'savings versus baseline' numbers.
    These are EXPECTED values (they assume calibrated probabilities and that actions work as
    assumed). For realised numbers use backtest_binary() on held-out customers.
    """
    s = {
        "customers": int(len(result)),
        "expected_value_engine": float(result["best_value"].sum()),
        "needs_review": int(result["needs_review"].sum()),
        "median_breakeven_pd": float(result["breakeven_pd"].median()),
        "share_not_kept_as_is": float((result["best_action"] != "keep").mean()),
        "expected_loss_if_all_kept": float(result["expected_loss_if_kept"].sum()),
        "action_counts": result["best_action"].value_counts().to_dict(),
        "risk_band_counts": result["risk_band"].value_counts().to_dict(),
    }
    if "value_keep" in result:
        s["expected_value_keep_all"] = float(result["value_keep"].sum())
        s["savings_vs_keep_all"] = s["expected_value_engine"] - s["expected_value_keep_all"]
        if "value_block" in result:
            fixed = np.where(result["default_probability"] >= fixed_threshold,
                             result["value_block"], result["value_keep"]).sum()
            s["expected_value_fixed_threshold"] = float(fixed)
            s["savings_vs_fixed_threshold"] = s["expected_value_engine"] - s["expected_value_fixed_threshold"]
    return s


def what_if(df: pd.DataFrame, base: Optional[CostAssumptions] = None, **changes) -> dict:
    """What-if simulator: what happens to the portfolio if lgd, profit rate, etc. change?"""
    base = base or CostAssumptions()
    new = base.with_changes(**changes)
    s0 = portfolio_summary(evaluate_portfolio(df, base))
    s1 = portfolio_summary(evaluate_portfolio(df, new))
    return {"changed": changes, "base": s0, "scenario": s1,
            "median_breakeven_change": s1["median_breakeven_pd"] - s0["median_breakeven_pd"],
            "customers_needing_review_change": s1["needs_review"] - s0["needs_review"]}


# =====================================================================================
# 5. Simple mode: fixed cost matrix, threshold-versus-cost, precision-recall, backtest
# =====================================================================================

def matrix_threshold(cost_missed_default: float, cost_false_alarm: float) -> float:
    """Break-even threshold of a fixed cost matrix:  false_alarm / (false_alarm + missed_default)."""
    return cost_false_alarm / (cost_false_alarm + cost_missed_default)


def threshold_cost_curve(y_true, probability, cost_missed_default: float, cost_false_alarm: float,
                         thresholds=None) -> pd.DataFrame:
    """
    For every threshold: confusion matrix, precision, recall and total business cost.
    'Positive' means flagged as likely to default (rejected / restricted).
      missed default (FN) = approved a customer who defaulted
      false alarm    (FP) = rejected a customer who would have paid
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probability, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in thresholds:
        flag = p >= t
        tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
        fn = int((~flag & (y == 1)).sum()); tn = int((~flag & (y == 0)).sum())
        rows.append({
            "threshold": float(t), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / (tp + fn) if tp + fn else np.nan,
            "total_cost": fn * cost_missed_default + fp * cost_false_alarm,
        })
    return pd.DataFrame(rows)


def best_threshold(curve: pd.DataFrame) -> tuple:
    """(threshold, total_cost) with the lowest total cost on the curve."""
    row = curve.loc[curve["total_cost"].idxmin()]
    return float(row["threshold"]), float(row["total_cost"])


def precision_recall_data(y_true, probability) -> tuple:
    """Data for the precision-recall chart, plus the average precision score."""
    from sklearn.metrics import average_precision_score, precision_recall_curve
    prec, rec, thr = precision_recall_curve(np.asarray(y_true).astype(int), np.asarray(probability))
    return (pd.DataFrame({"precision": prec[:-1], "recall": rec[:-1], "threshold": thr}),
            float(average_precision_score(y_true, probability)))


def backtest_binary(y_true, probability, threshold: float, cost_missed_default: float,
                    cost_false_alarm: float, baseline_threshold: float = 0.5) -> dict:
    """
    REALISED cost on customers whose outcome is known. Use customers the model was NOT
    trained on, otherwise the result is too optimistic.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probability, dtype=float)

    def cost(flag):
        return float((~flag & (y == 1)).sum() * cost_missed_default + (flag & (y == 0)).sum() * cost_false_alarm)

    engine = cost(p >= threshold)
    approve_all = cost(np.zeros(len(y), dtype=bool))
    reject_all = cost(np.ones(len(y), dtype=bool))
    baseline = cost(p >= baseline_threshold)
    return {
        "engine_cost": engine, "approve_all_cost": approve_all, "reject_all_cost": reject_all,
        "fixed_threshold_cost": baseline,
        "saving_vs_approve_all": approve_all - engine,
        "saving_vs_fixed_threshold": baseline - engine,
    }


# =====================================================================================
# 6. Reason codes (why was this customer flagged?) - uses XGBoost's built-in contributions
# =====================================================================================

FRIENDLY_NAMES = {
    "PAY_0": "Payment delay last month", "PAY_2": "Payment delay 2 months ago",
    "PAY_3": "Payment delay 3 months ago", "PAY_4": "Payment delay 4 months ago",
    "PAY_5": "Payment delay 5 months ago", "PAY_6": "Payment delay 6 months ago",
    "LIMIT_BAL": "Credit limit", "BILL_AMT1": "Latest bill amount", "PAY_AMT1": "Latest payment amount",
    "utilization_latest": "Share of limit currently used", "utilization_mean_6m": "Average limit usage",
    "late_month_count_6m": "Number of late months", "severe_late_month_count_6m": "Number of seriously late months",
    "recent_late_month_count_3m": "Late months in the last 3", "max_delay_severity_6m": "Worst payment delay",
    "consecutive_recent_late_months": "Consecutive late months", "payment_active_months_6m": "Months with a payment",
    "total_payment_6m": "Total paid in 6 months", "bill_trend_slope_6m": "Bill trend",
    "payment_trend_slope_6m": "Payment trend",
}


def reason_codes(model, X: pd.DataFrame, top_n: int = 3) -> list:
    """
    Top factors that RAISE each customer's risk. X must have the model's 43 feature columns.
    The flag 'uses_protected_attribute' tells you when sex, marital status, education or age
    is among the top reasons - an alarm for the fairness review, not something to hide.
    """
    import xgboost as xgb
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]
    names, values = list(X.columns), X.to_numpy(dtype=float)
    out = []
    for i in range(len(X)):
        top = [j for j in np.argsort(contrib[i])[::-1][:top_n] if contrib[i, j] > 0]
        out.append({
            "reasons": [{"feature": names[j], "label": FRIENDLY_NAMES.get(names[j], names[j]),
                         "value": float(values[i, j]), "impact": float(contrib[i, j])} for j in top],
            "uses_protected_attribute": any(names[j] in PROTECTED_ATTRIBUTES for j in top),
        })
    return out


# =====================================================================================
# 7. Fairness report
# =====================================================================================

def age_band(age) -> pd.Series:
    return pd.cut(pd.Series(age), bins=[0, 25, 35, 45, 55, 200],
                  labels=["25 or under", "26-35", "36-45", "46-55", "56+"])


def fairness_report(df: pd.DataFrame, group_col: str, restricted_col: str = "restricted",
                    defaulted_col: Optional[str] = None, min_ratio: float = 0.8) -> pd.DataFrame:
    """
    Compare groups (for example SEX, age_band, EDUCATION). df needs a boolean column
    'restricted' (e.g. best_action != 'keep') and optionally the real outcome 'defaulted'.
    impact_ratio < 0.8 is a common SCREENING flag, not a legal verdict: use it to start a review.
    """
    rows = []
    for group, sub in df.groupby(group_col, observed=True):
        r = {"group": group, "n": len(sub), "restriction_rate": float(sub[restricted_col].mean())}
        if "default_probability" in sub:
            r["avg_default_probability"] = float(sub["default_probability"].mean())
        if defaulted_col:
            good, bad = sub[sub[defaulted_col] == 0], sub[sub[defaulted_col] == 1]
            r["false_restriction_rate"] = float(good[restricted_col].mean()) if len(good) else np.nan
            r["missed_default_rate"] = float(1 - bad[restricted_col].mean()) if len(bad) else np.nan
        rows.append(r)
    rep = pd.DataFrame(rows)
    approval = 1 - rep["restriction_rate"]
    rep["impact_ratio"] = approval / approval.max()
    rep["review_flag"] = rep["impact_ratio"] < min_ratio
    return rep


# =====================================================================================
# 8. Audit trail: one record per recommendation (insert into the `decisions` table)
# =====================================================================================

def build_decision_record(customer_id: int, row, assumptions: CostAssumptions,
                          model_version: str, reasons: Optional[dict] = None) -> dict:
    """Everything needed to explain this decision months later."""
    row = dict(row)
    return {
        "customer_id": int(customer_id),
        "model_version": model_version,
        "assumptions_version": assumptions.version,
        "assumptions_fingerprint": assumptions.fingerprint(),
        "default_probability": round(float(row["default_probability"]), 6),
        "breakeven_pd": round(float(row["breakeven_pd"]), 6),
        "binary_decision": row["binary_decision"],
        "recommended_action": row["best_action"],
        "expected_value": round(float(row["best_value"]), 2),
        "robustness": round(float(row["robustness"]), 3),
        "needs_review": bool(row["needs_review"]),
        "reasons_json": json.dumps(reasons) if reasons else None,
    }


# =====================================================================================
# Demo with made-up data:  python cost_engine.py
# =====================================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 5000
    true_pd = rng.beta(1.2, 4.0, n)
    y = rng.binomial(1, true_pd)
    raw_pd = np.clip(true_pd ** 0.6, 0.001, 0.999)          # a deliberately miscalibrated "model"

    half = n // 2                                           # fit calibrator on one half, use the other
    cal = ProbabilityCalibrator("isotonic").fit(raw_pd[:half], y[:half])
    pd_cal = cal.transform(raw_pd[half:])
    y_test = y[half:]
    print(f"Calibration error  raw: {expected_calibration_error(y_test, raw_pd[half:]):.3f}"
          f"   calibrated: {expected_calibration_error(y_test, pd_cal):.3f}")

    limit = rng.lognormal(11.5, 0.7, n - half)
    bill = limit * rng.uniform(0, 1, n - half)
    data = pd.DataFrame({"customer_id": np.arange(n - half), "default_probability": pd_cal,
                         "limit_bal": limit, "latest_bill": bill, "avg_bill": bill * 0.9,
                         "SEX": rng.integers(1, 3, n - half), "defaulted": y_test})

    assumptions = CostAssumptions(version="demo-v1")
    result = evaluate_portfolio(data, assumptions)
    summary = portfolio_summary(result)
    print("\nBest actions:", summary["action_counts"])
    print(f"Need human review: {summary['needs_review']}   median break-even PD: {summary['median_breakeven_pd']:.1%}")
    print(f"Expected value  engine: {summary['expected_value_engine']:,.0f}   "
          f"keep everyone: {summary['expected_value_keep_all']:,.0f}   "
          f"fixed 0.5 rule: {summary['expected_value_fixed_threshold']:,.0f}")

    one = evaluate_customer(0.18, limit_bal=100000, latest_bill=60000, avg_bill=55000, assumptions=assumptions)
    print(f"\nOne customer -> action: {one['best_action']}, break-even PD: {one['breakeven_pd']:.1%}, "
          f"robustness: {one['robustness']:.2f}")

    t_simple = matrix_threshold(assumptions.cost_missed_default, assumptions.cost_false_alarm)
    curve = threshold_cost_curve(y_test, pd_cal, assumptions.cost_missed_default, assumptions.cost_false_alarm)
    t_best, c_best = best_threshold(curve)
    print(f"\nSimple mode: formula threshold {t_simple:.3f}, lowest-cost threshold on data {t_best:.2f}")
    print("Backtest:", {k: round(v) for k, v in backtest_binary(
        y_test, pd_cal, t_best, assumptions.cost_missed_default, assumptions.cost_false_alarm).items()})

    result["restricted"] = result["best_action"] != "keep"
    result["SEX"], result["defaulted"] = data["SEX"], data["defaulted"]
    print("\nFairness check on SEX:\n", fairness_report(result, "SEX", defaulted_col="defaulted").round(3).to_string(index=False))
