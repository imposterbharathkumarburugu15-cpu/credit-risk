"""Feature helpers shared by the training script and the API.

The CSV already ships with engineered columns, but the dashboard form only
collects the 23 raw fields. So we rebuild the engineered columns from raw
values here, and check in train.py that the result matches the CSV.
"""
import numpy as np
import pandas as pd

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

RAW_COLUMNS = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + PAY_COLS
    + BILL_COLS
    + PAY_AMT_COLS
)

ENGINEERED_COLUMNS = [
    "utilization_latest", "utilization_mean_6m", "utilization_max_6m",
    "utilization_std_6m", "utilization_trend_6m", "avg_bill_6m", "max_bill_6m",
    "bill_volatility_6m", "bill_trend_slope_6m", "total_payment_6m",
    "avg_payment_6m", "payment_volatility_6m", "payment_trend_slope_6m",
    "payment_active_months_6m", "late_month_count_6m",
    "severe_late_month_count_6m", "recent_late_month_count_3m",
    "max_delay_severity_6m", "avg_delay_severity_6m",
    "consecutive_recent_late_months",
]

FEATURE_COLUMNS = RAW_COLUMNS + ENGINEERED_COLUMNS
TARGET = "default.payment.next.month"


def clean_categories(df):
    """Fold the undocumented category codes into 'others'."""
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def _slope(values):
    """Least-squares slope per row. Columns must be in time order (old -> new)."""
    x = np.arange(values.shape[1], dtype=float)
    x = x - x.mean()
    centered = values - values.mean(axis=1, keepdims=True)
    return (centered * x).sum(axis=1) / (x ** 2).sum()


def add_engineered_features(raw):
    """Take a frame with the 23 raw columns, return it with the 20 derived ones."""
    df = raw[RAW_COLUMNS].astype(float).copy()

    # BILL_AMT1 / PAY_0 are the latest month, so flip them for trend calculations
    bills = df[BILL_COLS].to_numpy()
    payments = df[PAY_AMT_COLS].to_numpy()
    status = df[PAY_COLS].to_numpy()
    limit = df["LIMIT_BAL"].to_numpy()[:, None]

    util = bills / limit
    delay = np.clip(status, 0, None)      # on-time / revolving counts as 0 delay
    late = status > 0

    df["utilization_latest"] = util[:, 0]
    df["utilization_mean_6m"] = util.mean(axis=1)
    df["utilization_max_6m"] = util.max(axis=1)
    df["utilization_std_6m"] = util.std(axis=1)
    df["utilization_trend_6m"] = _slope(util[:, ::-1])

    df["avg_bill_6m"] = bills.mean(axis=1)
    df["max_bill_6m"] = bills.max(axis=1)
    df["bill_volatility_6m"] = bills.std(axis=1)
    df["bill_trend_slope_6m"] = _slope(bills[:, ::-1])

    df["total_payment_6m"] = payments.sum(axis=1)
    df["avg_payment_6m"] = payments.mean(axis=1)
    df["payment_volatility_6m"] = payments.std(axis=1)
    df["payment_trend_slope_6m"] = _slope(payments[:, ::-1])
    df["payment_active_months_6m"] = (payments > 0).sum(axis=1)

    df["late_month_count_6m"] = late.sum(axis=1)
    df["severe_late_month_count_6m"] = (status >= 2).sum(axis=1)
    df["recent_late_month_count_3m"] = late[:, :3].sum(axis=1)
    df["max_delay_severity_6m"] = delay.max(axis=1)
    df["avg_delay_severity_6m"] = delay.mean(axis=1)

    # how many months in a row, counting back from the latest, were late
    streak = np.zeros(len(df))
    still_late = np.ones(len(df), dtype=bool)
    for m in range(6):
        still_late &= late[:, m]
        streak += still_late
    df["consecutive_recent_late_months"] = streak

    return df[FEATURE_COLUMNS]
