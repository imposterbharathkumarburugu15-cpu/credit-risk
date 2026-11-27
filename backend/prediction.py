"""Prediction Module for CreditRisk-X.

Validates incoming features, ensures exact ordering according to the feature manifest,
computes engineered features when raw inputs are provided, and generates default probabilities.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

import config
from features import (
    ENGINEERED_COLUMNS,
    FEATURE_COLUMNS,
    RAW_COLUMNS,
    add_engineered_features,
    clean_categories,
)
from model import get_feature_manifest, get_model


class RawCustomerInput(BaseModel):
    """The 23 raw application features collected from the borrower."""
    LIMIT_BAL: float = Field(..., description="Credit limit in NT dollars")
    SEX: int = Field(..., ge=1, le=2, description="1=Male, 2=Female")
    EDUCATION: int = Field(..., ge=0, le=6, description="1=Grad, 2=Uni, 3=HighSchool, 4=Other")
    MARRIAGE: int = Field(..., ge=0, le=3, description="1=Married, 2=Single, 3=Other")
    AGE: int = Field(..., ge=18, le=100, description="Age in years")
    PAY_0: int = Field(..., ge=-2, le=8, description="Repayment status in Month 1 (last month)")
    PAY_2: int = Field(..., ge=-2, le=8, description="Repayment status in Month 2")
    PAY_3: int = Field(..., ge=-2, le=8, description="Repayment status in Month 3")
    PAY_4: int = Field(..., ge=-2, le=8, description="Repayment status in Month 4")
    PAY_5: int = Field(..., ge=-2, le=8, description="Repayment status in Month 5")
    PAY_6: int = Field(..., ge=-2, le=8, description="Repayment status in Month 6")
    BILL_AMT1: float = Field(..., description="Bill statement amount in Month 1")
    BILL_AMT2: float = Field(..., description="Bill statement amount in Month 2")
    BILL_AMT3: float = Field(..., description="Bill statement amount in Month 3")
    BILL_AMT4: float = Field(..., description="Bill statement amount in Month 4")
    BILL_AMT5: float = Field(..., description="Bill statement amount in Month 5")
    BILL_AMT6: float = Field(..., description="Bill statement amount in Month 6")
    PAY_AMT1: float = Field(..., ge=0, description="Amount paid in Month 1")
    PAY_AMT2: float = Field(..., ge=0, description="Amount paid in Month 2")
    PAY_AMT3: float = Field(..., ge=0, description="Amount paid in Month 3")
    PAY_AMT4: float = Field(..., ge=0, description="Amount paid in Month 4")
    PAY_AMT5: float = Field(..., ge=0, description="Amount paid in Month 5")
    PAY_AMT6: float = Field(..., ge=0, description="Amount paid in Month 6")


def validate_and_prepare_features(data: Union[Dict[str, Any], RawCustomerInput]) -> pd.DataFrame:
    """
    Validate incoming features and produce a single-row DataFrame
    with all 43 columns in the exact order required by the trained model.
    """
    if isinstance(data, BaseModel):
        data_dict = data.model_dump()
    else:
        data_dict = dict(data)

    # Check if all 43 features are already present
    manifest = get_feature_manifest()
    has_all_43 = all(feat in data_dict for feat in manifest)

    if has_all_43:
        # User supplied full feature set
        row_df = pd.DataFrame([{f: float(data_dict[f]) for f in manifest}])
        return row_df[manifest]

    # Otherwise verify raw 23 features are present
    missing_raw = [col for col in RAW_COLUMNS if col not in data_dict]
    if missing_raw:
        raise ValueError(
            f"Missing required raw features: {missing_raw}. Must provide either the 23 raw columns or all 43 features."
        )

    # Extract raw columns and clean categories (fold 0, 5, 6 -> 4 for education, 0 -> 3 for marriage)
    raw_df = pd.DataFrame([{col: float(data_dict[col]) for col in RAW_COLUMNS}])
    cleaned_df = clean_categories(raw_df)

    # Add 20 engineered features
    full_df = add_engineered_features(cleaned_df)
    return full_df[manifest]


def predict_probability(
    features: Union[Dict[str, Any], RawCustomerInput],
    model_name: str = "xgboost"
) -> Dict[str, Any]:
    """
    Generate default probability using the specified model (default: XGBoost).
    Returns probability and prepared feature matrix.
    """
    X = validate_and_prepare_features(features)
    model = get_model(model_name)

    # predict_proba returns [prob_non_default, prob_default]
    proba_default = float(model.predict_proba(X)[0, 1])
    proba_rounded = round(proba_default, 4)

    return {
        "default_probability": proba_rounded,
        "model_used": model_name,
        "model_version": config.MODEL_VERSION,
        "feature_count": len(X.columns),
        "features_df": X,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _impute_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Impute missing / null / empty-string values in a raw record using
    domain-safe defaults for each feature group.
    Returns (imputed_dict, list_of_imputed_column_names).
    """
    # Sensible median-ish defaults derived from the UCI credit dataset
    DEFAULTS: Dict[str, float] = {
        "LIMIT_BAL": 140000.0,  # median credit limit
        "SEX":        1.0,
        "EDUCATION":  2.0,       # university
        "MARRIAGE":   2.0,       # single
        "AGE":        35.0,
        "PAY_0":      0.0,       # on-time / revolving
        "PAY_2":      0.0,
        "PAY_3":      0.0,
        "PAY_4":      0.0,
        "PAY_5":      0.0,
        "PAY_6":      0.0,
        "BILL_AMT1":  0.0,
        "BILL_AMT2":  0.0,
        "BILL_AMT3":  0.0,
        "BILL_AMT4":  0.0,
        "BILL_AMT5":  0.0,
        "BILL_AMT6":  0.0,
        "PAY_AMT1":   0.0,
        "PAY_AMT2":   0.0,
        "PAY_AMT3":   0.0,
        "PAY_AMT4":   0.0,
        "PAY_AMT5":   0.0,
        "PAY_AMT6":   0.0,
    }
    out = dict(rec)
    imputed_cols = []
    for col in RAW_COLUMNS:
        v = rec.get(col)
        # Treat None, "", "null", "nan", "na", "n/a" as missing
        if v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "nan", "na", "n/a")):
            out[col] = DEFAULTS[col]
            imputed_cols.append(col)
        else:
            try:
                out[col] = float(v)
            except (TypeError, ValueError):
                out[col] = DEFAULTS[col]
                imputed_cols.append(col)
    return out, imputed_cols


def predict_batch_probabilities(
    records: List[Dict[str, Any]],
    model_name: str = "xgboost",
) -> Tuple[List[Optional[float]], List[str]]:
    """Vectorized batch probability prediction for up to 10,000+ rows.

    Null / missing feature values are imputed with domain-safe defaults rather
    than being skipped, so every uploaded row receives a prediction.
    Imputation details are surfaced via the ``imputed_count`` and ``errors``
    fields in the response.
    """
    manifest = get_feature_manifest()
    model_obj = get_model(model_name)
    if model_obj is None:
        raise ValueError(f"Model '{model_name}' not loaded.")

    probs: List[Optional[float]] = [None] * len(records)
    errors: List[str] = []
    valid_indices: List[int] = []
    raw_rows: List[Dict[str, float]] = []
    imputed_count: int = 0

    for idx, rec in enumerate(records):
        try:
            imputed_rec, imputed_cols = _impute_record(rec)
            if imputed_cols:
                imputed_count += 1
                if len(errors) < 10:
                    errors.append(
                        f"Row {idx + 1}: null values in [{', '.join(imputed_cols[:5])}] "
                        f"→ imputed with dataset defaults"
                    )
            raw_rows.append({col: imputed_rec[col] for col in RAW_COLUMNS})
            valid_indices.append(idx)
        except Exception as exc:
            if len(errors) < 10:
                errors.append(f"Row {idx + 1}: skipped — {exc}")

    if valid_indices:
        raw_df = pd.DataFrame(raw_rows)
        cleaned_df = clean_categories(raw_df)
        full_df = add_engineered_features(cleaned_df)[manifest]
        batch_probs = model_obj.predict_proba(full_df)[:, 1]
        for pos, orig_idx in enumerate(valid_indices):
            probs[orig_idx] = round(float(batch_probs[pos]), 4)

    return probs, errors, imputed_count
