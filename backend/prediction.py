"""Prediction Module for CreditRisk-X.

Validates incoming features, ensures exact ordering according to the feature manifest,
computes engineered features when raw inputs are provided, and generates default probabilities.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

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
