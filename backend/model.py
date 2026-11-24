"""Model Loading and Artifact Management for CreditRisk-X.

Loads the trained XGBoost model from Kaggle (`xgboost.json`),
as well as companion benchmark models from local artifacts.
"""
import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from xgboost import XGBClassifier

import config

logger = logging.getLogger("creditrisk_x.model")

_MODELS: Optional[Dict[str, Any]] = None
_SCORES: Optional[Dict[str, Any]] = None
_META: Optional[Dict[str, Any]] = None
_KAGGLE_XGB: Optional[XGBClassifier] = None


def find_kaggle_model_path() -> str:
    """
    Locate the Kaggle xgboost.json model artifact:
    1. Check local backend/artifacts/xgboost.json
    2. Check local backend/xgboost.json
    3. Check local cache or download via kagglehub
    """
    candidates = [
        config.ARTIFACT_DIR / config.KAGGLE_MODEL_FILE,
        config.BASE_DIR / config.KAGGLE_MODEL_FILE,
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    # Try downloading via kagglehub
    try:
        import kagglehub
        logger.info("Downloading Kaggle model: %s", config.KAGGLE_HANDLE)
        folder = kagglehub.model_download(config.KAGGLE_HANDLE)
        found = glob.glob(os.path.join(folder, "**", config.KAGGLE_MODEL_FILE), recursive=True)
        if found:
            return found[0]
    except Exception as e:
        logger.warning("kagglehub download attempt failed: %s", e)

    raise FileNotFoundError(
        f"Could not locate {config.KAGGLE_MODEL_FILE} in artifacts or via Kaggle."
    )


def load_kaggle_xgboost_model() -> XGBClassifier:
    """Load the trained XGBoost model from Kaggle xgboost.json."""
    global _KAGGLE_XGB
    if _KAGGLE_XGB is not None:
        return _KAGGLE_XGB

    model_path = find_kaggle_model_path()
    logger.info("Loading Kaggle XGBoost model from: %s", model_path)
    xgb = XGBClassifier()
    xgb.load_model(model_path)
    _KAGGLE_XGB = xgb
    return _KAGGLE_XGB


def load_artifacts() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load trained models, evaluation scores, and metadata once and cache them."""
    global _MODELS, _SCORES, _META
    if _MODELS is not None:
        return _MODELS, _SCORES, _META

    # Always ensure the Kaggle XGBoost model is loaded as primary
    kaggle_model = load_kaggle_xgboost_model()

    models_path = config.ARTIFACT_DIR / "models.joblib"
    scores_path = config.ARTIFACT_DIR / "scores.npz"
    meta_path = config.ARTIFACT_DIR / "meta.json"

    if models_path.exists():
        _MODELS = joblib.load(models_path)
    else:
        _MODELS = {}

    # Primary XGBoost model is the one from Kaggle
    _MODELS["xgboost"] = kaggle_model

    if scores_path.exists():
        _SCORES = dict(np.load(scores_path))
    else:
        _SCORES = {}

    if meta_path.exists():
        _META = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        # Fallback feature list from Kaggle model
        booster_feats = kaggle_model.get_booster().feature_names
        _META = {"features": booster_feats}

    return _MODELS, _SCORES, _META


def is_model_available() -> bool:
    """Check if model artifacts exist and can be loaded."""
    try:
        load_artifacts()
        return True
    except Exception:
        return False


def get_model(name: str = "xgboost"):
    """Retrieve a specific trained model."""
    if name == "xgboost":
        return load_kaggle_xgboost_model()
    models, _, _ = load_artifacts()
    if name not in models:
        raise KeyError(
            f"Model '{name}' not found. Available models: {list(models.keys())}"
        )
    return models[name]


def get_all_models() -> Dict[str, Any]:
    """Retrieve all available trained models."""
    models, _, _ = load_artifacts()
    return models


def get_scores() -> Dict[str, Any]:
    """Retrieve validation and test evaluation scores."""
    _, scores, _ = load_artifacts()
    return scores


def get_meta() -> Dict[str, Any]:
    """Retrieve metadata (feature list, decision tree rules, importances, samples)."""
    _, _, meta = load_artifacts()
    return meta


def get_feature_manifest() -> List[str]:
    """Retrieve the exact list of 43 features expected by the trained model."""
    meta = get_meta()
    if "features" in meta:
        return list(meta["features"])
    model = get_model("xgboost")
    return list(model.get_booster().feature_names)
