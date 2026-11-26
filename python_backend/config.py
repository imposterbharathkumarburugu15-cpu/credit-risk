"""Configuration settings for CreditRisk-X Python Backend."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DATA_DIR = ROOT_DIR / "data"

# Server configuration
PORT = int(os.getenv("PYTHON_PORT", "8000"))
HOST = os.getenv("PYTHON_HOST", "0.0.0.0")

# Database configuration (MySQL)
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
DATABASE_NAME = os.getenv("MYSQL_DATABASE", "creditrisk_x")
MYSQL_DATABASE = DATABASE_NAME

# Kaggle Model handle and filename
KAGGLE_HANDLE = "manivardhangundoju/credit-default-risk-prediction-model-xgboost/scikitLearn/default/1"
KAGGLE_MODEL_FILE = "xgboost.json"

# Risk & Decision Thresholds
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.35"))

# Cost Matrix Assumptions (in USD)
DEFAULT_FP_COST = float(os.getenv("FP_COST", "500.0"))
DEFAULT_FN_COST = float(os.getenv("FN_COST", "5000.0"))
DEFAULT_TP_COST = float(os.getenv("TP_COST", "0.0"))
DEFAULT_TN_COST = float(os.getenv("TN_COST", "0.0"))

# Model Version
MODEL_VERSION = "kaggle_xgboost_v1.0"
PRIMARY_MODEL_NAME = "xgboost"
