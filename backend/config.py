"""Configuration settings for CreditRisk-X Python Backend."""
import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DATA_DIR = ROOT_DIR / "data"


def _load_dotenv() -> None:
    """Load key=value pairs from .env in repo root or backend dir if present."""
    for env_path in (ROOT_DIR / ".env", BASE_DIR / ".env"):
        if not env_path.is_file():
            continue
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            pass


_load_dotenv()

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
MYSQL_SSL = os.getenv("MYSQL_SSL", "false").strip().lower() in ("1", "true", "yes")

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

# Email / SMTP Configuration (never hardcode credentials)
EMAIL_USER = os.getenv("EMAIL_USER", "").strip()
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "").strip()
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "CreditRisk-X").strip()
EMPLOYEE_REVIEW_EMAIL = os.getenv("EMPLOYEE_REVIEW_EMAIL", "").strip() or EMAIL_USER

# Authentication & Employee Authorization
AUTH_SECRET = os.getenv("AUTH_SECRET", "") or secrets.token_hex(32)
EMPLOYEE_API_KEY = os.getenv("EMPLOYEE_API_KEY", "").strip()
