"""MySQL database integration for CreditRisk-X."""
import atexit
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pymysql
from pymysql.cursors import DictCursor

import config

logger = logging.getLogger("creditrisk_x.database")

_client: Optional[pymysql.connections.Connection] = None
_schema_initialized: bool = False


def _ensure_schema(conn: pymysql.connections.Connection) -> None:
    """Create the target database and tables if they do not exist."""
    global _schema_initialized
    if _schema_initialized:
        return

    db_name = config.DATABASE_NAME.replace("`", "")
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
    conn.select_db(db_name)

    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                model_version VARCHAR(128) NULL,
                default_probability DOUBLE NULL,
                band VARCHAR(32) NULL,
                decision VARCHAR(128) NULL,
                payload JSON NOT NULL,
                created_at DATETIME(6) NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR(64) PRIMARY KEY,
                name VARCHAR(120) NOT NULL,
                email VARCHAR(255) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                role VARCHAR(32) NOT NULL DEFAULT 'customer',
                created_at DATETIME(6) NOT NULL
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS applications (
                application_id VARCHAR(64) PRIMARY KEY,
                customer_email VARCHAR(255) NOT NULL,
                customer_name VARCHAR(120) NOT NULL,
                status VARCHAR(64) NOT NULL,
                risk_classification VARCHAR(32) NOT NULL,
                flagged TINYINT(1) NOT NULL DEFAULT 0,
                default_probability DOUBLE NULL,
                missing_documents JSON NOT NULL,
                final_decision VARCHAR(64) NULL,
                notes TEXT NULL,
                created_at DATETIME(6) NOT NULL,
                updated_at DATETIME(6) NOT NULL,
                INDEX idx_applications_email (customer_email)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS email_logs (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                idempotency_key VARCHAR(255) NULL,
                success TINYINT(1) NOT NULL DEFAULT 0,
                payload JSON NOT NULL,
                created_at DATETIME(6) NOT NULL,
                INDEX idx_email_logs_idem (idempotency_key, success)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
    _schema_initialized = True


def close_client() -> None:
    """Close the cached MySQL connection on process exit."""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


atexit.register(close_client)


def get_client() -> Optional[pymysql.connections.Connection]:
    """Retrieve or initialize the MySQL connection gracefully."""
    global _client
    try:
        if _client is not None:
            _client.ping(reconnect=True)
            _ensure_schema(_client)
            return _client
    except Exception as e:
        logger.warning("Existing MySQL connection lost (%s); reconnecting.", e)
        close_client()

    try:
        connect_kwargs: Dict[str, Any] = {
            "host": config.MYSQL_HOST,
            "port": config.MYSQL_PORT,
            "user": config.MYSQL_USER,
            "password": config.MYSQL_PASSWORD,
            "connect_timeout": 4,
            "read_timeout": 10,
            "write_timeout": 10,
            "autocommit": True,
            "charset": "utf8mb4",
            "cursorclass": DictCursor,
        }
        if getattr(config, "MYSQL_SSL", False):
            connect_kwargs["ssl"] = {"ssl": {}}
        _client = pymysql.connect(**connect_kwargs)
        _ensure_schema(_client)
        logger.info(
            "Successfully connected to MySQL at %s:%s/%s",
            config.MYSQL_HOST,
            config.MYSQL_PORT,
            config.DATABASE_NAME,
        )
    except Exception as e:
        logger.warning("MySQL connection could not be established: %s", e)
        _client = None
    return _client


def get_db() -> Optional[pymysql.connections.Connection]:
    """Retrieve MySQL database connection handle."""
    return get_client()


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _row_to_application(row: Dict[str, Any]) -> Dict[str, Any]:
    missing_docs = row.get("missing_documents")
    if isinstance(missing_docs, str):
        missing_docs = json.loads(missing_docs)
    return {
        "application_id": row["application_id"],
        "customer_email": row["customer_email"],
        "customer_name": row["customer_name"],
        "status": row["status"],
        "risk_classification": row["risk_classification"],
        "flagged": bool(row["flagged"]),
        "default_probability": row["default_probability"],
        "missing_documents": missing_docs or [],
        "final_decision": row["final_decision"],
        "notes": row["notes"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def save_prediction_record(record: Dict[str, Any]) -> Optional[str]:
    """
    Save prediction record to the 'predictions' table.
    Gracefully returns None if database is offline.
    """
    conn = get_db()
    if conn is None:
        return None
    try:
        doc = dict(record)
        created_at = doc.get("created_at")
        if isinstance(created_at, datetime):
            created_at = created_at.astimezone(timezone.utc).replace(tzinfo=None) if created_at.tzinfo else created_at
        else:
            created_at = _utc_now_naive()
            doc["created_at"] = created_at.isoformat()

        doc["model_version"] = config.MODEL_VERSION
        payload_json = json.dumps(doc, default=str)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO predictions (model_version, default_probability, band, decision, payload, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    doc.get("model_version"),
                    doc.get("default_probability"),
                    doc.get("band"),
                    doc.get("decision"),
                    payload_json,
                    created_at,
                ),
            )
            return str(cur.lastrowid)
    except Exception as e:
        logger.error("Failed to save prediction record: %s", e)
        return None


def get_prediction_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent prediction history from MySQL."""
    conn = get_db()
    if conn is None:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT payload FROM predictions ORDER BY created_at DESC, id DESC LIMIT %s",
                (int(limit),),
            )
            rows = cur.fetchall()
        results: List[Dict[str, Any]] = []
        for row in rows:
            payload = row.get("payload")
            if isinstance(payload, str):
                results.append(json.loads(payload))
            elif isinstance(payload, dict):
                results.append(payload)
        return results
    except Exception as e:
        logger.error("Failed to fetch prediction history: %s", e)
        return []


# ---------------------------------------------------------------------------
# In-memory fallbacks when MySQL is offline (keeps workflows & tests working)
# ---------------------------------------------------------------------------
_mem_users: Dict[str, Dict[str, Any]] = {}
_mem_applications: Dict[str, Dict[str, Any]] = {}
_mem_email_logs: List[Dict[str, Any]] = []


def create_user(user_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a new user in MySQL (or in-memory fallback)."""
    email = user_doc["email"].lower()
    now = _utc_now_naive()
    conn = get_db()
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM users WHERE email = %s LIMIT 1", (email,))
            if cur.fetchone():
                raise ValueError("User with this email already exists.")
            doc = dict(user_doc)
            doc["email"] = email
            doc.setdefault("role", "customer")
            doc.setdefault("created_at", now)
            cur.execute(
                """
                INSERT INTO users (user_id, name, email, password_hash, role, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    doc["user_id"],
                    doc["name"],
                    doc["email"],
                    doc["password_hash"],
                    doc["role"],
                    doc["created_at"],
                ),
            )
            return doc

    if email in _mem_users:
        raise ValueError("User with this email already exists.")
    doc = dict(user_doc)
    doc["email"] = email
    doc.setdefault("created_at", now)
    _mem_users[email] = doc
    return doc


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Fetch a user by email from MySQL or in-memory fallback."""
    normalized = email.strip().lower()
    conn = get_db()
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, name, email, password_hash, role, created_at FROM users WHERE email = %s LIMIT 1",
                (normalized,),
            )
            row = cur.fetchone()
            if row:
                return dict(row)
    return _mem_users.get(normalized)


def save_application(app_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a new credit application."""
    now = _utc_now_naive()
    doc = dict(app_doc)
    doc.setdefault("created_at", now)
    doc.setdefault("updated_at", now)
    conn = get_db()
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO applications (
                    application_id, customer_email, customer_name, status,
                    risk_classification, flagged, default_probability,
                    missing_documents, final_decision, notes, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    doc["application_id"],
                    doc["customer_email"],
                    doc["customer_name"],
                    doc["status"],
                    doc["risk_classification"],
                    1 if doc.get("flagged") else 0,
                    doc.get("default_probability"),
                    json.dumps(doc.get("missing_documents", [])),
                    doc.get("final_decision"),
                    doc.get("notes"),
                    doc["created_at"],
                    doc["updated_at"],
                ),
            )
    _mem_applications[doc["application_id"]] = doc
    return doc


def get_application(application_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a credit application by application_id."""
    conn = get_db()
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM applications WHERE application_id = %s LIMIT 1",
                (application_id,),
            )
            row = cur.fetchone()
            if row:
                return _row_to_application(row)
    return _mem_applications.get(application_id)


def update_application(application_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update fields on an existing credit application."""
    patch = dict(updates)
    patch["updated_at"] = _utc_now_naive()
    allowed_cols = {
        "status",
        "risk_classification",
        "flagged",
        "default_probability",
        "missing_documents",
        "final_decision",
        "notes",
        "updated_at",
    }
    conn = get_db()
    if conn is not None:
        set_clauses: List[str] = []
        values: List[Any] = []
        for col, val in patch.items():
            if col not in allowed_cols:
                continue
            set_clauses.append(f"{col} = %s")
            if col == "missing_documents":
                values.append(json.dumps(val))
            elif col == "flagged":
                values.append(1 if val else 0)
            else:
                values.append(val)

        if set_clauses:
            values.append(application_id)
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE applications SET {', '.join(set_clauses)} WHERE application_id = %s",
                    tuple(values),
                )
        found = get_application(application_id)
        if found is not None:
            _mem_applications[application_id] = found
            return found

    existing = _mem_applications.get(application_id)
    if existing is None:
        return None
    existing.update(patch)
    return existing


def has_sent_email(idempotency_key: str) -> bool:
    """Check if an email with this idempotency_key has already been sent."""
    conn = get_db()
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM email_logs WHERE idempotency_key = %s AND success = 1 LIMIT 1",
                (idempotency_key,),
            )
            if cur.fetchone() is not None:
                return True
    return any(
        log.get("idempotency_key") == idempotency_key and log.get("success") is True
        for log in _mem_email_logs
    )


def save_email_log(entry: Dict[str, Any]) -> None:
    """Record an email delivery attempt."""
    doc = dict(entry)
    created_at = _utc_now_naive()
    doc.setdefault("created_at", created_at.isoformat())
    _mem_email_logs.append(doc)
    conn = get_db()
    if conn is not None:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO email_logs (idempotency_key, success, payload, created_at)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    doc.get("idempotency_key"),
                    1 if doc.get("success") else 0,
                    json.dumps(doc, default=str),
                    created_at,
                ),
            )


def get_email_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent email delivery logs from MySQL or in-memory fallback."""
    conn = get_db()
    if conn is not None:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT payload FROM email_logs ORDER BY created_at DESC, id DESC LIMIT %s",
                    (int(limit),),
                )
                rows = cur.fetchall()
            results: List[Dict[str, Any]] = []
            for row in rows:
                payload = row.get("payload")
                if isinstance(payload, str):
                    results.append(json.loads(payload))
                elif isinstance(payload, dict):
                    results.append(payload)
            return results
        except Exception as e:
            logger.error("Failed to fetch email logs: %s", e)
    return list(reversed(_mem_email_logs[-int(limit):]))
