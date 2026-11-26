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
    """Create the target database and predictions table if they do not exist."""
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
        _client = pymysql.connect(
            host=config.MYSQL_HOST,
            port=config.MYSQL_PORT,
            user=config.MYSQL_USER,
            password=config.MYSQL_PASSWORD,
            connect_timeout=2,
            read_timeout=10,
            write_timeout=10,
            autocommit=True,
            charset="utf8mb4",
            cursorclass=DictCursor,
        )
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
            created_at = datetime.now(timezone.utc).replace(tzinfo=None)
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
