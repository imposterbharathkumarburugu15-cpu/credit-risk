"""MongoDB database integration for CreditRisk-X."""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pymongo
from pymongo import MongoClient

import config

logger = logging.getLogger("creditrisk_x.database")

_client: Optional[MongoClient] = None
_db = None


def get_client() -> Optional[MongoClient]:
    """Retrieve or initialize the MongoDB client gracefully."""
    global _client
    if _client is None:
        try:
            _client = MongoClient(
                config.MONGO_URI,
                serverSelectionTimeoutMS=2000,
                connectTimeoutMS=2000,
            )
            # Test connectivity
            _client.admin.command("ping")
            logger.info("Successfully connected to MongoDB at %s", config.MONGO_URI)
        except Exception as e:
            logger.warning("MongoDB connection could not be established: %s", e)
            _client = None
    return _client


def get_db():
    """Retrieve MongoDB database instance."""
    global _db
    client = get_client()
    if client is not None:
        _db = client[config.DATABASE_NAME]
    return _db


def save_prediction_record(record: Dict[str, Any]) -> Optional[str]:
    """
    Save prediction record to the 'predictions' collection.
    Gracefully returns None if database is offline.
    """
    db = get_db()
    if db is None:
        return None
    try:
        doc = dict(record)
        if "created_at" not in doc:
            doc["created_at"] = datetime.now(timezone.utc)
        doc["model_version"] = config.MODEL_VERSION
        res = db.predictions.insert_one(doc)
        return str(res.inserted_id)
    except Exception as e:
        logger.error("Failed to save prediction record: %s", e)
        return None


def get_prediction_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent prediction history from MongoDB."""
    db = get_db()
    if db is None:
        return []
    try:
        cursor = db.predictions.find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
        return list(cursor)
    except Exception as e:
        logger.error("Failed to fetch prediction history: %s", e)
        return []


# ---------------------------------------------------------------------------
# In-memory fallbacks when MongoDB is offline (keeps workflows & tests working)
# ---------------------------------------------------------------------------
_mem_users: Dict[str, Dict[str, Any]] = {}
_mem_applications: Dict[str, Dict[str, Any]] = {}
_mem_email_logs: List[Dict[str, Any]] = []


def create_user(user_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a new user in MongoDB (or in-memory fallback)."""
    email = user_doc["email"].lower()
    db = get_db()
    if db is not None:
        existing = db.users.find_one({"email": email}, {"_id": 0})
        if existing:
            raise ValueError("User with this email already exists.")
        doc = dict(user_doc)
        doc["email"] = email
        doc.setdefault("created_at", datetime.now(timezone.utc))
        db.users.insert_one(doc)
        doc.pop("_id", None)
        return doc

    if email in _mem_users:
        raise ValueError("User with this email already exists.")
    doc = dict(user_doc)
    doc["email"] = email
    doc.setdefault("created_at", datetime.now(timezone.utc))
    _mem_users[email] = doc
    return doc


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Fetch a user by email from MongoDB or in-memory fallback."""
    normalized = email.strip().lower()
    db = get_db()
    if db is not None:
        return db.users.find_one({"email": normalized}, {"_id": 0})
    return _mem_users.get(normalized)


def save_application(app_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a new credit application."""
    doc = dict(app_doc)
    doc.setdefault("created_at", datetime.now(timezone.utc))
    doc.setdefault("updated_at", datetime.now(timezone.utc))
    db = get_db()
    if db is not None:
        db.applications.insert_one(dict(doc))
    _mem_applications[doc["application_id"]] = doc
    return doc


def get_application(application_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a credit application by application_id."""
    db = get_db()
    if db is not None:
        found = db.applications.find_one({"application_id": application_id}, {"_id": 0})
        if found:
            return found
    return _mem_applications.get(application_id)


def update_application(application_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update fields on an existing credit application."""
    patch = dict(updates)
    patch["updated_at"] = datetime.now(timezone.utc)
    db = get_db()
    if db is not None:
        res = db.applications.find_one_and_update(
            {"application_id": application_id},
            {"$set": patch},
            projection={"_id": 0},
            return_document=pymongo.ReturnDocument.AFTER,
        )
        if res is not None:
            _mem_applications[application_id] = res
            return res
    existing = _mem_applications.get(application_id)
    if existing is None:
        return None
    existing.update(patch)
    return existing


def has_sent_email(idempotency_key: str) -> bool:
    """Check if an email with this idempotency_key has already been sent."""
    db = get_db()
    if db is not None:
        found = db.email_logs.find_one(
            {"idempotency_key": idempotency_key, "success": True},
            {"_id": 1},
        )
        if found is not None:
            return True
    return any(
        log.get("idempotency_key") == idempotency_key and log.get("success") is True
        for log in _mem_email_logs
    )


def save_email_log(entry: Dict[str, Any]) -> None:
    """Record an email delivery attempt."""
    doc = dict(entry)
    doc.setdefault("created_at", datetime.now(timezone.utc))
    _mem_email_logs.append(doc)
    db = get_db()
    if db is not None:
        db.email_logs.insert_one(dict(doc))
