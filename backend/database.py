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
