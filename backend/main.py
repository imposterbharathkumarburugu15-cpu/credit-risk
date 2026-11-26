"""Main FastAPI Application for CreditRisk-X.

Integrates all risk prediction, cost optimization, explainability, analytics,
and FAQ chatbot endpoints while serving the dashboard frontend.
"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import config
import database
from model import load_artifacts
from routes import router

app = FastAPI(
    title="CreditRisk-X — Cost-Sensitive Intelligence Platform API",
    description="Backend API for credit default risk prediction, cost optimization, explainability, and analytics.",
    version="1.0.0",
)

# CORS middleware for development across ports (e.g. Node backend on 5000, Python on 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount core API routes
app.include_router(router)


@app.on_event("startup")
def startup_event():
    """Load model artifacts and initialize database connection on startup."""
    try:
        load_artifacts()
    except Exception as e:
        print(f"[WARNING] Could not load model artifacts on startup: {e}")

    try:
        database.get_client()
    except Exception as e:
        print(f"[WARNING] MySQL connection skipped or failed: {e}")


# Root check
@app.get("/health")
def root_health():
    """Top-level health check."""
    return {"status": "ok", "app": "CreditRisk-X", "port": config.PORT}


# Static Frontend Mounting
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)
