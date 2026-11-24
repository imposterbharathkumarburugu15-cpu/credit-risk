"""FastAPI app. Run with:  uvicorn main:app --reload"""
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import service
from costs import DEFAULT_COSTS

app = FastAPI(title="Credit Default Risk API")

# handy when the frontend is opened from a different port during development
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class CostInput(BaseModel):
    fn: float = Field(DEFAULT_COSTS["fn"], ge=0)
    fp: float = Field(DEFAULT_COSTS["fp"], ge=0)
    tp: float = Field(DEFAULT_COSTS["tp"], ge=0)
    tn: float = Field(DEFAULT_COSTS["tn"], ge=0)

    def as_dict(self):
        return {"fn": self.fn, "fp": self.fp, "tp": self.tp, "tn": self.tn}


class Customer(BaseModel):
    LIMIT_BAL: float = Field(gt=0)
    SEX: int = Field(ge=1, le=2)
    EDUCATION: int = Field(ge=0, le=6)
    MARRIAGE: int = Field(ge=0, le=3)
    AGE: int = Field(ge=18, le=100)
    PAY_0: int = Field(ge=-2, le=9)
    PAY_2: int = Field(ge=-2, le=9)
    PAY_3: int = Field(ge=-2, le=9)
    PAY_4: int = Field(ge=-2, le=9)
    PAY_5: int = Field(ge=-2, le=9)
    PAY_6: int = Field(ge=-2, le=9)
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)
    PAY_AMT3: float = Field(ge=0)
    PAY_AMT4: float = Field(ge=0)
    PAY_AMT5: float = Field(ge=0)
    PAY_AMT6: float = Field(ge=0)


class ScoreRequest(BaseModel):
    customer: Customer
    costs: Optional[CostInput] = None


def costs_from_query(fn, fp, tp, tn):
    return {"fn": fn, "fp": fp, "tp": tp, "tn": tn}


@app.get("/api/models/comparison")
def models_comparison(
    cost_fn: float = Query(DEFAULT_COSTS["fn"], ge=0),
    cost_fp: float = Query(DEFAULT_COSTS["fp"], ge=0),
    cost_tp: float = Query(DEFAULT_COSTS["tp"], ge=0),
    cost_tn: float = Query(DEFAULT_COSTS["tn"], ge=0),
):
    return service.comparison(costs_from_query(cost_fn, cost_fp, cost_tp, cost_tn))


@app.get("/api/cost-curves")
def cost_curves(
    cost_fn: float = Query(DEFAULT_COSTS["fn"], ge=0),
    cost_fp: float = Query(DEFAULT_COSTS["fp"], ge=0),
    cost_tp: float = Query(DEFAULT_COSTS["tp"], ge=0),
    cost_tn: float = Query(DEFAULT_COSTS["tn"], ge=0),
):
    return service.cost_curves(costs_from_query(cost_fn, cost_fp, cost_tp, cost_tn))


@app.get("/api/pr-curves")
def pr_curves():
    return service.pr_curves()


@app.get("/api/rules")
def rules():
    return service.rules()


@app.get("/api/sample/{kind}")
def sample(kind: str):
    row = service.sample_customer(kind)
    if row is None:
        raise HTTPException(status_code=404, detail="kind must be low, medium or high")
    return row


@app.post("/api/predict")
def predict(req: ScoreRequest):
    costs = (req.costs or CostInput()).as_dict()
    return service.score_customer(req.customer.model_dump(), costs)


# the dashboard (must be mounted last so it doesn't swallow the /api routes)
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
