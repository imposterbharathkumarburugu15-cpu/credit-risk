"""API Route definitions for CreditRisk-X FastAPI backend."""
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

import analytics
import chatbot
import config
import cost
import database
import decision
import explainability
from model import get_meta, is_model_available
import optimization
import prediction
import service
from workflows import workflow_router

router = APIRouter(prefix="/api", tags=["CreditRisk-X Core API"])
router.include_router(workflow_router)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class CostCalcRequest(BaseModel):
    fp_count: int = Field(..., ge=0, description="False positive count")
    fn_count: int = Field(..., ge=0, description="False negative count")
    fp_cost: float = Field(config.DEFAULT_FP_COST, ge=0)
    fn_cost: float = Field(config.DEFAULT_FN_COST, ge=0)


class CostCompareRequest(BaseModel):
    baseline_fp: int = Field(..., ge=0)
    baseline_fn: int = Field(..., ge=0)
    optimized_fp: int = Field(..., ge=0)
    optimized_fn: int = Field(..., ge=0)
    fp_cost: float = Field(config.DEFAULT_FP_COST, ge=0)
    fn_cost: float = Field(config.DEFAULT_FN_COST, ge=0)


class ThresholdOptimizeRequest(BaseModel):
    model_name: str = "xgboost"
    fp_cost: float = config.DEFAULT_FP_COST
    fn_cost: float = config.DEFAULT_FN_COST
    min_recall: Optional[float] = None


class ThresholdCompareRequest(BaseModel):
    thresholds: List[float] = Field(default_factory=lambda: [0.16, 0.35, 0.50])
    model_name: str = "xgboost"
    fp_cost: float = config.DEFAULT_FP_COST
    fn_cost: float = config.DEFAULT_FN_COST


class AnalyticsEvaluateRequest(BaseModel):
    y_true: Optional[List[int]] = None
    y_proba: Optional[List[float]] = None


class PortfolioAnalyzeRequest(BaseModel):
    probabilities: List[float] = Field(..., min_length=1)
    threshold: Optional[float] = config.DEFAULT_THRESHOLD


class ExplainRequest(BaseModel):
    features: Dict[str, Any]
    top_n: int = 6
    model_name: str = "xgboost"


class ChatbotRequest(BaseModel):
    message: str


class PredictBatchRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., min_length=1)
    threshold: Optional[float] = config.DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# 1. Health Endpoint
# ---------------------------------------------------------------------------
@router.get("/health")
def health():
    db_connected = False
    try:
        client = database.get_client()
        if client is not None:
            client.ping(reconnect=True)
            db_connected = True
    except Exception:
        db_connected = False

    return {
        "status": "ok",
        "service": "CreditRisk-X Python Core Engine",
        "model_loaded": is_model_available(),
        "model_version": config.MODEL_VERSION,
        "features_expected": 43,
        "mysql_connected": db_connected,
        "default_threshold": config.DEFAULT_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# 2. Prediction Endpoint (supports both full, raw, and wrapped schemas)
# ---------------------------------------------------------------------------
@router.post("/predict")
def predict_endpoint(payload: Dict[str, Any]):
    if not is_model_available():
        raise HTTPException(status_code=503, detail="ML model artifacts are not available.")

    # Unpack customer features if nested
    customer_data = payload.get("customer", payload.get("features", payload))
    threshold = payload.get("threshold", config.DEFAULT_THRESHOLD)

    try:
        pred_res = prediction.predict_probability(customer_data, model_name=config.PRIMARY_MODEL_NAME)
        prob = pred_res["default_probability"]
        dec_res = decision.make_decision(prob, threshold=threshold)

        # Generate TreeSHAP explanations
        explanation = explainability.explain_prediction(
            features=pred_res["features_df"],
            top_n=6,
            model_name=config.PRIMARY_MODEL_NAME
        )

        response_payload = {
            "default_probability": prob,
            "threshold": dec_res["threshold"],
            "band": dec_res["risk_classification"],
            "risk_classification": dec_res["risk_classification"],
            "decision": dec_res["suggested_action"],
            "suggested_action": dec_res["suggested_action"],
            "review_recommendation": dec_res["review_recommendation"],
            "flagged": dec_res["flagged"],
            "reasons": explanation["top_features"],
            "model": "XGBoost (primary)",
            "model_version": config.MODEL_VERSION,
        }

        # Persist to database if available
        database.save_prediction_record(response_payload)
        return response_payload

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict/raw")
def predict_raw_endpoint(data: prediction.RawCustomerInput):
    return predict_endpoint({"customer": data.model_dump()})


@router.post("/predict/batch")
def predict_batch_endpoint(payload: PredictBatchRequest):
    """
    Batch prediction endpoint that separates results into:
      - TABLE 1: PREDICTED DEFAULT (HIGH risk, probability >= threshold)
      - TABLE 2: PREDICTED NON-DEFAULT (LOW / MEDIUM risk)
    """
    if not is_model_available():
        raise HTTPException(status_code=503, detail="ML model artifacts are not available.")

    threshold = payload.threshold or config.DEFAULT_THRESHOLD
    table_predicted_default: List[Dict[str, Any]] = []
    table_predicted_non_default: List[Dict[str, Any]] = []
    skipped_count = 0
    errors: List[str] = []

    for idx, rec in enumerate(payload.records):
        cust_id = rec.get("customer_id", rec.get("record_id", rec.get("id", f"CUST-{idx+1:04d}")))
        cust_name = rec.get("customer_name", rec.get("name", f"Applicant #{idx+1}"))
        email = rec.get("email", rec.get("email_address", ""))
        app_id = rec.get("application_id", rec.get("app_id", f"APP-{idx+1001}"))

        try:
            pred_res = prediction.predict_probability(rec, model_name=config.PRIMARY_MODEL_NAME)
            prob = pred_res["default_probability"]
            dec_res = decision.make_decision(prob, threshold=threshold)

            # Determine email eligibility
            has_valid_email = bool(email and "@" in email and "." in email)
            email_status = "ELIGIBLE" if has_valid_email else "NO_EMAIL"

            item = {
                "customer_id": str(cust_id),
                "customer_name": str(cust_name),
                "email": str(email),
                "application_id": str(app_id),
                "default_probability": prob,
                "risk_classification": dec_res["risk_classification"],
                "suggested_action": dec_res["suggested_action"],
                "review_recommendation": dec_res["review_recommendation"],
                "email_status": email_status,
                "flagged": dec_res["flagged"],
                "features_summary": {
                    "limit_bal": rec.get("LIMIT_BAL"),
                    "age": rec.get("AGE"),
                    "sex": rec.get("SEX"),
                    "education": rec.get("EDUCATION"),
                    "marriage": rec.get("MARRIAGE"),
                    "pay_0": rec.get("PAY_0"),
                    "due_date": rec.get("due_date"),
                    "due_amount": rec.get("due_amount"),
                    "bills": [
                        rec.get("BILL_AMT1", 0),
                        rec.get("BILL_AMT2", 0),
                        rec.get("BILL_AMT3", 0),
                        rec.get("BILL_AMT4", 0),
                        rec.get("BILL_AMT5", 0),
                        rec.get("BILL_AMT6", 0),
                    ],
                    "payments": [
                        rec.get("PAY_AMT1", 0),
                        rec.get("PAY_AMT2", 0),
                        rec.get("PAY_AMT3", 0),
                        rec.get("PAY_AMT4", 0),
                        rec.get("PAY_AMT5", 0),
                        rec.get("PAY_AMT6", 0),
                    ],
                    "pay_statuses": [
                        rec.get("PAY_0", 0),
                        rec.get("PAY_2", 0),
                        rec.get("PAY_3", 0),
                        rec.get("PAY_4", 0),
                        rec.get("PAY_5", 0),
                        rec.get("PAY_6", 0),
                    ],
                }
            }

            if dec_res["flagged"] or prob >= threshold:
                table_predicted_default.append(item)
            else:
                table_predicted_non_default.append(item)

        except Exception as e:
            skipped_count += 1
            if len(errors) < 5:
                errors.append(f"Row {idx + 1}: {e}")
            continue

    return {
        "threshold": threshold,
        "total_processed": len(payload.records),
        "skipped_count": skipped_count,
        "errors": errors,
        "predicted_default_count": len(table_predicted_default),
        "predicted_non_default_count": len(table_predicted_non_default),
        "table_predicted_default": table_predicted_default,
        "table_predicted_non_default": table_predicted_non_default,
    }


# ---------------------------------------------------------------------------
# 3. Cost Engine Endpoints
# ---------------------------------------------------------------------------
@router.post("/cost/calculate")
def cost_calculate(req: CostCalcRequest):
    return cost.compute_classification_cost(
        fp_count=req.fp_count,
        fn_count=req.fn_count,
        fp_cost=req.fp_cost,
        fn_cost=req.fn_cost,
    )


@router.post("/cost/compare")
def cost_compare(req: CostCompareRequest):
    return cost.compare_policy_costs(
        baseline_fp=req.baseline_fp,
        baseline_fn=req.baseline_fn,
        optimized_fp=req.optimized_fp,
        optimized_fn=req.optimized_fn,
        fp_cost=req.fp_cost,
        fn_cost=req.fn_cost,
    )


# ---------------------------------------------------------------------------
# 4. Threshold Optimization Endpoints
# ---------------------------------------------------------------------------
@router.post("/threshold/optimize")
def threshold_optimize(req: ThresholdOptimizeRequest):
    try:
        return optimization.optimize_threshold(
            model_name=req.model_name,
            fp_cost=req.fp_cost,
            fn_cost=req.fn_cost,
            min_recall=req.min_recall,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/threshold/compare")
def threshold_compare(req: ThresholdCompareRequest):
    try:
        return optimization.compare_thresholds(
            thresholds=req.thresholds,
            model_name=req.model_name,
            fp_cost=req.fp_cost,
            fn_cost=req.fn_cost,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# 5. Explainability Endpoints
# ---------------------------------------------------------------------------
@router.post("/explain")
def explain_endpoint(req: ExplainRequest):
    try:
        return explainability.explain_prediction(
            features=req.features,
            top_n=req.top_n,
            model_name=req.model_name,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# 6. Analytics Endpoints
# ---------------------------------------------------------------------------
@router.post("/analytics/evaluate")
def analytics_evaluate(req: AnalyticsEvaluateRequest):
    try:
        if req.y_true is not None and req.y_proba is not None:
            return analytics.evaluate_model_performance(req.y_true, req.y_proba)
        # Default to pre-computed test set metrics
        return analytics.get_stored_test_metrics()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analytics/portfolio")
def analytics_portfolio(req: PortfolioAnalyzeRequest):
    try:
        return analytics.analyze_portfolio(req.probabilities, req.threshold)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ---------------------------------------------------------------------------
# 7. Chatbot Endpoint
# ---------------------------------------------------------------------------
@router.post("/chatbot")
def chatbot_endpoint(req: ChatbotRequest):
    return chatbot.answer_query(req.message)


# ---------------------------------------------------------------------------
# 8. Dashboard Compatibility Routes (matches original app.js expectations)
# ---------------------------------------------------------------------------
def _parse_costs(cost_fn: float, cost_fp: float, cost_tp: float, cost_tn: float) -> Dict[str, float]:
    return {"fn": cost_fn, "fp": cost_fp, "tp": cost_tp, "tn": cost_tn}


@router.get("/models/comparison")
def models_comparison(
    cost_fn: float = Query(5.0, ge=0),
    cost_fp: float = Query(1.0, ge=0),
    cost_tp: float = Query(0.0, ge=0),
    cost_tn: float = Query(0.0, ge=0),
):
    costs = _parse_costs(cost_fn, cost_fp, cost_tp, cost_tn)
    return service.comparison(costs)


@router.get("/cost-curves")
def cost_curves(
    cost_fn: float = Query(5.0, ge=0),
    cost_fp: float = Query(1.0, ge=0),
    cost_tp: float = Query(0.0, ge=0),
    cost_tn: float = Query(0.0, ge=0),
):
    costs = _parse_costs(cost_fn, cost_fp, cost_tp, cost_tn)
    return service.cost_curves(costs)


@router.get("/pr-curves")
def pr_curves():
    return service.pr_curves()


@router.get("/rules")
def rules():
    return service.rules()


@router.get("/sample/{kind}")
def sample(kind: str):
    row = service.sample_customer(kind)
    if row is None:
        raise HTTPException(status_code=404, detail="Sample not found. Must be low, medium, or high.")
    return row


@router.get("/demo-applicants")
def demo_applicants(limit: int = Query(5, ge=1, le=15)):
    """Return a small set of real sample applicants from data/sample_applications_with_emails.csv scored by XGBoost."""
    import csv
    csv_path = config.DATA_DIR / "sample_applications_with_emails.csv"
    if not csv_path.is_file():
        return {"records": []}

    records: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx >= limit:
                break
            parsed: Dict[str, Any] = {}
            for k, v in row.items():
                if k in ("record_id", "customer_name", "email", "application_id", "due_date"):
                    parsed[k] = v
                else:
                    try:
                        parsed[k] = float(v) if "." in str(v) else int(v)
                    except (TypeError, ValueError):
                        parsed[k] = v
            records.append(parsed)

    if not records or not is_model_available():
        return {"records": records}

    scored = predict_batch_endpoint(PredictBatchRequest(records=records, threshold=config.DEFAULT_THRESHOLD))
    combined = scored["table_predicted_default"] + scored["table_predicted_non_default"]
    return {"records": combined, "threshold": scored["threshold"]}
