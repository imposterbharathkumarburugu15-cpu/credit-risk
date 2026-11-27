"""Authenticated Customer, Application, and Email Notification Workflows."""
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

import config
import database
import decision
from email_service import EmailService, validate_email_address, validate_smtp_env
from model import is_model_available
import prediction

workflow_router = APIRouter(tags=["Workflows & Email Automation"])
email_service = EmailService(log_store=database)

ALLOWED_STATUSES = {
    "SUBMITTED",
    "UNDER_REVIEW",
    "MISSING_DOCUMENTS",
    "APPROVED",
    "DECLINED",
}
ALLOWED_DECISIONS = {"APPROVED", "DECLINED"}
ALLOWED_UI_EVENTS = {
    "application_submitted",
    "missing_documents",
    "status_change",
    "final_decision",
    "high_risk_review",
    "credit_limit_adjustment",
}


# ---------------------------------------------------------------------------
# Password Hashing & Token Helpers (zero external dependencies)
# ---------------------------------------------------------------------------
def _hash_password(password: str, salt: Optional[str] = None) -> str:
    salt_hex = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_hex.encode("utf-8"), 100_000)
    return f"{salt_hex}${dk.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    if "$" not in stored_hash:
        return False
    salt_hex, expected_hex = stored_hash.split("$", 1)
    actual = _hash_password(password, salt=salt_hex).split("$", 1)[1]
    return hmac.compare_digest(actual, expected_hex)


def _create_token(email: str, role: str, name: str, ttl_seconds: int = 86400) -> str:
    payload = {
        "sub": email.lower(),
        "role": role,
        "name": name,
        "exp": int(time.time()) + ttl_seconds,
    }
    raw = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii")
    sig = hmac.new(config.AUTH_SECRET.encode("utf-8"), raw.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{raw}.{sig}"


def _decode_token(token: str) -> Dict[str, Any]:
    if "." not in token:
        raise HTTPException(status_code=401, detail="Invalid authentication token.")
    raw, sig = token.split(".", 1)
    expected_sig = hmac.new(config.AUTH_SECRET.encode("utf-8"), raw.encode("ascii"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        raise HTTPException(status_code=401, detail="Invalid token signature.")
    try:
        payload = json.loads(base64.urlsafe_b64decode(raw.encode("ascii")).decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=401, detail="Malformed authentication token.")
    if payload.get("exp", 0) < int(time.time()):
        raise HTTPException(status_code=401, detail="Authentication token has expired.")
    return payload


def get_current_user(
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    token = authorization.split(" ", 1)[1].strip()
    return _decode_token(token)


def require_employee(
    authorization: Optional[str] = Header(default=None),
    x_employee_key: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """Allow access only to authenticated employees or valid X-Employee-Key."""
    configured_key = os.getenv("EMPLOYEE_API_KEY", config.EMPLOYEE_API_KEY).strip()
    if configured_key and x_employee_key and hmac.compare_digest(x_employee_key, configured_key):
        return {"sub": "employee-api", "role": "employee", "name": "Risk Officer"}

    if authorization and authorization.lower().startswith("bearer "):
        user = _decode_token(authorization.split(" ", 1)[1].strip())
        if user.get("role") == "employee":
            return user
        raise HTTPException(status_code=403, detail="Employee privileges required.")

    raise HTTPException(status_code=401, detail="Employee authentication required.")


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    email: str
    password: str = Field(..., min_length=6, max_length=128)


class LoginRequest(BaseModel):
    email: str
    password: str


class ApplicationSubmitRequest(BaseModel):
    # Note: any email field sent by frontend is intentionally ignored;
    # the authenticated user's email from the token is always used.
    customer_features: Optional[Dict[str, Any]] = None
    threshold: Optional[float] = None
    notes: Optional[str] = Field(default=None, max_length=500)


class MissingDocumentsRequest(BaseModel):
    missing_documents: List[str] = Field(..., min_length=1)


class StatusUpdateRequest(BaseModel):
    status: str


class FinalDecisionRequest(BaseModel):
    decision: str


class SendNotificationRequest(BaseModel):
    event_type: str
    recipient_email: Optional[str] = None
    application_id: Optional[str] = None
    customer_name: Optional[str] = None
    status: Optional[str] = None
    decision: Optional[str] = None
    missing_documents: Optional[List[str]] = None
    risk_classification: Optional[str] = None


# ---------------------------------------------------------------------------
# Auth Endpoints
# ---------------------------------------------------------------------------
@workflow_router.post("/auth/register", status_code=201)
def register_customer(req: RegisterRequest):
    valid, email_or_err = validate_email_address(req.email)
    if not valid:
        raise HTTPException(status_code=400, detail=email_or_err)

    user_doc = {
        "user_id": f"USR-{secrets.token_hex(4).upper()}",
        "name": req.name.strip(),
        "email": email_or_err.lower(),
        "password_hash": _hash_password(req.password),
        "role": "customer",
    }
    try:
        saved = database.create_user(user_doc)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    # Trigger Event 1: Welcome email AFTER DB save succeeds
    email_res = email_service.send_event_email(
        event_type="registration",
        to_email=saved["email"],
        context={"customer_name": saved["name"]},
        idempotency_key=f"registration:{saved['email']}",
    )

    token = _create_token(saved["email"], saved["role"], saved["name"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "user_id": saved["user_id"],
            "name": saved["name"],
            "email": saved["email"],
            "role": saved["role"],
        },
        "email_notification": email_res,
    }


@workflow_router.post("/auth/login")
def login_user(req: LoginRequest):
    valid, email_or_err = validate_email_address(req.email)
    if not valid:
        raise HTTPException(status_code=400, detail=email_or_err)

    user = database.get_user_by_email(email_or_err.lower())
    if not user or not _verify_password(req.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    token = _create_token(user["email"], user.get("role", "customer"), user.get("name", "Customer"))
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "user_id": user.get("user_id"),
            "name": user.get("name"),
            "email": user["email"],
            "role": user.get("role", "customer"),
        },
    }


# ---------------------------------------------------------------------------
# Application Workflow Endpoints
# ---------------------------------------------------------------------------
@workflow_router.post("/applications", status_code=201)
def submit_application(
    req: ApplicationSubmitRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Submit a credit application using the authenticated customer's email."""
    auth_email = current_user["sub"]
    customer_name = current_user.get("name", "Customer")
    app_id = f"APP-{secrets.token_hex(4).upper()}"

    risk_classification = "LOW"
    flagged = False
    default_probability = None

    if req.customer_features and is_model_available():
        pred_res = prediction.predict_probability(req.customer_features, model_name=config.PRIMARY_MODEL_NAME)
        default_probability = pred_res["default_probability"]
        dec_res = decision.make_decision(default_probability, threshold=req.threshold)
        risk_classification = dec_res["risk_classification"]
        flagged = dec_res["flagged"]

    initial_status = "UNDER_REVIEW" if (flagged or risk_classification == "HIGH") else "SUBMITTED"

    app_doc = {
        "application_id": app_id,
        "customer_email": auth_email,
        "customer_name": customer_name,
        "status": initial_status,
        "risk_classification": risk_classification,
        "flagged": flagged,
        "default_probability": default_probability,
        "missing_documents": [],
        "final_decision": None,
        "notes": req.notes,
    }
    saved = database.save_application(app_doc)

    # Trigger Event 2: Confirmation email AFTER DB save succeeds
    confirm_res = email_service.send_event_email(
        event_type="application_submitted",
        to_email=auth_email,
        context={"customer_name": customer_name, "application_id": app_id},
        idempotency_key=f"application_submitted:{app_id}",
    )

    notifications: Dict[str, Any] = {"customer_confirmation": confirm_res}

    # Trigger Event 6: High-risk application requiring employee review
    if flagged or risk_classification == "HIGH":
        review_inbox = os.getenv("EMPLOYEE_REVIEW_EMAIL", config.EMPLOYEE_REVIEW_EMAIL).strip() or os.getenv("EMAIL_USER", "").strip()
        if review_inbox:
            emp_res = email_service.send_event_email(
                event_type="high_risk_review",
                to_email=review_inbox,
                context={"application_id": app_id, "risk_classification": risk_classification},
                idempotency_key=f"high_risk_review:{app_id}",
            )
            notifications["employee_review_alert"] = emp_res

    return {
        "application_id": saved["application_id"],
        "status": saved["status"],
        "risk_classification": saved["risk_classification"],
        "flagged": saved["flagged"],
        "email_notifications": notifications,
    }


@workflow_router.post("/applications/{app_id}/missing-documents")
def request_missing_documents(
    app_id: str,
    req: MissingDocumentsRequest,
    _employee: Dict[str, Any] = Depends(require_employee),
):
    app_record = database.get_application(app_id)
    if not app_record:
        raise HTTPException(status_code=404, detail="Application not found.")

    updated = database.update_application(
        app_id,
        {"status": "MISSING_DOCUMENTS", "missing_documents": req.missing_documents},
    )

    # Trigger Event 3: Document request email AFTER DB update succeeds
    doc_key = ",".join(sorted(d.strip() for d in req.missing_documents))
    email_res = email_service.send_event_email(
        event_type="missing_documents",
        to_email=updated["customer_email"],
        context={
            "customer_name": updated.get("customer_name", "Customer"),
            "application_id": app_id,
            "missing_documents": req.missing_documents,
        },
        idempotency_key=f"missing_documents:{app_id}:{doc_key}",
    )
    return {"application_id": app_id, "status": updated["status"], "email_notification": email_res}


@workflow_router.patch("/applications/{app_id}/status")
def update_application_status(
    app_id: str,
    req: StatusUpdateRequest,
    _employee: Dict[str, Any] = Depends(require_employee),
):
    new_status = req.status.strip().upper()
    if new_status not in ALLOWED_STATUSES:
        raise HTTPException(status_code=400, detail=f"Status must be one of {sorted(ALLOWED_STATUSES)}")

    app_record = database.get_application(app_id)
    if not app_record:
        raise HTTPException(status_code=404, detail="Application not found.")

    updated = database.update_application(app_id, {"status": new_status})

    # Trigger Event 4: Status change notification AFTER DB update succeeds
    email_res = email_service.send_event_email(
        event_type="status_change",
        to_email=updated["customer_email"],
        context={
            "customer_name": updated.get("customer_name", "Customer"),
            "application_id": app_id,
            "status": new_status,
        },
        idempotency_key=f"status_change:{app_id}:{new_status}",
    )
    return {"application_id": app_id, "status": updated["status"], "email_notification": email_res}


@workflow_router.post("/applications/{app_id}/decision")
def record_final_decision(
    app_id: str,
    req: FinalDecisionRequest,
    _employee: Dict[str, Any] = Depends(require_employee),
):
    final_dec = req.decision.strip().upper()
    if final_dec not in ALLOWED_DECISIONS:
        raise HTTPException(status_code=400, detail=f"Decision must be one of {sorted(ALLOWED_DECISIONS)}")

    app_record = database.get_application(app_id)
    if not app_record:
        raise HTTPException(status_code=404, detail="Application not found.")

    updated = database.update_application(
        app_id,
        {"status": final_dec, "final_decision": final_dec},
    )

    # Trigger Event 5: Final decision notification AFTER DB update succeeds
    email_res = email_service.send_event_email(
        event_type="final_decision",
        to_email=updated["customer_email"],
        context={
            "customer_name": updated.get("customer_name", "Customer"),
            "application_id": app_id,
            "decision": final_dec,
        },
        idempotency_key=f"final_decision:{app_id}:{final_dec}",
    )
    return {
        "application_id": app_id,
        "status": updated["status"],
        "final_decision": updated["final_decision"],
        "email_notification": email_res,
    }


# ---------------------------------------------------------------------------
# Protected Email Status & Dashboard "Send" Endpoint
# ---------------------------------------------------------------------------
@workflow_router.get("/email/status")
def email_config_status():
    """Return safe SMTP readiness metadata without exposing credentials."""
    ok_env, msg = validate_smtp_env()
    return {
        "configured": ok_env,
        "message": msg,
        "smtp_host": os.getenv("SMTP_HOST", config.SMTP_HOST),
        "smtp_port": int(os.getenv("SMTP_PORT", str(config.SMTP_PORT))),
    }


@workflow_router.get("/email/logs")
def list_email_logs(
    limit: int = 50,
    _employee: Dict[str, Any] = Depends(require_employee),
):
    """Return recent email delivery logs (employee-only)."""
    return {"logs": database.get_email_logs(limit=limit)}


@workflow_router.post("/notifications/send")
def send_template_notification(
    req: SendNotificationRequest,
    authorization: Optional[str] = Header(default=None),
    x_employee_key: Optional[str] = Header(default=None),
):
    """Send a template-based workflow notification (used by dashboard Send button).

    Security:
      - Requires employee credentials (employee bearer token or X-Employee-Key) OR
        an authenticated customer token (restricted strictly to their own email).
      - Only allows predefined workflow templates; never arbitrary email relays.
    """
    if req.event_type not in ALLOWED_UI_EVENTS:
        raise HTTPException(status_code=400, detail=f"Unsupported event_type. Allowed: {sorted(ALLOWED_UI_EVENTS)}")

    is_employee = False
    caller_email: Optional[str] = None

    configured_key = os.getenv("EMPLOYEE_API_KEY", config.EMPLOYEE_API_KEY).strip()
    if configured_key and x_employee_key and hmac.compare_digest(x_employee_key, configured_key):
        is_employee = True
    elif authorization and authorization.lower().startswith("bearer "):
        user = _decode_token(authorization.split(" ", 1)[1].strip())
        caller_email = user["sub"]
        is_employee = user.get("role") == "employee"
    else:
        raise HTTPException(status_code=401, detail="Authentication required to send notifications.")

    if is_employee:
        target_email = req.recipient_email
        if req.event_type == "high_risk_review" and not target_email:
            target_email = os.getenv("EMPLOYEE_REVIEW_EMAIL", config.EMPLOYEE_REVIEW_EMAIL).strip()
    else:
        # Customers can never send emails to arbitrary addresses or trigger internal alerts
        if req.event_type in {"missing_documents", "status_change", "final_decision", "high_risk_review", "credit_limit_adjustment"}:
            raise HTTPException(status_code=403, detail="Only employees can send this notification type.")
        target_email = caller_email

    if not target_email:
        raise HTTPException(status_code=400, detail="Recipient email is required.")

    valid, email_or_err = validate_email_address(target_email)
    if not valid:
        raise HTTPException(status_code=400, detail=email_or_err)

    app_id = req.application_id or f"APP-{secrets.token_hex(3).upper()}"
    context = {
        "customer_name": req.customer_name or "Customer",
        "application_id": app_id,
        "status": req.status or "UNDER_REVIEW",
        "decision": req.decision or "APPROVED",
        "missing_documents": req.missing_documents or [],
        "risk_classification": req.risk_classification or "HIGH",
    }

    res = email_service.send_event_email(
        event_type=req.event_type,
        to_email=email_or_err,
        context=context,
        idempotency_key=f"ui:{req.event_type}:{app_id}:{email_or_err}",
    )
    return res
