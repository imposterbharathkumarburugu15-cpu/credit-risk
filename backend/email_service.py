"""Reusable SMTP Email Automation Service for CreditRisk-X.

Handles SMTP transporter configuration, recipient validation, reusable HTML/text
email templates, duplicate notification prevention, and resilient error handling.
"""
from email.message import EmailMessage
import html
import logging
import os
import re
import smtplib
from typing import Any, Dict, List, Optional, Tuple

import config

logger = logging.getLogger("creditrisk_x.email")

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")


def validate_email_address(email: Optional[str]) -> Tuple[bool, str]:
    """Validate email format and reject header-injection characters."""
    if not email or not isinstance(email, str):
        return False, "Recipient email is required."
    cleaned = email.strip()
    if any(ch in cleaned for ch in ("\r", "\n", "\x00")):
        return False, "Invalid characters in email address."
    if len(cleaned) > 254 or not EMAIL_REGEX.match(cleaned):
        return False, "Invalid email address format."
    return True, cleaned


def get_email_credentials() -> Tuple[str, str]:
    """Read EMAIL_USER and EMAIL_APP_PASSWORD dynamically from environment/config."""
    user = os.getenv("EMAIL_USER", config.EMAIL_USER).strip()
    password = os.getenv("EMAIL_APP_PASSWORD", config.EMAIL_APP_PASSWORD).strip()
    return user, password


def validate_smtp_env() -> Tuple[bool, str]:
    """Verify required SMTP environment variables are configured."""
    user, password = get_email_credentials()
    if not user:
        return False, "EMAIL_USER environment variable is not configured."
    valid_sender, _ = validate_email_address(user)
    if not valid_sender:
        return False, "EMAIL_USER is not a valid email address."
    if not password:
        return False, "EMAIL_APP_PASSWORD environment variable is not configured."
    return True, "SMTP configuration is valid."


def _wrap_html(title: str, body_html: str) -> str:
    """Wrap template body in a clean, responsive email layout."""
    safe_title = html.escape(title)
    return f"""<!DOCTYPE html>
<html lang="en">
<body style="font-family:Segoe UI,Arial,sans-serif;background:#f4f6f9;margin:0;padding:24px;color:#1f2933;">
  <div style="max-width:560px;margin:0 auto;background:#ffffff;border:1px solid #dde3ea;border-radius:8px;overflow:hidden;">
    <div style="background:#2b5fa5;color:#ffffff;padding:18px 24px;">
      <h2 style="margin:0;font-size:18px;">CreditRisk-X</h2>
      <div style="font-size:13px;opacity:0.9;margin-top:4px;">{safe_title}</div>
    </div>
    <div style="padding:22px 24px;line-height:1.55;font-size:14px;">
      {body_html}
      <hr style="border:none;border-top:1px solid #e6eaf0;margin:20px 0;">
      <p style="font-size:12px;color:#6b7785;margin:0;">
        This is an automated message from CreditRisk-X. Please do not reply with sensitive personal or financial details.
      </p>
    </div>
  </div>
</body>
</html>"""


def render_template(event_type: str, context: Dict[str, Any]) -> Tuple[str, str, str]:
    """Render (subject, text_body, html_body) for supported workflow events.

    Intentionally excludes sensitive financial figures (credit limits, balances,
    income, or internal model scores) from email bodies.
    """
    name = html.escape(str(context.get("customer_name") or "Customer"))
    raw_name = str(context.get("customer_name") or "Customer")
    app_id = html.escape(str(context.get("application_id") or "N/A"))
    raw_app_id = str(context.get("application_id") or "N/A")

    if event_type == "registration":
        subject = "Welcome to CreditRisk-X"
        text = (
            f"Hello {raw_name},\n\n"
            "Welcome to CreditRisk-X! Your customer account has been created.\n"
            "You can now sign in to submit credit applications and track updates.\n\n"
            "Regards,\nCreditRisk-X Team"
        )
        body = (
            f"<p>Hello <strong>{name}</strong>,</p>"
            "<p>Welcome to <strong>CreditRisk-X</strong>! Your customer account has been created.</p>"
            "<p>You can now sign in to submit credit applications and track status updates.</p>"
        )
        return subject, text, _wrap_html("Account Registration", body)

    if event_type == "application_submitted":
        subject = f"Application Received ({raw_app_id})"
        text = (
            f"Hello {raw_name},\n\n"
            f"We have received your credit application ({raw_app_id}).\n"
            "Our team is reviewing your submission and will notify you of any updates.\n\n"
            "Regards,\nCreditRisk-X Team"
        )
        body = (
            f"<p>Hello <strong>{name}</strong>,</p>"
            f"<p>We have received your credit application (<strong>{app_id}</strong>).</p>"
            "<p>Our team is reviewing your submission and will notify you when its status changes.</p>"
        )
        return subject, text, _wrap_html("Application Confirmation", body)

    if event_type == "missing_documents":
        docs: List[str] = [str(d) for d in (context.get("missing_documents") or []) if str(d).strip()]
        if not docs:
            docs = ["Proof of identity", "Recent income verification"]
        doc_lines = "\n".join(f"- {d}" for d in docs)
        doc_html = "".join(f"<li>{html.escape(d)}</li>" for d in docs)
        subject = f"Action Required: Additional Documents Needed ({raw_app_id})"
        text = (
            f"Hello {raw_name},\n\n"
            f"To continue processing application {raw_app_id}, please provide the following documents:\n"
            f"{doc_lines}\n\n"
            "Please upload them through your CreditRisk-X portal.\n\n"
            "Regards,\nCreditRisk-X Team"
        )
        body = (
            f"<p>Hello <strong>{name}</strong>,</p>"
            f"<p>To continue processing application <strong>{app_id}</strong>, please provide:</p>"
            f"<ul>{doc_html}</ul>"
            "<p>Please submit these documents through your CreditRisk-X portal.</p>"
        )
        return subject, text, _wrap_html("Document Request", body)

    if event_type == "status_change":
        new_status = str(context.get("status") or "UPDATED")
        safe_status = html.escape(new_status)
        subject = f"Application Status Update ({raw_app_id}): {new_status}"
        text = (
            f"Hello {raw_name},\n\n"
            f"The status of your application ({raw_app_id}) has been updated to: {new_status}.\n"
            "Sign in to your account for next steps.\n\n"
            "Regards,\nCreditRisk-X Team"
        )
        body = (
            f"<p>Hello <strong>{name}</strong>,</p>"
            f"<p>The status of your application (<strong>{app_id}</strong>) has been updated to: "
            f"<strong>{safe_status}</strong>.</p>"
            "<p>Please sign in to your account to view next steps.</p>"
        )
        return subject, text, _wrap_html("Application Status Update", body)

    if event_type == "final_decision":
        decision_val = str(context.get("decision") or "REVIEWED")
        safe_decision = html.escape(decision_val)
        subject = f"Final Decision for Application {raw_app_id}: {decision_val}"
        text = (
            f"Hello {raw_name},\n\n"
            f"A final decision has been recorded for your application ({raw_app_id}): {decision_val}.\n"
            "Please log in to your CreditRisk-X portal for full details.\n\n"
            "Regards,\nCreditRisk-X Team"
        )
        body = (
            f"<p>Hello <strong>{name}</strong>,</p>"
            f"<p>A final decision has been recorded for application <strong>{app_id}</strong>: "
            f"<strong>{safe_decision}</strong>.</p>"
            "<p>Please log in to your CreditRisk-X portal for full details.</p>"
        )
        return subject, text, _wrap_html("Final Application Decision", body)

    if event_type == "high_risk_review":
        risk_band = str(context.get("risk_classification") or "HIGH")
        safe_band = html.escape(risk_band)
        subject = f"[Review Required] High-Risk Application {raw_app_id}"
        text = (
            "Hello Risk Review Team,\n\n"
            f"Application {raw_app_id} has been flagged as {risk_band} risk and requires employee review.\n"
            "Please open the internal review queue to inspect the application.\n\n"
            "CreditRisk-X Automated Alert"
        )
        body = (
            "<p>Hello <strong>Risk Review Team</strong>,</p>"
            f"<p>Application <strong>{app_id}</strong> has been flagged as <strong>{safe_band}</strong> risk "
            "and requires manual employee review.</p>"
            "<p>Please open the internal review queue to inspect this application.</p>"
        )
        return subject, text, _wrap_html("High-Risk Employee Review Alert", body)

    raise ValueError(f"Unsupported email event type: {event_type}")


class EmailService:
    """Reusable SMTP email service with idempotency and safe error handling."""

    def __init__(self, log_store=None):
        # In-memory idempotency cache; optionally backed by database module
        self._sent_keys: set[str] = set()
        self._log_store = log_store

    def _is_duplicate(self, idempotency_key: Optional[str]) -> bool:
        if not idempotency_key:
            return False
        if idempotency_key in self._sent_keys:
            return True
        if self._log_store is not None:
            try:
                if self._log_store.has_sent_email(idempotency_key):
                    self._sent_keys.add(idempotency_key)
                    return True
            except Exception:
                pass
        return False

    def _record_email_log(self, entry: Dict[str, Any]) -> None:
        key = entry.get("idempotency_key")
        if entry.get("success") and key:
            self._sent_keys.add(key)
        if self._log_store is not None:
            try:
                self._log_store.save_email_log(entry)
            except Exception as exc:
                logger.warning("Could not persist email log: %s", exc)

    def create_transporter(self):
        """Create and authenticate an SMTP client connection."""
        user, password = get_email_credentials()
        host = os.getenv("SMTP_HOST", config.SMTP_HOST).strip() or "smtp.gmail.com"
        port = int(os.getenv("SMTP_PORT", str(config.SMTP_PORT)))

        if port == 465:
            server = smtplib.SMTP_SSL(host, port, timeout=15)
        else:
            server = smtplib.SMTP(host, port, timeout=15)
            server.ehlo()
            server.starttls()
            server.ehlo()
        server.login(user, password)
        return server

    def send_email(
        self,
        to_email: str,
        subject: str,
        text_body: str,
        html_body: Optional[str] = None,
        event_type: str = "notification",
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send an email safely. Never raises on SMTP failure."""
        valid_rcpt, rcpt_or_err = validate_email_address(to_email)
        if not valid_rcpt:
            return {
                "success": False,
                "status": "invalid_recipient",
                "event_type": event_type,
                "recipient": None,
                "message": rcpt_or_err,
                "idempotency_key": idempotency_key,
            }

        if self._is_duplicate(idempotency_key):
            return {
                "success": True,
                "status": "skipped_duplicate",
                "event_type": event_type,
                "recipient": rcpt_or_err,
                "message": "Duplicate notification suppressed.",
                "idempotency_key": idempotency_key,
            }

        ok_env, env_msg = validate_smtp_env()
        if not ok_env:
            logger.warning("Email skipped (%s): %s", event_type, env_msg)
            return {
                "success": False,
                "status": "misconfigured",
                "event_type": event_type,
                "recipient": rcpt_or_err,
                "message": env_msg,
                "idempotency_key": idempotency_key,
            }

        sender_user, _ = get_email_credentials()
        from_name = os.getenv("EMAIL_FROM_NAME", config.EMAIL_FROM_NAME).strip() or "CreditRisk-X"

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = f"{from_name} <{sender_user}>"
        msg["To"] = rcpt_or_err
        msg.set_content(text_body)
        if html_body:
            msg.add_alternative(html_body, subtype="html")

        try:
            with self.create_transporter() as server:
                server.send_message(msg)
            result = {
                "success": True,
                "status": "sent",
                "event_type": event_type,
                "recipient": rcpt_or_err,
                "message": "Email sent successfully.",
                "idempotency_key": idempotency_key,
            }
            self._record_email_log(result)
            return result
        except Exception as exc:
            # Never log or return raw SMTP credentials
            logger.error("SMTP send failed for event=%s: %s", event_type, type(exc).__name__)
            result = {
                "success": False,
                "status": "failed",
                "event_type": event_type,
                "recipient": rcpt_or_err,
                "message": f"SMTP delivery failed: {type(exc).__name__}",
                "idempotency_key": idempotency_key,
            }
            self._record_email_log(result)
            return result

    def send_event_email(
        self,
        event_type: str,
        to_email: str,
        context: Dict[str, Any],
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Render a standard template and send it."""
        subject, text_body, html_body = render_template(event_type, context)
        return self.send_email(
            to_email=to_email,
            subject=subject,
            text_body=text_body,
            html_body=html_body,
            event_type=event_type,
            idempotency_key=idempotency_key,
        )
