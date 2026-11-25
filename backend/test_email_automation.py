"""Tests for CreditRisk-X Email Automation, Workflows, Reliability, and Security."""
import os
import secrets
import smtplib
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import config
import database
import email_service
from main import app
import model
import workflows


class TestEmailAutomation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model.load_artifacts()
        cls.client = TestClient(app)
        cls.run_tag = secrets.token_hex(4)

    def setUp(self):
        # Fresh EmailService per unit test unless testing workflow integration
        self.svc = email_service.EmailService()
        self.env_patcher = patch.dict(
            os.environ,
            {
                "EMAIL_USER": "sender@example.com",
                "EMAIL_APP_PASSWORD": "test-app-password-16ch",
                "SMTP_HOST": "smtp.gmail.com",
                "SMTP_PORT": "587",
                "EMPLOYEE_REVIEW_EMAIL": "reviewer@example.com",
                "EMPLOYEE_API_KEY": "emp-secret-key-123",
            },
        )
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    def test_01_env_var_validation(self):
        """Missing or invalid EMAIL_USER / EMAIL_APP_PASSWORD is detected."""
        ok, _ = email_service.validate_smtp_env()
        self.assertTrue(ok)

        with patch.dict(os.environ, {"EMAIL_USER": "", "EMAIL_APP_PASSWORD": ""}):
            ok_missing, msg = email_service.validate_smtp_env()
            self.assertFalse(ok_missing)
            self.assertIn("EMAIL_USER", msg)

        with patch.dict(os.environ, {"EMAIL_USER": "sender@example.com", "EMAIL_APP_PASSWORD": ""}):
            ok_no_pw, msg_pw = email_service.validate_smtp_env()
            self.assertFalse(ok_no_pw)
            self.assertIn("EMAIL_APP_PASSWORD", msg_pw)

    def test_02_smtp_configuration_and_successful_send(self):
        """SMTP transporter configures TLS/login and sends message cleanly."""
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)

        with patch("email_service.smtplib.SMTP", return_value=mock_smtp) as smtp_cls:
            res = self.svc.send_event_email(
                event_type="registration",
                to_email="customer@example.com",
                context={"customer_name": "Alice"},
                idempotency_key=f"reg:{self.run_tag}:1",
            )

        self.assertTrue(res["success"])
        self.assertEqual(res["status"], "sent")
        smtp_cls.assert_called_once_with("smtp.gmail.com", 587, timeout=15)
        mock_smtp.starttls.assert_called_once()
        mock_smtp.login.assert_called_once_with("sender@example.com", "test-app-password-16ch")
        mock_smtp.send_message.assert_called_once()

    def test_03_invalid_recipient_handling(self):
        """Invalid or header-injected recipient emails are rejected before SMTP."""
        for bad in ["", "not-an-email", "user@bad", "a@b.com\r\nBcc:evil@x.com"]:
            res = self.svc.send_email(
                to_email=bad,
                subject="Test",
                text_body="Body",
            )
            self.assertFalse(res["success"])
            self.assertEqual(res["status"], "invalid_recipient")

    def test_04_smtp_failure_resilience(self):
        """SMTP exceptions return failed result without crashing or leaking credentials."""
        with patch("email_service.smtplib.SMTP", side_effect=smtplib.SMTPAuthenticationError(535, b"Auth failed")):
            res = self.svc.send_event_email(
                event_type="application_submitted",
                to_email="customer@example.com",
                context={"customer_name": "Bob", "application_id": "APP-FAIL1"},
            )
        self.assertFalse(res["success"])
        self.assertEqual(res["status"], "failed")
        self.assertNotIn("test-app-password-16ch", res["message"])

    def test_05_duplicate_notification_prevention(self):
        """Same idempotency key is sent once and suppressed on repeat calls."""
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)

        with patch("email_service.smtplib.SMTP", return_value=mock_smtp):
            r1 = self.svc.send_event_email(
                event_type="status_change",
                to_email="dup@example.com",
                context={"customer_name": "Dana", "application_id": "APP-DUP", "status": "UNDER_REVIEW"},
                idempotency_key=f"status:APP-DUP:UNDER_REVIEW:{self.run_tag}",
            )
            r2 = self.svc.send_event_email(
                event_type="status_change",
                to_email="dup@example.com",
                context={"customer_name": "Dana", "application_id": "APP-DUP", "status": "UNDER_REVIEW"},
                idempotency_key=f"status:APP-DUP:UNDER_REVIEW:{self.run_tag}",
            )

        self.assertEqual(r1["status"], "sent")
        self.assertEqual(r2["status"], "skipped_duplicate")
        self.assertEqual(mock_smtp.send_message.call_count, 1)

    def test_06_full_workflow_and_smtp_failure_does_not_fail_application(self):
        """Registration, high-risk submission, missing docs, status, decision, and SMTP failure resilience."""
        cust_email = f"cust_{self.run_tag}@example.com"
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)

        with patch("email_service.smtplib.SMTP", return_value=mock_smtp):
            # 1. Register -> Welcome email
            reg_res = self.client.post(
                "/api/auth/register",
                json={"name": "Test Customer", "email": cust_email, "password": "strongpassword"},
            )
            self.assertEqual(reg_res.status_code, 201)
            token = reg_res.json()["access_token"]
            self.assertEqual(reg_res.json()["email_notification"]["status"], "sent")

            # 2. Submit HIGH-risk application (even if frontend tries to pass a spoofed email, it is ignored)
            high_risk_features = {
                "LIMIT_BAL": 20000.0, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 35,
                "PAY_0": 3, "PAY_2": 2, "PAY_3": 2, "PAY_4": 2, "PAY_5": 0, "PAY_6": 0,
                "BILL_AMT1": 19500.0, "BILL_AMT2": 19000.0, "BILL_AMT3": 18500.0,
                "BILL_AMT4": 18000.0, "BILL_AMT5": 17500.0, "BILL_AMT6": 17000.0,
                "PAY_AMT1": 0.0, "PAY_AMT2": 0.0, "PAY_AMT3": 0.0,
                "PAY_AMT4": 0.0, "PAY_AMT5": 0.0, "PAY_AMT6": 0.0,
            }
            app_res = self.client.post(
                "/api/applications",
                headers={"Authorization": f"Bearer {token}"},
                json={"email": "attacker@evil.com", "customer_features": high_risk_features},
            )
            self.assertEqual(app_res.status_code, 201)
            app_data = app_res.json()
            app_id = app_data["application_id"]
            self.assertEqual(app_data["risk_classification"], "HIGH")
            self.assertEqual(app_data["email_notifications"]["customer_confirmation"]["recipient"], cust_email)
            self.assertEqual(app_data["email_notifications"]["employee_review_alert"]["recipient"], "reviewer@example.com")

            # 3. Missing documents (employee only)
            doc_res = self.client.post(
                f"/api/applications/{app_id}/missing-documents",
                headers={"X-Employee-Key": "emp-secret-key-123"},
                json={"missing_documents": ["Government ID", "Latest Payslip"]},
            )
            self.assertEqual(doc_res.status_code, 200)
            self.assertEqual(doc_res.json()["email_notification"]["status"], "sent")

            # 4. Status change (employee only)
            st_res = self.client.patch(
                f"/api/applications/{app_id}/status",
                headers={"X-Employee-Key": "emp-secret-key-123"},
                json={"status": "UNDER_REVIEW"},
            )
            self.assertEqual(st_res.status_code, 200)
            self.assertEqual(st_res.json()["email_notification"]["status"], "sent")

            # 5. Final decision (employee only)
            dec_res = self.client.post(
                f"/api/applications/{app_id}/decision",
                headers={"X-Employee-Key": "emp-secret-key-123"},
                json={"decision": "DECLINED"},
            )
            self.assertEqual(dec_res.status_code, 200)
            self.assertEqual(dec_res.json()["email_notification"]["status"], "sent")

        # Verify application still saves cleanly even when SMTP fails
        with patch("email_service.smtplib.SMTP", side_effect=OSError("SMTP down")):
            res_fail_smtp = self.client.post(
                "/api/applications",
                headers={"Authorization": f"Bearer {token}"},
                json={"notes": "Saved even if SMTP is down"},
            )
            self.assertEqual(res_fail_smtp.status_code, 201)
            self.assertEqual(res_fail_smtp.json()["email_notifications"]["customer_confirmation"]["status"], "failed")

    def test_07_security_and_sensitive_data_controls(self):
        """Customers cannot call employee endpoints or send arbitrary emails; templates omit financial figures."""
        cust_email = f"sec_{self.run_tag}@example.com"
        with patch("email_service.smtplib.SMTP") as smtp_cls:
            mock_smtp = MagicMock()
            mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp.__exit__ = MagicMock(return_value=False)
            smtp_cls.return_value = mock_smtp

            reg = self.client.post(
                "/api/auth/register",
                json={"name": "Sec User", "email": cust_email, "password": "strongpassword"},
            )
            token = reg.json()["access_token"]

        # Unauthenticated send blocked
        unauth = self.client.post(
            "/api/notifications/send",
            json={"event_type": "application_submitted", "recipient_email": "x@example.com"},
        )
        self.assertEqual(unauth.status_code, 401)

        # Customer blocked from sending employee-only notifications
        cust_forbidden = self.client.post(
            "/api/notifications/send",
            headers={"Authorization": f"Bearer {token}"},
            json={"event_type": "final_decision", "recipient_email": "other@example.com"},
        )
        self.assertEqual(cust_forbidden.status_code, 403)

        # Email status endpoint never exposes credentials
        status_res = self.client.get("/api/email/status")
        self.assertEqual(status_res.status_code, 200)
        self.assertNotIn("test-app-password-16ch", status_res.text)

        # Rendered templates do not include sensitive financial fields
        _, text_body, html_body = email_service.render_template(
            "application_submitted",
            {"customer_name": "Sec User", "application_id": "APP-SEC1"},
        )
        self.assertNotIn("LIMIT_BAL", text_body)
        self.assertNotIn("BILL_AMT", html_body)


if __name__ == "__main__":
    unittest.main(verbosity=2)
