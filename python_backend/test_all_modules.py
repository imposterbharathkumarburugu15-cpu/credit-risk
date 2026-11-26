"""Comprehensive Test Suite for CreditRisk-X Python Backend Modules.

Verifies:
1. Module imports and configurations
2. Kaggle XGBoost model loading
3. Prediction engine (both raw 23 features and full 43 features)
4. Decision engine risk banding (LOW, MEDIUM, HIGH)
5. Cost efficiency engine (FP $500, FN $5,000)
6. Threshold optimization on validation data
7. TreeSHAP explainability
8. Analytics metrics (ROC-AUC, PR-AUC, Brier score, log loss, portfolio)
9. Rule-based FAQ chatbot
10. MySQL connectivity and record insertion
11. FastAPI test client routes (/api/health, /api/predict, /api/cost/calculate, /api/threshold/optimize, /api/chatbot)
"""
import json
import sys
import unittest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import analytics
import chatbot
import config
import cost
import database
import decision
import explainability
import model
import optimization
import prediction
from main import app


class TestCreditRiskXBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n--- Initializing CreditRisk-X Test Suite ---")
        # Ensure model artifacts are loaded
        models, scores, meta = model.load_artifacts()
        cls.models = models
        cls.scores = scores
        cls.meta = meta
        cls.client = TestClient(app)

    def test_01_model_loading(self):
        """Test that Kaggle XGBoost model and companion models load properly."""
        self.assertTrue(model.is_model_available())
        xgb = model.get_model("xgboost")
        self.assertIsNotNone(xgb)
        # Check that booster feature names match expected 43 features
        feature_names = model.get_feature_manifest()
        self.assertEqual(len(feature_names), 43)
        self.assertIn("LIMIT_BAL", feature_names)
        self.assertIn("utilization_latest", feature_names)
        print("✓ Test 01: Model loading and feature manifest validated (43 features).")

    def test_02_prediction_raw_and_full(self):
        """Test prediction with raw 23 features and with full 43 features."""
        # Raw 23 features test sample
        sample_raw = {
            "LIMIT_BAL": 50000.0,
            "SEX": 2,
            "EDUCATION": 2,
            "MARRIAGE": 1,
            "AGE": 35,
            "PAY_0": 2,
            "PAY_2": 2,
            "PAY_3": 0,
            "PAY_4": 0,
            "PAY_5": 0,
            "PAY_6": 0,
            "BILL_AMT1": 20000.0,
            "BILL_AMT2": 19000.0,
            "BILL_AMT3": 18000.0,
            "BILL_AMT4": 17000.0,
            "BILL_AMT5": 16000.0,
            "BILL_AMT6": 15000.0,
            "PAY_AMT1": 1000.0,
            "PAY_AMT2": 1000.0,
            "PAY_AMT3": 1000.0,
            "PAY_AMT4": 1000.0,
            "PAY_AMT5": 1000.0,
            "PAY_AMT6": 1000.0,
        }

        res_raw = prediction.predict_probability(sample_raw)
        self.assertIn("default_probability", res_raw)
        self.assertGreaterEqual(res_raw["default_probability"], 0.0)
        self.assertLessEqual(res_raw["default_probability"], 1.0)
        self.assertEqual(res_raw["feature_count"], 43)

        # Full features dataframe test
        df_full = res_raw["features_df"]
        res_full = prediction.predict_probability(df_full.iloc[0].to_dict())
        self.assertEqual(res_raw["default_probability"], res_full["default_probability"])
        print(f"✓ Test 02: Prediction engine validated (P(Default)={res_raw['default_probability']}).")

    def test_03_decision_engine(self):
        """Test decision risk banding (LOW, MEDIUM, HIGH) with 0.35 threshold."""
        threshold = 0.35

        # High risk: proba >= 0.35
        dec_high = decision.make_decision(0.45, threshold=threshold)
        self.assertEqual(dec_high["risk_classification"], "HIGH")
        self.assertTrue(dec_high["flagged"])

        # Medium risk: 0.175 <= proba < 0.35
        dec_med = decision.make_decision(0.25, threshold=threshold)
        self.assertEqual(dec_med["risk_classification"], "MEDIUM")
        self.assertFalse(dec_med["flagged"])

        # Low risk: proba < 0.175
        dec_low = decision.make_decision(0.08, threshold=threshold)
        self.assertEqual(dec_low["risk_classification"], "LOW")
        self.assertFalse(dec_low["flagged"])
        print("✓ Test 03: Decision engine risk banding verified across all tiers.")

    def test_04_cost_engine(self):
        """Test FP ($500) and FN ($5,000) cost calculations and comparisons."""
        calc = cost.compute_classification_cost(fp_count=10, fn_count=5, fp_cost=500.0, fn_cost=5000.0)
        expected_total = 10 * 500.0 + 5 * 5000.0
        self.assertEqual(calc["total_cost"], expected_total)
        self.assertEqual(calc["total_fp_cost"], 5000.0)
        self.assertEqual(calc["total_fn_cost"], 25000.0)

        # Comparison test
        comp = cost.compare_policy_costs(
            baseline_fp=20, baseline_fn=10,
            optimized_fp=30, optimized_fn=4,
            fp_cost=500.0, fn_cost=5000.0
        )
        base_cost = 20 * 500.0 + 10 * 5000.0  # 60,000
        opt_cost = 30 * 500.0 + 4 * 5000.0    # 35,000
        self.assertEqual(comp["cost_reduction_amount"], base_cost - opt_cost)
        self.assertAlmostEqual(comp["cost_reduction_percentage"], (25000 / 60000) * 100, places=1)
        print(f"✓ Test 04: Cost engine validated (Base=${base_cost:,.0f}, Opt=${opt_cost:,.0f}, Savings=${comp['cost_reduction_amount']:,.0f}).")

    def test_05_threshold_optimization(self):
        """Test threshold optimization on validation data."""
        opt_res = optimization.optimize_threshold(model_name="xgboost", fp_cost=500.0, fn_cost=5000.0)
        self.assertEqual(opt_res["dataset_used"], "validation")
        self.assertIn("optimal_threshold", opt_res)
        self.assertIn("minimum_cost", opt_res)
        self.assertIn("threshold_curve", opt_res)
        self.assertGreater(len(opt_res["threshold_curve"]), 50)
        self.assertGreater(opt_res["savings_amount"], 0)
        print(f"✓ Test 05: Threshold optimization validated (Optimal threshold: {opt_res['optimal_threshold']}, Savings: ${opt_res['savings_amount']:,.0f}).")

    def test_06_explainability_shap(self):
        """Test TreeSHAP feature explanation calculation and directionality."""
        sample = {
            "LIMIT_BAL": 20000.0,
            "SEX": 1,
            "EDUCATION": 2,
            "MARRIAGE": 1,
            "AGE": 25,
            "PAY_0": 2,
            "PAY_2": 2,
            "PAY_3": 0,
            "PAY_4": 0,
            "PAY_5": 0,
            "PAY_6": 0,
            "BILL_AMT1": 15000.0,
            "BILL_AMT2": 14000.0,
            "BILL_AMT3": 13000.0,
            "BILL_AMT4": 12000.0,
            "BILL_AMT5": 11000.0,
            "BILL_AMT6": 10000.0,
            "PAY_AMT1": 0.0,
            "PAY_AMT2": 500.0,
            "PAY_AMT3": 500.0,
            "PAY_AMT4": 500.0,
            "PAY_AMT5": 500.0,
            "PAY_AMT6": 500.0,
        }
        expl = explainability.explain_prediction(sample, top_n=6)
        self.assertEqual(len(expl["top_features"]), 6)
        first_feat = expl["top_features"][0]
        self.assertIn("feature", first_feat)
        self.assertIn("label", first_feat)
        self.assertIn("contribution", first_feat)
        self.assertIn("direction", first_feat)
        self.assertIn("disclaimer", expl)
        print(f"✓ Test 06: TreeSHAP explainability validated (Top feature: {first_feat['label']}, Direction: {first_feat['direction']}).")

    def test_07_analytics_engine(self):
        """Test ROC-AUC, PR-AUC, Brier score, and log loss calculation."""
        y_true = [0, 0, 1, 1, 0, 1, 0, 1, 0, 0]
        y_proba = [0.1, 0.2, 0.8, 0.9, 0.15, 0.75, 0.3, 0.85, 0.05, 0.4]
        metrics = analytics.evaluate_model_performance(y_true, y_proba)
        self.assertGreater(metrics["roc_auc"], 0.9)
        self.assertGreater(metrics["pr_auc"], 0.8)
        self.assertLess(metrics["brier_score"], 0.1)

        # Portfolio distribution test
        portfolio = analytics.analyze_portfolio([0.05, 0.10, 0.25, 0.45, 0.60], threshold=0.35)
        self.assertEqual(portfolio["total_applications"], 5)
        self.assertEqual(portfolio["high_risk_count"], 2)
        self.assertEqual(portfolio["medium_risk_count"], 1)
        self.assertEqual(portfolio["low_risk_count"], 2)
        print("✓ Test 07: Analytics and portfolio metrics validated.")

    def test_08_chatbot(self):
        """Test rule-based deterministic FAQ chatbot."""
        q1 = chatbot.answer_query("What is the cost of false negatives?")
        self.assertEqual(q1["intent"], "cost_matrix")
        self.assertIn("$5,000", q1["reply"])

        q2 = chatbot.answer_query("Explain the threshold policy")
        self.assertEqual(q2["intent"], "threshold_policy")
        self.assertIn("0.35", q2["reply"])

        q3 = chatbot.answer_query("How does SHAP explainability work?")
        self.assertEqual(q3["intent"], "explainability")
        self.assertIn("TreeSHAP", q3["reply"])

        q4 = chatbot.answer_query("What are the email compliance rules?")
        self.assertEqual(q4["intent"], "email_policy")
        self.assertIn("never tell a customer they have defaulted", q4["reply"].lower())

        q_fallback = chatbot.answer_query("What is the weather in Tokyo?")
        self.assertEqual(q_fallback["intent"], "fallback")
        print("✓ Test 08: FAQ Chatbot intent matching and responses validated.")

    def test_09_database_connectivity(self):
        """Test MySQL connectivity and record saving."""
        client = database.get_client()
        if client is None:
            self.assertIsNone(database.save_prediction_record({"test_run": True}))
            self.assertEqual(database.get_prediction_history(limit=5), [])
            self.skipTest("MySQL server not accessible with current MYSQL_* env settings")

        db = database.get_db()
        self.assertIsNotNone(db)

        # Save test prediction record
        rec_id = database.save_prediction_record({
            "test_run": True,
            "default_probability": 0.245,
            "band": "MEDIUM",
            "decision": "Approve with monitoring",
        })
        self.assertIsNotNone(rec_id)

        history = database.get_prediction_history(limit=5)
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)
        print(f"✓ Test 09: MySQL integration tested (Inserted test record ID: {rec_id}).")

    def test_10_api_endpoints(self):
        """Test FastAPI endpoints via TestClient."""
        # 1. Health
        res_health = self.client.get("/api/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "ok")
        self.assertTrue(res_health.json()["model_loaded"])

        # 2. Predict
        res_predict = self.client.post("/api/predict", json={
            "LIMIT_BAL": 50000.0, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30,
            "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
            "BILL_AMT1": 5000.0, "BILL_AMT2": 5000.0, "BILL_AMT3": 5000.0,
            "BILL_AMT4": 5000.0, "BILL_AMT5": 5000.0, "BILL_AMT6": 5000.0,
            "PAY_AMT1": 1000.0, "PAY_AMT2": 1000.0, "PAY_AMT3": 1000.0,
            "PAY_AMT4": 1000.0, "PAY_AMT5": 1000.0, "PAY_AMT6": 1000.0
        })
        self.assertEqual(res_predict.status_code, 200)
        p_json = res_predict.json()
        self.assertIn("default_probability", p_json)
        self.assertIn("band", p_json)
        self.assertIn("reasons", p_json)

        # 3. Cost calculate
        res_cost = self.client.post("/api/cost/calculate", json={"fp_count": 5, "fn_count": 2})
        self.assertEqual(res_cost.status_code, 200)
        self.assertEqual(res_cost.json()["total_cost"], 5 * 500.0 + 2 * 5000.0)

        # 4. Threshold optimize
        res_opt = self.client.post("/api/threshold/optimize", json={"model_name": "xgboost"})
        self.assertEqual(res_opt.status_code, 200)
        self.assertIn("optimal_threshold", res_opt.json())

        # 5. Chatbot
        res_chat = self.client.post("/api/chatbot", json={"message": "What is the cost matrix?"})
        self.assertEqual(res_chat.status_code, 200)
        self.assertEqual(res_chat.json()["intent"], "cost_matrix")

        # 6. Predict batch (separating Table 1 and Table 2)
        batch_payload = {
            "records": [
                {
                    "customer_id": "CUST-001",
                    "customer_name": "Marcus Vance",
                    "email": "marcus@example.com",
                    "application_id": "APP-001",
                    "LIMIT_BAL": 20000.0, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
                    "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
                    "BILL_AMT1": 3913.0, "BILL_AMT2": 3102.0, "BILL_AMT3": 689.0,
                    "BILL_AMT4": 0.0, "BILL_AMT5": 0.0, "BILL_AMT6": 0.0,
                    "PAY_AMT1": 0.0, "PAY_AMT2": 689.0, "PAY_AMT3": 0.0,
                    "PAY_AMT4": 0.0, "PAY_AMT5": 0.0, "PAY_AMT6": 0.0,
                },
                {
                    "customer_id": "CUST-002",
                    "customer_name": "Elena Rostova",
                    "email": "elena@example.com",
                    "application_id": "APP-002",
                    "LIMIT_BAL": 120000.0, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 2, "AGE": 26,
                    "PAY_0": -1, "PAY_2": 2, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 2,
                    "BILL_AMT1": 2682.0, "BILL_AMT2": 1725.0, "BILL_AMT3": 2682.0,
                    "BILL_AMT4": 3272.0, "BILL_AMT5": 3455.0, "BILL_AMT6": 3261.0,
                    "PAY_AMT1": 0.0, "PAY_AMT2": 1000.0, "PAY_AMT3": 1000.0,
                    "PAY_AMT4": 1000.0, "PAY_AMT5": 0.0, "PAY_AMT6": 2000.0,
                }
            ],
            "threshold": 0.35,
        }
        res_batch = self.client.post("/api/predict/batch", json=batch_payload)
        self.assertEqual(res_batch.status_code, 200)
        b_json = res_batch.json()
        self.assertIn("table_predicted_default", b_json)
        self.assertIn("table_predicted_non_default", b_json)
        print("✓ Test 10: FastAPI endpoints (/health, /predict, /cost, /threshold, /chatbot, /predict/batch) all validated.")


if __name__ == "__main__":
    unittest.main()
