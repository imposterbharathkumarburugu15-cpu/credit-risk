"""Rule-based Predefined FAQ Chatbot for CreditRisk-X.

Provides instant, strictly controlled answers regarding CreditRisk-X platform concepts,
cost assumptions, model architecture, decision policies, and email automation guidelines.
Uses deterministic intent matching without external LLM calls or exposure of private customer records.
"""
import re
from typing import Any, Dict, List, Tuple

FAQ_KNOWLEDGE_BASE: List[Dict[str, Any]] = [
    {
        "intent": "cost_matrix",
        "keywords": ["cost", "matrix", "fp cost", "fn cost", "false positive", "false negative", "dollar", "assumption"],
        "patterns": [r"cost", r"false\s*positive", r"false\s*negative", r"\$500", r"\$5000", r"financial\s*loss"],
        "response": (
            "In CreditRisk-X, our baseline cost matrix models asymmetric lending risk:\n"
            "• False Positive (FP Cost = $500): Rejecting or flagging a creditworthy borrower loses processing overhead and future interest income.\n"
            "• False Negative (FN Cost = $5,000): Approving a borrower who defaults results in charged-off principal balance and recovery expenses.\n"
            "Because an FN is ~10x more costly than an FP, standard 0.50 probability thresholds produce severe financial losses. "
            "Our cost-optimization engine dynamically determines the threshold that minimizes overall financial loss."
        ),
    },
    {
        "intent": "threshold_policy",
        "keywords": ["threshold", "cutoff", "0.35", "optimal threshold", "decision threshold", "tune"],
        "patterns": [r"threshold", r"cutoff", r"0\.35", r"0\.50", r"optimal"],
        "response": (
            "CreditRisk-X uses cost-sensitive threshold optimization on validation data:\n"
            "• Standard 0.50 threshold assumes equal costs for FP and FN, which is inappropriate for credit risk.\n"
            "• The project reported baseline threshold is 0.35 (and empirically ~0.14-0.16 on validation data under $5,000 FN / $500 FP costs).\n"
            "• Operating at the cost-optimal threshold significantly reduces total bank losses by catching defaults before they happen, "
            "saving up to 25-30% compared to a naive 0.50 cutoff."
        ),
    },
    {
        "intent": "models",
        "keywords": ["model", "xgboost", "logistic regression", "svm", "random forest", "decision tree", "algorithm"],
        "patterns": [r"model", r"algorithm", r"xgboost", r"random\s*forest", r"logistic", r"svm"],
        "response": (
            "CreditRisk-X trains and benchmarks five distinct machine learning models on 30,000 records:\n"
            "1. XGBoost (Primary): Gradient boosted trees with AUC-PR early stopping; achieves ~0.783 ROC-AUC and highest PR-AUC (~0.558).\n"
            "2. Random Forest: Ensemble of 300 bagged trees for non-linear stability.\n"
            "3. Decision Tree: Constrained to depth 4 to yield transparent, human-auditable if-then rules.\n"
            "4. Linear SVM: Calibrated via Platt scaling for reliable probability outputs.\n"
            "5. Logistic Regression: Standardized linear baseline showing coefficient weights."
        ),
    },
    {
        "intent": "features",
        "keywords": ["feature", "engineered", "input", "raw", "columns", "utilization", "slope", "trend"],
        "patterns": [r"feature", r"input", r"utilization", r"engineered", r"variables"],
        "response": (
            "The model utilizes 43 total features:\n"
            "• 23 Raw Features: Demographic data (Age, Sex, Education, Marriage), Credit limit (LIMIT_BAL), "
            "6 months of payment history (PAY_0 through PAY_6), 6 bill statements (BILL_AMT1-6), and 6 payment amounts (PAY_AMT1-6).\n"
            "• 20 Engineered Features: Rebuilt using historical trend analysis, including latest and 6-month utilization, "
            "utilization trend slope, average/peak bill balances, payment volatility, consecutive late months streak, and max delay severity."
        ),
    },
    {
        "intent": "explainability",
        "keywords": ["explain", "shap", "reason", "waterfall", "contribution", "why", "interpret"],
        "patterns": [r"explain", r"shap", r"interpret", r"why", r"reason"],
        "response": (
            "CreditRisk-X provides local TreeSHAP feature attributions for every borrower:\n"
            "• It calculates exact log-odds contributions for each feature, showing whether a factor pushes risk UP (red) or DOWN (green).\n"
            "• Features like recent payment delays (PAY_0) and high credit utilization consistently carry the largest attribution weights.\n"
            "• Important regulatory note: SHAP explains model mechanics; it does NOT constitute legal proof of causality or protected-attribute fairness."
        ),
    },
    {
        "intent": "decision_bands",
        "keywords": ["band", "risk band", "classification", "high risk", "medium risk", "low risk", "recommendation"],
        "patterns": [r"risk\s*band", r"classification", r"high\s*risk", r"medium\s*risk", r"low\s*risk"],
        "response": (
            "Our three-tier risk policy operates as follows:\n"
            "• HIGH RISK (p ≥ threshold): Flagged. Automated loan decline or mandatory referral to senior underwriter for manual review.\n"
            "• MEDIUM RISK (threshold/2 ≤ p < threshold): Conditional Approval with reduced credit limit, mandatory autopay, and 60-day monitoring.\n"
            "• LOW RISK (p < threshold/2): Standard automated approval on standard bank terms."
        ),
    },
    {
        "intent": "email_policy",
        "keywords": ["email", "campaign", "notification", "automation", "compliance", "template"],
        "patterns": [r"email", r"campaign", r"template", r"notification", r"smtp"],
        "response": (
            "Email Automation Guidelines in CreditRisk-X:\n"
            "• Ethical Rule: An ML prediction is NOT proof of customer default. We never tell a customer they have defaulted based solely on an ML prediction.\n"
            "• Never expose internal credit risk scores, SHAP values, or internal bank notes in customer communications.\n"
            "• Supported templates: Payment Attention (requires actual overdue payment records), Account Review Notification, Application Update, and General Customer Notification.\n"
            "• Authorization: Campaigns are NEVER sent automatically on CSV upload. An authorized employee (Loan Officer or Admin) must explicitly authorize the campaign."
        ),
    },
    {
        "intent": "greeting",
        "keywords": ["hello", "hi", "hey", "help", "who are you", "start"],
        "patterns": [r"^hi\b", r"^hello\b", r"^hey\b", r"help", r"who\s*are\s*you"],
        "response": (
            "Hello! I am the CreditRisk-X Assistant. I can help answer questions about:\n"
            "1. Cost Matrix assumptions (FP $500 vs FN $5,000)\n"
            "2. Threshold Optimization (0.35 baseline vs cost-optimal curves)\n"
            "3. ML Models & Performance (XGBoost, Random Forest, etc.)\n"
            "4. Feature Engineering (43 total features)\n"
            "5. Explainability & SHAP values\n"
            "6. Risk Bands (Low, Medium, High)\n"
            "7. Email Automation & Compliance Policies\n\n"
            "How can I assist your lending analysis today?"
        ),
    },
]

FALLBACK_RESPONSE = (
    "I can only assist with questions regarding the CreditRisk-X platform. "
    "Please ask about our cost matrix ($500 FP / $5,000 FN), threshold optimization, "
    "XGBoost model performance, 43 engineered features, SHAP explainability, risk bands, "
    "or email automation compliance rules."
)


def answer_query(user_message: str) -> Dict[str, Any]:
    """
    Process incoming user message and return structured chatbot response.
    Never exposes private customer data.
    """
    cleaned = (user_message or "").strip().lower()
    if not cleaned:
        return {
            "intent": "empty",
            "reply": "Please enter a question about CreditRisk-X risk models or policies.",
        }

    # Match intents by pattern or keywords
    best_intent = None
    best_score = 0
    selected_response = FALLBACK_RESPONSE

    for entry in FAQ_KNOWLEDGE_BASE:
        score = 0
        # Check regex patterns
        for pattern in entry.get("patterns", []):
            if re.search(pattern, cleaned, re.IGNORECASE):
                score += 3

        # Check keywords
        for kw in entry.get("keywords", []):
            if kw in cleaned:
                score += 1

        if score > best_score:
            best_score = score
            best_intent = entry["intent"]
            selected_response = entry["response"]

    if best_score == 0:
        best_intent = "fallback"
        selected_response = FALLBACK_RESPONSE

    return {
        "intent": best_intent,
        "query": user_message,
        "reply": selected_response,
        "confidence": "high" if best_score >= 3 else ("medium" if best_score > 0 else "low"),
    }
