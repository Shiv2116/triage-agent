"""
config.py - Configuration, thresholds, and constants.
"""

from pathlib import Path
from typing import Set
import os

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TICKETS_DIR = PROJECT_ROOT / "support_tickets"
INPUT_CSV = TICKETS_DIR / "support_tickets.csv"
OUTPUT_CSV = TICKETS_DIR / "output.csv"

# Corpus paths
DEVPLATFORM_CORPUS = DATA_DIR / "devplatform"
CLAUDE_CORPUS = DATA_DIR / "claude"
VISA_CORPUS = DATA_DIR / "visa"
API_SPECS_DIR = DATA_DIR / "api_specs"

# ============================================================================
# LLM SETTINGS
# ============================================================================

# Model selection - DeepSeek V4 Flash (free)
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = 0.0  # Deterministic
LLM_MAX_TOKENS = 2000
LLM_TIMEOUT = 30  # seconds
LLM_RETRIES = 2

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"

# ============================================================================
# RETRIEVAL SETTINGS
# ============================================================================

BM25_TOP_K = 10  # Initial BM25 retrieval
RERANK_TOP_K = 3  # After LLM re-ranking

# Hybrid rerank mixing weight: weight given to LLM score vs normalized BM25 score.
# final_score = RERANK_ALPHA * llm_score + (1-RERANK_ALPHA) * bm25_norm
RERANK_ALPHA = float(os.getenv("RERANK_ALPHA", "0.6"))

# ============================================================================
# INJECTION DETECTION
# ============================================================================

INJECTION_KEYWORDS: Set[str] = {
    "ignore",
    "forget",
    "override",
    "new instructions",
    "change behavior",
    "forget your",
    "system prompt",
    "you are now",
    "pretend you're",
    "act as",
    "bypass",
    "disable safety",
    "don't follow",
    "disregard",
    "ignore all previous",
    "completely ignore",
    "override this",
    "switch to",
    "new role",
    "respond as",
    "behave like",
}

# Additional injection patterns (regex-like)
INJECTION_PATTERNS = [
    r"ignore.*instructions",
    r"new.*instructions",
    r"you are now",
    r"pretend.*you",
    r"act as.*different",
    r"switch.*role",
]

MAX_ISSUE_LENGTH = 5000  # Flag anomalously long inputs
MIN_INJECTION_SCORE = 0.7  # Confidence threshold for marking as injection

# ============================================================================
# PII DETECTION
# ============================================================================

# Patterns for PII detection
PII_PATTERNS = {
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "zip_code": r"\b\d{5}(-\d{4})?\b",
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}

# ============================================================================
# CONFIDENCE & RISK THRESHOLDS
# ============================================================================

CONFIDENCE_ESCALATION_THRESHOLD = 0.30  # If confidence < this, escalate
CONFIDENCE_HIGH = 0.85  # High confidence threshold
CONFIDENCE_MEDIUM = 0.60  # Medium confidence threshold
CONFIDENCE_LOW = 0.30  # Low confidence threshold

# Risk level thresholds
RISK_CRITICAL_KEYWORDS = [
    "fraud",
    "identity theft",
    "lawsuit",
    "legal action",
    "complaint",
    "investigation",
]

RISK_HIGH_KEYWORDS = [
    "refund",
    "money back",
    "credit card",
    "account locked",
    "unauthorized",
    "billing",
]

RISK_MEDIUM_KEYWORDS = [
    "bug",
    "error",
    "issue",
    "problem",
    "broken",
    "not working",
]

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

# Max tickets to process (safety limit)
MAX_TICKETS = 10000

# Timeout per ticket (seconds)
TICKET_TIMEOUT = 30

# ============================================================================
# TOOL SPECIFICATIONS
# ============================================================================

# Tool prerequisites (e.g., must verify identity before refund)
TOOL_PREREQUISITES = {
    "issue_refund": ["verify_identity"],
    "lock_account": [],
    "reset_password": [],
    "verify_identity": [],
}

# ============================================================================
# OUTPUT COLUMNS
# ============================================================================

CSV_COLUMNS = [
    "status",
    "product_area",
    "response",
    "justification",
    "request_type",
    "confidence_score",
    "source_documents",
    "risk_level",
    "pii_detected",
    "language",
    "actions_taken",
]

# ============================================================================
# LANGUAGE DETECTION
# ============================================================================

# language lookups removed (unused); prefer `langdetect` or explicit LLM-detected codes


# ============================================================================
# PRODUCT AREAS (ROUTING)
# ============================================================================

PRODUCT_KEYWORDS = {
    "DevPlatform": [
        "devplatform",
        "test",
        "interview",
        "assessment",
        "hire",
        "candidate",
        "skill",
    ],
    "Claude": [
        "claude",
        "ai",
        "model",
        "api",
        "tokens",
        "conversation",
    ],
    "Visa": [
        "visa",
        "payment",
        "transaction",
        "card",
        "merchant",
        "dispute",
    ],
}

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Prompts moved to `prompts.py` (import directly from modules that need them).
