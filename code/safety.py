"""
safety.py - Safety layer for injection detection, PII handling, and output validation.
"""

import re
import logging
from typing import Tuple, List, Dict, Any
from config import (
    INJECTION_KEYWORDS,
    INJECTION_PATTERNS,
    MAX_ISSUE_LENGTH,
    PII_PATTERNS,
)

logger = logging.getLogger(__name__)


class SafetyLayer:
    """Handles injection detection, PII detection, and sanitization."""

    def __init__(self):
        """Initialize safety layer."""
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in PII_PATTERNS.items()
        }
        self.compiled_injection_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS
        ]

    def detect_prompt_injection(self, text: str) -> Tuple[bool, float, str]:
        """
        Detect prompt injection attempts in text.

        Returns:
            (is_injection, confidence, reason)
        """
        if not text:
            return False, 0.0, "Empty text"

        # Check length anomaly
        if len(text) > MAX_ISSUE_LENGTH:
            return True, 0.8, f"Anomalously long input ({len(text)} chars)"

        text_lower = text.lower()
        injection_score = 0.0
        reasons = []

        # Check for injection keywords
        keyword_matches = sum(1 for keyword in INJECTION_KEYWORDS if keyword in text_lower)
        if keyword_matches > 0:
            injection_score += 0.3 * min(keyword_matches / 5, 1.0)
            reasons.append(f"Found {keyword_matches} injection keywords")

        # Check for injection patterns
        pattern_matches = sum(1 for pattern in self.compiled_injection_patterns if pattern.search(text))
        if pattern_matches > 0:
            injection_score += 0.4
            reasons.append(f"Matched {pattern_matches} injection patterns")

        # Check for suspicious sequences
        suspicious_sequences = [
            "ignore all",
            "disregard",
            "new instructions",
            "system prompt",
            "you are now",
        ]
        for seq in suspicious_sequences:
            if seq in text_lower:
                injection_score += 0.2
                reasons.append(f"Found suspicious sequence: '{seq}'")

        # Check for escape attempts (excessive punctuation/symbols)
        symbol_count = sum(1 for c in text if c in "!@#$%^&*(){}[]<>")
        if symbol_count > 20:
            injection_score += 0.1
            reasons.append(f"Excessive symbols ({symbol_count})")

        is_injection = injection_score >= 0.5
        confidence = min(injection_score, 1.0)
        reason = " | ".join(reasons) if reasons else "No injection detected"

        if is_injection:
            logger.warning(f"Injection detected (confidence={confidence:.2f}): {reason}")

        return is_injection, confidence, reason

    def detect_pii(self, text: str) -> Tuple[bool, List[str], Dict[str, List[str]]]:
        """
        Detect PII in text.

        Returns:
            (has_pii, pii_types, matches_dict)
        """
        if not text:
            return False, [], {}

        matches_dict: Dict[str, List[str]] = {}
        pii_types = []

        for pii_type, pattern in self.compiled_patterns.items():
            matches = pattern.findall(text)
            if matches:
                matches_dict[pii_type] = matches
                pii_types.append(pii_type)
                logger.warning(f"Detected {pii_type}: {len(matches)} instances")

        has_pii = len(pii_types) > 0
        return has_pii, pii_types, matches_dict

    def sanitize_pii(self, text: str) -> str:
        """
        Replace PII with generic placeholders.

        Returns:
            Sanitized text
        """
        sanitized = text

        # Credit cards: show last 4 digits
        sanitized = re.sub(
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
            "****-****-****-XXXX",
            sanitized,
            flags=re.IGNORECASE,
        )

        # SSN: X##-##-####
        sanitized = re.sub(
            r"\b\d{3}-\d{2}-\d{4}\b",
            "XXX-XX-XXXX",
            sanitized,
            flags=re.IGNORECASE,
        )

        # Phone numbers
        sanitized = re.sub(
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "(XXX) XXX-XXXX",
            sanitized,
            flags=re.IGNORECASE,
        )

        # Emails
        sanitized = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "[EMAIL_REDACTED]",
            sanitized,
            flags=re.IGNORECASE,
        )

        # Zip codes
        sanitized = re.sub(
            r"\b\d{5}(-\d{4})?\b",
            "XXXXX",
            sanitized,
            flags=re.IGNORECASE,
        )

        # IP addresses
        sanitized = re.sub(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "X.X.X.X",
            sanitized,
            flags=re.IGNORECASE,
        )

        return sanitized

    def mask_pii_in_response(self, text: str) -> str:
        """
        Mask PII in response to prevent leaking sensitive info.

        Returns:
            Response with PII masked
        """
        return self.sanitize_pii(text)

    def validate_output_schema(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate output against expected schema.

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # Required fields
        required_fields = [
            "status",
            "product_area",
            "response",
            "justification",
            "request_type",
            "confidence_score",
            "risk_level",
        ]
        for field in required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")

        # Enum validations
        if "status" in data:
            if data["status"] not in ["replied", "escalated"]:
                errors.append(f"Invalid status: {data['status']}")

        if "request_type" in data:
            if data["request_type"] not in ["product_issue", "feature_request", "bug", "invalid"]:
                errors.append(f"Invalid request_type: {data['request_type']}")

        if "risk_level" in data:
            if data["risk_level"] not in ["low", "medium", "high", "critical"]:
                errors.append(f"Invalid risk_level: {data['risk_level']}")

        # Confidence score range
        if "confidence_score" in data:
            try:
                score = float(data["confidence_score"])
                if not (0.0 <= score <= 1.0):
                    errors.append(f"confidence_score out of range: {score}")
            except (ValueError, TypeError):
                errors.append(f"confidence_score not a number: {data['confidence_score']}")

        # PII in response
        if "response" in data and "pii_detected" in data:
            if data["pii_detected"]:
                has_pii, _, _ = self.detect_pii(data["response"])
                if has_pii:
                    errors.append("PII detected in response (should be masked)")

        return len(errors) == 0, errors

    def check_response_pii_safety(self, response: str, original_ticket: str) -> Tuple[bool, str]:
        """
        Check if response echoes back PII from original ticket.

        Returns:
            (is_safe, warning_message)
        """
        # Detect PII in original
        has_pii, pii_types, pii_matches = self.detect_pii(original_ticket)

        if not has_pii:
            return True, "No PII detected in original ticket"

        # Check if any PII appears in response
        response_lower = response.lower()
        violations = []

        for pii_type, matches in pii_matches.items():
            for match in matches:
                # Check if match (or significant portion of it) appears in response
                if match in response or match.replace("-", "") in response.replace("-", ""):
                    violations.append(f"{pii_type}: {match}")

        if violations:
            return False, f"PII echoed in response: {', '.join(violations)}"

        return True, "Response safely avoids PII"


def create_safety_layer() -> SafetyLayer:
    """Factory function to create SafetyLayer instance."""
    return SafetyLayer()
