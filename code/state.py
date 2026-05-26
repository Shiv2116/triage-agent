"""
state.py - Ticket state management.

Maintains in-memory state for each ticket during processing.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for tickets."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IdentityStatus(Enum):
    """Identity verification status."""
    UNVERIFIED = "unverified"
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"


@dataclass
class ConversationMessage:
    """Single message in conversation."""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class TicketState:
    """In-memory state for a ticket during processing."""

    ticket_id: str
    conversation: List[ConversationMessage] = field(default_factory=list)
    subject: Optional[str] = None
    company_hint: Optional[str] = None

    # Extracted information
    product_detected: Optional[str] = None
    inferred_product: Optional[str] = None
    identity_verified: IdentityStatus = field(default_factory=lambda: IdentityStatus.UNVERIFIED)
    customer_id: Optional[str] = None
    transaction_id: Optional[str] = None
    request_type: Optional[str] = None

    # Safety flags
    pii_detected: bool = False
    pii_types: List[str] = field(default_factory=list)
    injection_detected: bool = False
    injection_confidence: float = 0.0

    # Risk assessment
    risk_level: RiskLevel = field(default_factory=lambda: RiskLevel.LOW)
    risk_flags: List[str] = field(default_factory=list)

    # Processing state
    previous_actions: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_documents: List[str] = field(default_factory=list)
    confidence_score: float = 0.5

    # Language
    language: str = "en"

    # Error tracking
    errors: List[str] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        """Add message to conversation."""
        self.conversation.append(ConversationMessage(role=role, content=content))

    def get_conversation_text(self) -> str:
        """Get full conversation as single text."""
        lines = []
        for msg in self.conversation:
            lines.append(f"{msg.role}: {msg.content}")
        return "\n".join(lines)

    def add_risk_flag(self, flag: str) -> None:
        """Add risk flag."""
        if flag not in self.risk_flags:
            self.risk_flags.append(flag)
        logger.info(f"Risk flag added: {flag}")

    def update_risk_level(self) -> None:
        """Update risk level based on flags."""
        critical_flags = [
            "fraud",
            "identity_theft",
            "legal_threat",
            "account_takeover",
        ]
        high_flags = [
            "refund_request",
            "large_amount",
            "suspicious_pattern",
            "multiple_disputes",
        ]

        if any(flag in self.risk_flags for flag in critical_flags):
            self.risk_level = RiskLevel.CRITICAL
        elif any(flag in self.risk_flags for flag in high_flags):
            self.risk_level = RiskLevel.HIGH
        elif any(flag in self.risk_flags for flag in ["bug", "error"]):
            self.risk_level = RiskLevel.MEDIUM
        else:
            self.risk_level = RiskLevel.LOW

    def add_pii_detection(self, pii_types: List[str]) -> None:
        """Record PII detection."""
        if pii_types:
            self.pii_detected = True
            self.pii_types.extend(pii_types)
            self.pii_types = list(set(self.pii_types))  # deduplicate
            logger.warning(f"PII detected: {self.pii_types}")

    def add_previous_action(self, action: str, parameters: Dict[str, Any]) -> None:
        """Record a previous action taken."""
        self.previous_actions.append({
            "action": action,
            "parameters": parameters,
        })

    def add_retrieved_document(self, doc_path: str) -> None:
        """Track retrieved documents."""
        if doc_path not in self.retrieved_documents:
            self.retrieved_documents.append(doc_path)

    def add_error(self, error: str) -> None:
        """Record an error during processing."""
        self.errors.append(error)
        logger.error(f"Ticket {self.ticket_id} error: {error}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "ticket_id": self.ticket_id,
            "conversation_length": len(self.conversation),
            "product_detected": self.product_detected,
            "pii_detected": self.pii_detected,
            "injection_detected": self.injection_detected,
            "risk_level": self.risk_level.value,
            "identity_verified": self.identity_verified.value,
            "confidence_score": self.confidence_score,
            "language": self.language,
            "errors": self.errors,
        }


class TicketStateManager:
    """Manages in-memory state for all tickets."""

    def __init__(self):
        """Initialize state manager."""
        self.states: Dict[str, TicketState] = {}

    def create_state(self, ticket_id: str) -> TicketState:
        """Create new ticket state."""
        if ticket_id in self.states:
            logger.warning(f"State already exists for ticket {ticket_id}")
        state = TicketState(ticket_id=ticket_id)
        self.states[ticket_id] = state
        return state

    def get_state(self, ticket_id: str) -> Optional[TicketState]:
        """Get ticket state."""
        return self.states.get(ticket_id)

    def delete_state(self, ticket_id: str) -> None:
        """Delete ticket state after processing."""
        if ticket_id in self.states:
            del self.states[ticket_id]

    def get_all_states(self) -> Dict[str, TicketState]:
        """Get all states."""
        return dict(self.states)

    def clear_all(self) -> None:
        """Clear all states."""
        self.states.clear()

    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of all states."""
        return {
            "total_tickets": len(self.states),
            "tickets": {
                ticket_id: state.to_dict()
                for ticket_id, state in self.states.items()
            }
        }
