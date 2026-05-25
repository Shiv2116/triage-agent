"""
schemas.py - Pydantic models for ticket processing and output validation.

Defines strict schemas for:
- Ticket input
- Ticket state
- LLM output
- Final CSV output
"""

from typing import Literal, Optional, List
from pydantic import BaseModel, Field, validator
import json


class ToolCall(BaseModel):
    """Represents a single tool invocation."""
    action: str = Field(..., description="Tool action name")
    parameters: dict = Field(default_factory=dict, description="Action parameters")


class TicketResponse(BaseModel):
    """Complete response for a support ticket."""
    status: Literal["replied", "escalated"] = Field(..., description="Reply or escalate")
    product_area: str = Field(..., description="Relevant support category")
    response: str = Field(..., description="User-facing answer")
    justification: str = Field(..., description="Explanation of decision")
    request_type: Literal["product_issue", "feature_request", "bug", "invalid"] = Field(
        ..., description="Type of request"
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence 0.0-1.0")
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        ..., description="Risk assessment"
    )
    pii_detected: bool = Field(default=False, description="PII present in ticket")
    language: str = Field(default="en", description="ISO 639-1 language code")
    source_documents: List[str] = Field(default_factory=list, description="Corpus file paths")
    actions_taken: List[ToolCall] = Field(default_factory=list, description="Tool calls")

    @validator("confidence_score")
    def validate_confidence(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence_score must be between 0.0 and 1.0")
        return round(v, 2)

    @validator("response")
    def validate_response_not_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("response cannot be empty")
        return v

    @validator("product_area")
    def validate_product_area_not_empty(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("product_area cannot be empty")
        return v

    def to_csv_row(self) -> dict:
        """Convert to CSV row format."""
        return {
            "status": self.status,
            "product_area": self.product_area,
            "response": self.response,
            "justification": self.justification,
            "request_type": self.request_type,
            "confidence_score": self.confidence_score,
            "source_documents": "|".join(self.source_documents),
            "risk_level": self.risk_level,
            "pii_detected": str(self.pii_detected).lower(),
            "language": self.language,
            "actions_taken": json.dumps([t.dict() for t in self.actions_taken]),
        }


class ConversationMessage(BaseModel):
    """Single message in conversation history."""
    role: Literal["user", "assistant", "system"]
    content: str


class TicketInput(BaseModel):
    """Input ticket from CSV."""
    issue: str  # JSON string
    subject: Optional[str] = None
    company: Optional[Literal["DevPlatform", "Claude", "Visa"]] = None

    @validator("issue", pre=True)
    def parse_issue_json(cls, v):
        """Parse JSON string to validate it's valid JSON."""
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if not isinstance(parsed, list):
                    raise ValueError("issue must be JSON array")
                return v
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid JSON in issue: {e}")
        return v


class TicketRow(BaseModel):
    """Single row from support_tickets.csv."""
    index: int
    issue: str
    subject: Optional[str]
    company: Optional[str]


class EscalationResponse(BaseModel):
    """Standard escalation response."""
    status: Literal["escalated"] = "escalated"
    product_area: str
    response: str
    justification: str
    request_type: Literal["invalid"] = "invalid"
    confidence_score: float = 0.1
    risk_level: Literal["critical"] = "critical"
    pii_detected: bool = False
    language: str = "en"
    source_documents: List[str] = Field(default_factory=list)
    actions_taken: List[ToolCall] = Field(default_factory=list)

    def to_ticket_response(self) -> TicketResponse:
        """Convert to TicketResponse."""
        return TicketResponse(**self.dict())


class LLMOutput(BaseModel):
    """Raw LLM output (before validation)."""
    status: Optional[str] = None
    product_area: Optional[str] = None
    response: Optional[str] = None
    justification: Optional[str] = None
    request_type: Optional[str] = None
    confidence_score: Optional[float] = None
    risk_level: Optional[str] = None
    pii_detected: Optional[bool] = None
    language: Optional[str] = None
    source_documents: Optional[list] = None
    actions_taken: Optional[list] = None
