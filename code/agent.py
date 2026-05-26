"""
agent.py - Core orchestration logic for ticket processing.

Coordinates between retriever, safety layer, LLM, and state management.
"""

import logging
from typing import Optional, List, Tuple

from schemas import TicketResponse, LLMOutput
from state import TicketState, RiskLevel
from safety import SafetyLayer
from retriever import BM25Retriever
from llm_client import LLMClient
from config import (
    RERANK_TOP_K,
    CONFIDENCE_ESCALATION_THRESHOLD,
    RISK_CRITICAL_KEYWORDS,
    RISK_HIGH_KEYWORDS,
    PRODUCT_KEYWORDS,
)
from tool_executor import create_tool_executor

logger = logging.getLogger(__name__)


class SupportAgent:
    """Main support triage agent."""

    def __init__(
        self,
        retriever: BM25Retriever,
        llm_client: LLMClient,
        safety_layer: SafetyLayer,
        tool_executor=None,
    ):
        """
        Initialize agent.

        Args:
            retriever: BM25Retriever instance
            llm_client: LLMClient instance
            safety_layer: SafetyLayer instance
        """
        self.retriever = retriever
        self.llm = llm_client
        self.safety = safety_layer
        self.call_count = 0
        # Tool executor (audit-mode by default)
        self.tool_executor = tool_executor or create_tool_executor()

    def process_ticket(self, state: TicketState) -> TicketResponse:
        """
        Process a single support ticket.

        Main pipeline:
        1. Detect injection
        2. Detect PII
        3. Infer product
        4. Assess risk
        5. Retrieve documents
        6. Generate response via LLM
        7. Validate output
        8. Override if needed (e.g., low confidence)

        Args:
            state: TicketState with ticket info

        Returns:
            TicketResponse
        """
        self.call_count += 1
        logger.info(f"Processing ticket {state.ticket_id} (call #{self.call_count})")

        try:
            # ===== LAYER 1: INJECTION DETECTION =====
            conversation_text = state.get_conversation_text()

            is_injection, inj_confidence, inj_reason = self.safety.detect_prompt_injection(
                conversation_text
            )

            if is_injection:
                state.injection_detected = True
                state.injection_confidence = inj_confidence
                logger.warning(f"Injection detected: {inj_reason}")
                return self._escalate_response(
                    state,
                    "Prompt injection detected. Request escalated for review.",
                    reason="injection_detected",
                )

            # ===== LAYER 2: PII DETECTION =====
            has_pii, pii_types, pii_matches = self.safety.detect_pii(conversation_text)
            if has_pii:
                state.add_pii_detection(pii_types)

            # Sanitize conversation text for LLM
            sanitized_text = self.safety.sanitize_pii(conversation_text)

            # ===== LAYER 3: PRODUCT DETECTION =====
            product = self._detect_product(conversation_text, state.company_hint)
            state.inferred_product = product
            logger.info(f"Product detected: {product}")

            # ===== LAYER 4: RISK ASSESSMENT =====
            risk_level = self._assess_risk(conversation_text, state)
            state.update_risk_level()

            # If critical risk, escalate immediately
            if state.risk_level == RiskLevel.CRITICAL:
                logger.warning(f"Critical risk detected, escalating")
                return self._escalate_response(
                    state,
                    "This request requires immediate attention from our support team.",
                    reason="critical_risk",
                )

            # ===== LAYER 5: RETRIEVAL =====
            retrieved_docs = self._retrieve_documents(sanitized_text, product)
            for doc_path, _, _ in retrieved_docs:
                state.add_retrieved_document(doc_path)

            if not retrieved_docs:
                logger.warning("No relevant documents found")
                return self._escalate_response(
                    state,
                    "Your request is outside our current knowledge base. A specialist will review it.",
                    reason="no_documents_found",
                )

            # ===== LAYER 6: LLM REASONING =====
            context = self._prepare_context(retrieved_docs)
            llm_output, llm_errors = self._call_llm(
                sanitized_text,
                context,
                state,
            )

            if not llm_output or llm_errors:
                logger.error(f"LLM call failed: {llm_errors}")
                return self._escalate_response(
                    state,
                    "Unable to process your request at this time. Please try again.",
                    reason="llm_error",
                )

            # ===== LAYER 6.5: SIMULATE/VALIDATE TOOL CALLS =====
            actions = getattr(llm_output, "actions_taken", None) or []
            if actions:
                logger.info(f"LLM proposed {len(actions)} action(s); validating in audit-mode")
                tool_results = self.tool_executor.simulate_actions(state.ticket_id, actions, state)

                # If any action failed validation or prereq checks, escalate
                for tr in tool_results:
                    if not tr.get("success"):
                        logger.warning(f"Tool simulation failed: {tr}")
                        return self._escalate_response(
                            state,
                            "Action proposed by the agent failed safety or prerequisite checks. Escalating to human.",
                            reason="tool_validation_failed",
                        )

            # ===== LAYER 7: OUTPUT VALIDATION & OVERRIDES =====
            response = self._validate_and_finalize(llm_output, state, retrieved_docs)

            logger.info(f"Ticket {state.ticket_id} processed: status={response.status}")
            return response

        except Exception as e:
            logger.exception(f"Error processing ticket {state.ticket_id}: {e}")
            state.add_error(str(e))
            return self._escalate_response(
                state,
                "An error occurred processing your request. Our team will review it.",
                reason="processing_error",
            )

    def _detect_product(self, text: str, company_hint: Optional[str]) -> str:
        """
        Infer product from text and company hint.

        Returns:
            Product name (DevPlatform, Claude, Visa)
        """
        text_lower = text.lower()

        # Score each product
        scores = {}
        for product, keywords in PRODUCT_KEYWORDS.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            scores[product] = score

        # If company hint is present and has non-zero score, prefer it
        if company_hint and company_hint in scores:
            scores[company_hint] += 2  # Boost hint

        # Return highest score
        best_product = max(scores, key=scores.get)
        return best_product if scores[best_product] > 0 else "Unknown"

    def _assess_risk(self, text: str, state: TicketState) -> RiskLevel:
        """
        Assess risk level of ticket.

        Returns:
            RiskLevel
        """
        text_lower = text.lower()

        # Critical keywords
        for keyword in RISK_CRITICAL_KEYWORDS:
            if keyword in text_lower:
                state.add_risk_flag(keyword)
                return RiskLevel.CRITICAL

        # High keywords
        for keyword in RISK_HIGH_KEYWORDS:
            if keyword in text_lower:
                state.add_risk_flag(keyword)

        # PII is high risk
        if state.pii_detected:
            state.add_risk_flag("pii_present")

        state.update_risk_level()
        return state.risk_level

    def _retrieve_documents(
        self,
        query: str,
        product: str,
    ) -> List[Tuple[str, float, str]]:
        """
        Retrieve relevant documents.

        Returns:
            List of (path, score, content) tuples
        """
        logger.info(f"Retrieving documents for: {query[:100]}...")

        # BM25 retrieval
        docs = self.retriever.retrieve_by_product(query, product) if product != "Unknown" else self.retriever.retrieve(query)

        logger.info(f"Retrieved {len(docs)} documents via BM25")
        return docs[:RERANK_TOP_K]

    def _prepare_context(self, docs: List[Tuple[str, float, str]]) -> str:
        """
        Prepare context string from documents.

        Returns:
            Formatted context
        """
        context_parts = []

        for path, score, content in docs:
            # Truncate content
            truncated = content[:500] if len(content) > 500 else content
            context_parts.append(f"[{path}]\n{truncated}\n")

        return "\n---\n".join(context_parts)

    def _call_llm(
        self,
        query: str,
        context: str,
        state: TicketState,
    ) -> Tuple[Optional[LLMOutput], List[str]]:
        """
        Call LLM for response generation.

        Returns:
            (LLMOutput, errors)
        """
        logger.info("Calling LLM for response generation")

        # Build prompt
        full_query = f"""Support Ticket:
{query}

Relevant Documentation:
{context}

Generate a structured response."""

        llm_output, errors = self.llm.generate_structured_response(
            ticket_query=full_query,
            retrieved_context=context,
        )

        return llm_output, errors

    def _validate_and_finalize(
        self,
        llm_output: LLMOutput,
        state: TicketState,
        retrieved_docs: List[Tuple[str, float, str]],
    ) -> TicketResponse:
        """
        Validate and finalize response with overrides.

        Returns:
            TicketResponse
        """
        # Convert LLMOutput to dict for manipulation
        data = llm_output.dict()

        # OVERRIDE 1: Low confidence → escalate
        if llm_output.confidence_score < CONFIDENCE_ESCALATION_THRESHOLD:
            logger.info(f"Low confidence ({llm_output.confidence_score}), escalating")
            data["status"] = "escalated"
            data["response"] = "Your request requires specialist review. We'll connect you shortly."
            data["confidence_score"] = 0.1

        # OVERRIDE 2: PII detected → be cautious
        if state.pii_detected:
            data["pii_detected"] = True

        # OVERRIDE 3: Mask PII in response
        if state.pii_detected and data["response"]:
            data["response"] = self.safety.mask_pii_in_response(data["response"])

        # Add retrieved document paths
        data["source_documents"] = [doc[0] for doc in retrieved_docs]

        # Validate schema
        is_valid, errors = self.safety.validate_output_schema(data)
        if not is_valid:
            logger.warning(f"Schema validation failed: {errors}")
            return self._escalate_response(
                state,
                "Request requires manual review.",
                reason="validation_error",
            )

        # Convert to TicketResponse
        try:
            response = TicketResponse(**data)
            return response
        except Exception as e:
            logger.error(f"Failed to create TicketResponse: {e}")
            return self._escalate_response(
                state,
                "Unable to generate response.",
                reason="response_creation_error",
            )

    def _escalate_response(
        self,
        state: TicketState,
        message: str,
        reason: str,
    ) -> TicketResponse:
        """
        Create escalation response.

        Returns:
            TicketResponse with status=escalated
        """
        return TicketResponse(
            status="escalated",
            product_area=state.inferred_product or "Support",
            response=message,
            justification=f"Escalated: {reason}",
            request_type="invalid",
            confidence_score=0.1,
            risk_level=state.risk_level.value,
            pii_detected=state.pii_detected,
            language=state.language,
            source_documents=[],
            actions_taken=[],
        )
