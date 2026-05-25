"""
llm_client.py - LLM client for structured reasoning and response generation.

Integrates with DeepSeek API (OpenAI-compatible endpoint).
"""

import json
import logging
import time
from typing import Optional, Dict, Any, Tuple, List
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_API_BASE,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
    LLM_RETRIES,
    SYSTEM_PROMPT,
)
from schemas import LLMOutput

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for LLM interactions via DeepSeek API."""

    def __init__(self):
        """Initialize LLM client with DeepSeek API."""
        if not DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY not set")

        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)
        self.model = LLM_MODEL
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = LLM_MAX_TOKENS
        self.timeout = LLM_TIMEOUT
        self.retries = LLM_RETRIES

    def generate_response(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[list] = None,
    ) -> Tuple[str, bool]:
        """
        Generate LLM response with retries.

        Args:
            user_message: User query
            system_prompt: System prompt (default: SYSTEM_PROMPT)
            conversation_history: Previous messages

        Returns:
            (response_text, success)
        """
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        for attempt in range(self.retries):
            try:
                logger.info(f"LLM call (attempt {attempt + 1}/{self.retries})")

                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=messages,
                    timeout=self.timeout,
                )

                # Extract text
                if response.choices and len(response.choices) > 0:
                    text = response.choices[0].message.content
                    logger.info(f"LLM response received ({len(text)} chars)")
                    return text, True
                else:
                    logger.warning("Empty LLM response")
                    return "", False

            except RateLimitError:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Rate limited, waiting {wait_time}s")
                time.sleep(wait_time)

            except (APIConnectionError, APIError) as e:
                logger.error(f"LLM API error: {e}")
                if attempt < self.retries - 1:
                    time.sleep(2)
                else:
                    return "", False

            except Exception as e:
                logger.error(f"Unexpected error in LLM call: {e}")
                return "", False

        return "", False

    def parse_json_response(self, response_text: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """
        Parse JSON from LLM response.

        Returns:
            (parsed_json, errors)
        """
        errors = []

        if not response_text or len(response_text.strip()) == 0:
            errors.append("Empty response")
            return None, errors

        # Try to extract JSON from response
        try:
            # First, try direct parsing
            parsed = json.loads(response_text)
            return parsed, errors
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in response
        lines = response_text.split("\n")
        in_json = False
        json_lines = []

        for line in lines:
            if "{" in line:
                in_json = True
            if in_json:
                json_lines.append(line)
            if "}" in line and in_json:
                break

        if json_lines:
            json_text = "\n".join(json_lines)
            try:
                parsed = json.loads(json_text)
                return parsed, errors
            except json.JSONDecodeError as e:
                errors.append(f"Failed to parse JSON: {e}")

        errors.append("Could not find valid JSON in response")
        return None, errors

    def generate_structured_response(
        self,
        ticket_query: str,
        retrieved_context: str,
        conversation_history: Optional[list] = None,
    ) -> Tuple[Optional[LLMOutput], List[str]]:
        """
        Generate structured ticket response.

        Args:
            ticket_query: The customer's question
            retrieved_context: Relevant documentation
            conversation_history: Previous messages

        Returns:
            (LLMOutput, errors)
        """
        errors = []

        # Build prompt
        prompt = f"""Analyze this support ticket and provide a structured response.

CUSTOMER QUERY:
{ticket_query}

RELEVANT DOCUMENTATION:
{retrieved_context}

IMPORTANT: 
- Respond ONLY with valid JSON
- Do not include any text before or after the JSON
- All fields are required
- Confidence score must be between 0.0 and 1.0

Generate JSON response matching this structure:
{{
  "status": "replied or escalated",
  "product_area": "category",
  "response": "user-facing answer",
  "justification": "explanation",
  "request_type": "product_issue|feature_request|bug|invalid",
  "confidence_score": 0.0-1.0,
  "risk_level": "low|medium|high|critical",
  "pii_detected": true or false,
  "language": "en",
  "source_documents": ["path1", "path2"],
  "actions_taken": []
}}
"""

        # Call LLM
        response_text, success = self.generate_response(
            user_message=prompt,
            system_prompt=SYSTEM_PROMPT,
            conversation_history=conversation_history,
        )

        if not success:
            errors.append("Failed to get LLM response")
            return None, errors

        # Parse JSON
        parsed, parse_errors = self.parse_json_response(response_text)
        errors.extend(parse_errors)

        if not parsed:
            return None, errors

        # Convert to LLMOutput
        try:
            output = LLMOutput(**parsed)
            return output, errors
        except Exception as e:
            errors.append(f"Failed to validate output: {e}")
            return None, errors

    def rerank_documents(
        self,
        query: str,
        documents: list,
    ) -> Tuple[Optional[list], List[str]]:
        """
        Use LLM to re-rank retrieved documents by relevance.

        Args:
            query: Original query
            documents: List of (path, content, score) tuples

        Returns:
            (ranked_paths, errors)
        """
        errors = []

        if not documents:
            return [], errors

        # Prepare document summaries
        doc_summaries = []
        for i, (path, content, score) in enumerate(documents):
            summary = content[:300].replace("\n", " ")  # First 300 chars
            doc_summaries.append(f"{i}. [{path}]\n{summary}")

        prompt = f"""Given this query and documents, rank them by relevance to answer the query.

QUERY: {query}

DOCUMENTS:
{chr(10).join(doc_summaries)}

Return JSON array of indices sorted by relevance (most relevant first):
[0, 2, 1]
"""

        response_text, success = self.generate_response(
            user_message=prompt,
            system_prompt="You are a document relevance ranker. Respond ONLY with a JSON array of indices.",
        )

        if not success:
            errors.append("Failed to get LLM response for re-ranking")
            return None, errors

        # Parse response
        parsed, parse_errors = self.parse_json_response(response_text)
        errors.extend(parse_errors)

        if not isinstance(parsed, list):
            errors.append("Re-ranking response is not an array")
            return None, errors

        # Map back to paths
        ranked_paths = []
        for idx in parsed:
            if isinstance(idx, int) and 0 <= idx < len(documents):
                ranked_paths.append(documents[idx][0])

        return ranked_paths, errors

    def check_injection(self, text: str) -> Tuple[bool, float, str]:
        """
        Use LLM to check for prompt injection (secondary check).

        Args:
            text: Text to check

        Returns:
            (is_injection, confidence, reason)
        """
        from config import INJECTION_DETECTION_PROMPT

        prompt = INJECTION_DETECTION_PROMPT.format(text=text[:1000])

        response_text, success = self.generate_response(
            user_message=prompt,
            system_prompt="You are a prompt injection detector. Respond ONLY with JSON.",
        )

        if not success:
            return False, 0.0, "LLM detection unavailable"

        parsed, _ = self.parse_json_response(response_text)

        if parsed:
            return (
                parsed.get("is_injection", False),
                parsed.get("confidence", 0.0),
                parsed.get("reason", ""),
            )

        return False, 0.0, "Failed to parse injection response"


def create_llm_client() -> LLMClient:
    """Factory function to create LLM client."""
    return LLMClient()
