#!/usr/bin/env python3
"""
MLE Hiring Challenge — Output Validator

Run this script to check your output.csv for format compliance before submission.
This validates STRUCTURE ONLY — it does NOT evaluate correctness.

Usage:
    python validate_output.py
"""

import csv
import sys
import os
import json

from schemas import TicketResponse

EXPECTED_HEADERS = [
    "issue", "subject", "company", "response", "product_area",
    "status", "request_type", "justification", "confidence_score",
    "source_documents", "risk_level", "pii_detected", "language",
    "actions_taken"
]

VALID_STATUS = {"replied", "escalated"}
VALID_REQUEST_TYPE = {"product_issue", "feature_request", "bug", "invalid"}
VALID_RISK_LEVEL = {"low", "medium", "high", "critical"}
VALID_PII_DETECTED = {"true", "false"}

def validate():
    output_path = os.path.join(os.path.dirname(__file__), "..", "support_tickets", "output.csv")
    input_path = os.path.join(os.path.dirname(__file__), "..", "support_tickets", "support_tickets.csv")
    
    if not os.path.exists(output_path):
        print("❌ FAIL: output.csv not found at", output_path)
        return False
    
    if not os.path.exists(input_path):
        print("❌ FAIL: support_tickets.csv not found at", input_path)
        return False
    
    # Count input rows
    with open(input_path, "r", encoding="utf-8") as f:
        input_reader = csv.reader(f)
        next(input_reader)  # skip header
        input_count = sum(1 for _ in input_reader)
    
    # Validate output
    errors = []
    warnings = []
    
    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Check headers
        actual_headers = reader.fieldnames
        if actual_headers is None:
            print("❌ FAIL: output.csv is empty or has no headers")
            return False

        rows = list(reader)
        output_count = len(rows)

        if output_count != input_count:
            errors.append(f"Row count mismatch: expected {input_count}, got {output_count}")

        # Validate each row by attempting to construct TicketResponse (Pydantic)
        for i, row in enumerate(rows, start=1):
            # Normalize and convert fields to expected types
            data = {}
            # Map CSV fields to TicketResponse fields where possible
            data["status"] = row.get("status")
            data["product_area"] = row.get("product_area") or row.get("Product Area") or row.get("product area")
            data["response"] = row.get("response") or row.get("Response")
            data["justification"] = row.get("justification") or row.get("Justification") or ""
            data["request_type"] = row.get("request_type") or row.get("Request Type")
            # confidence to float
            conf = row.get("confidence_score") or row.get("Confidence Score")
            try:
                data["confidence_score"] = float(conf) if conf not in (None, "") else None
            except Exception:
                errors.append(f"Row {i}: invalid confidence_score '{conf}'")
                data["confidence_score"] = None

            # risk level
            data["risk_level"] = (row.get("risk_level") or row.get("Risk Level"))

            # pii_detected
            pii = row.get("pii_detected") or row.get("PII Detected")
            if pii is not None:
                pii_val = str(pii).strip().lower()
                if pii_val in ("true", "false"):
                    data["pii_detected"] = (pii_val == "true")
                else:
                    errors.append(f"Row {i}: invalid pii_detected '{pii}'")
                    data["pii_detected"] = False
            else:
                data["pii_detected"] = False

            data["language"] = row.get("language") or row.get("Language") or "en"

            # source_documents may be pipe-separated
            src = row.get("source_documents") or row.get("Source Documents") or ""
            data["source_documents"] = [p for p in (src.split("|") if src else []) if p]

            # actions_taken expected JSON array
            actions_raw = row.get("actions_taken") or row.get("Actions Taken") or "[]"
            try:
                parsed_actions = json.loads(actions_raw) if actions_raw else []
                data["actions_taken"] = parsed_actions
            except json.JSONDecodeError as e:
                errors.append(f"Row {i}: actions_taken is not valid JSON ({str(e)})")
                data["actions_taken"] = []

            # Attempt Pydantic validation
            try:
                TicketResponse(**data)
            except Exception as e:
                errors.append(f"Row {i}: schema validation failed: {e}")
    
    # Print results
    print("=" * 60)
    print("MLE Hiring Challenge — Output Validation Report")
    print("=" * 60)
    print(f"\nInput tickets:  {input_count}")
    print(f"Output rows:    {output_count}")
    print(f"Columns found:  {len(actual_headers)}/{len(EXPECTED_HEADERS)}")
    
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for e in errors[:20]:  # cap at 20
            print(f"   • {e}")
        if len(errors) > 20:
            print(f"   ... and {len(errors) - 20} more errors")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings[:10]:  # cap at 10
            print(f"   • {w}")
        if len(warnings) > 10:
            print(f"   ... and {len(warnings) - 10} more warnings")
    
    if not errors:
        print("\n✅ PASS: Output format is valid.")
        print("   Note: This validates structure only, NOT correctness.")
        print("   Your submission will also be evaluated on a hidden test set.")
    else:
        print(f"\n❌ FAIL: {len(errors)} errors found. Fix them before submitting.")
    
    print("=" * 60)
    return len(errors) == 0


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
