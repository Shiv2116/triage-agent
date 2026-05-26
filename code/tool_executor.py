"""
tool_executor.py - Safe ToolExecutor (audit-mode)

Implements a non-destructive simulator for internal tool calls defined
in `data/api_specs/internal_tools.json`. Validates call names, required
parameters, and prerequisite actions (e.g., `verify_identity` before
`issue_refund`). Writes an audit log to `support_tickets/tool_audit.log`.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import API_SPECS_DIR, PROJECT_ROOT, TOOL_PREREQUISITES
from state import TicketState, IdentityStatus

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Safe executor that validates and simulates tool calls.

    By default runs in "audit" mode: does not call external systems,
    but validates schemas/prereqs and writes an audit record.
    """

    def __init__(self, audit_log: Optional[Path] = None):
        specs_path = API_SPECS_DIR / "internal_tools.json"
        try:
            with open(specs_path, "r", encoding="utf-8") as f:
                tools = json.load(f)
        except Exception:
            tools = []

        # Map tool name -> spec
        self.tools: Dict[str, Dict[str, Any]] = {t["name"]: t for t in tools}
        self.prereqs = TOOL_PREREQUISITES if TOOL_PREREQUISITES is not None else {}

        self.audit_log = audit_log or (PROJECT_ROOT / "support_tickets" / "tool_audit.log")
        # ensure parent exists
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)

    def _audit_entry(self, ticket_id: str, entry: Dict[str, Any]) -> None:
        entry_text = json.dumps({"timestamp": datetime.utcnow().isoformat() + "Z", "ticket_id": ticket_id, **entry})
        try:
            with open(self.audit_log, "a", encoding="utf-8") as f:
                f.write(entry_text + "\n")
        except Exception:
            logger.exception("Failed to write audit log")

    def _validate_schema(self, action: str, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errs: List[str] = []
        spec = self.tools.get(action)
        if not spec:
            errs.append(f"Unknown action '{action}'")
            return False, errs

        # Minimal validation: required parameters present
        params_spec = spec.get("parameters", {})
        required = params_spec.get("required", [])
        for r in required:
            if r not in parameters:
                errs.append(f"Missing required parameter '{r}' for action '{action}'")

        return (len(errs) == 0), errs

    def _check_prereqs(self, action: str, state: TicketState) -> Tuple[bool, List[str]]:
        errs: List[str] = []
        needed = self.prereqs.get(action, [])
        for p in needed:
            # If prereq is verify_identity check state
            if p == "verify_identity":
                if state.identity_verified != IdentityStatus.VERIFIED:
                    errs.append("identity not verified")
            else:
                # Check previous actions
                if not any(a.get("action") == p for a in state.previous_actions):
                    errs.append(f"missing prerequisite action '{p}'")

        return (len(errs) == 0), errs

    def simulate_action(self, ticket_id: str, action_obj: Dict[str, Any] | str, state: TicketState) -> Dict[str, Any]:
        """Validate and simulate a single action.

        Returns a result dict with keys: action, parameters, success, messages
        """
        # Support either a dict {"action":..., "parameters": {...}} or a simple string action name
        if isinstance(action_obj, str):
            action = action_obj
            parameters = {}
        else:
            action = action_obj.get("action")
            parameters = action_obj.get("parameters", {}) or {}

        result: Dict[str, Any] = {"action": action, "parameters": parameters, "success": False, "messages": []}

        # Basic validation
        ok, errs = self._validate_schema(action, parameters)
        if not ok:
            result["messages"].extend(errs)
            self._audit_entry(ticket_id, {"action": action, "parameters": parameters, "status": "invalid", "messages": errs})
            return result

        # Prerequisite check
        ok2, pre_errs = self._check_prereqs(action, state)
        if not ok2:
            result["messages"].extend(pre_errs)
            self._audit_entry(ticket_id, {"action": action, "parameters": parameters, "status": "prereq_failed", "messages": pre_errs})
            return result

        # Simulate execution: for safe actions like verify_identity, update state
        if action == "verify_identity":
            # In audit mode we simulate success and mark identity as VERIFIED
            state.identity_verified = IdentityStatus.VERIFIED
            state.add_previous_action(action, parameters)
            result["success"] = True
            result["messages"].append("simulated: identity verified")
            self._audit_entry(ticket_id, {"action": action, "parameters": parameters, "status": "simulated_success", "messages": result["messages"]})
            return result

        # For other actions, record as simulated success and do not perform side-effects
        state.add_previous_action(action, parameters)
        result["success"] = True
        result["messages"].append("simulated: action recorded (audit-mode)")
        self._audit_entry(ticket_id, {"action": action, "parameters": parameters, "status": "simulated_success", "messages": result["messages"]})
        return result

    def simulate_actions(self, ticket_id: str, actions: List[Dict[str, Any]], state: TicketState) -> List[Dict[str, Any]]:
        """Simulate a sequence of actions in order.

        Returns list of result dicts.
        """
        results: List[Dict[str, Any]] = []
        for act in actions or []:
            try:
                res = self.simulate_action(ticket_id, act, state)
            except Exception as e:
                # act may be a string or malformed; defensively extract fields
                if isinstance(act, str):
                    act_name = act
                    act_params = {}
                else:
                    act_name = act.get("action") if isinstance(act, dict) else None
                    act_params = act.get("parameters", {}) if isinstance(act, dict) else {}

                res = {"action": act_name, "parameters": act_params, "success": False, "messages": [str(e)]}
                self._audit_entry(ticket_id, {"action": act_name, "parameters": act_params, "status": "error", "messages": [str(e)]})
            results.append(res)
            # If a required prereq was missing, stop executing further actions
            if not res.get("success"):
                break

        return results


# Simple factory
def create_tool_executor() -> ToolExecutor:
    return ToolExecutor()
