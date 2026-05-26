import unittest
from pathlib import Path

from tool_executor import ToolExecutor
from state import TicketState, IdentityStatus


class TestToolExecutor(unittest.TestCase):

    def setUp(self):
        # Use a temp audit log in the repo tmp dir
        self.audit_path = Path("/tmp/tool_audit_test.log")
        try:
            self.audit_path.unlink()
        except Exception:
            pass

        self.exec = ToolExecutor(audit_log=self.audit_path)
        self.state = TicketState(ticket_id="test-1")

    def test_verify_identity_success(self):
        res = self.exec.simulate_action("test-1", {"action": "verify_identity", "parameters": {"method": "email_otp", "target": "user@example.com"}}, self.state)
        self.assertTrue(res.get("success"))
        self.assertEqual(self.state.identity_verified, IdentityStatus.VERIFIED)

    def test_lock_account_string_action(self):
        # lock_account requires parameters; as string it should fail schema validation
        res = self.exec.simulate_action("test-1", "lock_account", self.state)
        self.assertFalse(res.get("success"))

    def test_missing_required_params(self):
        res = self.exec.simulate_action("test-1", {"action": "issue_refund", "parameters": {"amount": 10}}, self.state)
        self.assertFalse(res.get("success"))


if __name__ == "__main__":
    unittest.main()
