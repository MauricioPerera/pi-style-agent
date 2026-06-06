"""Tests for runtime.hard.output_sanitize. No LLM, no network."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from runtime.hard.output_sanitize import sanitize


class TestSanitize(unittest.TestCase):
    def test_clean_text(self):
        r = sanitize("Hello, how can I help you?")
        self.assertTrue(r.clean)
        self.assertEqual(r.text, "Hello, how can I help you?")
        self.assertEqual(r.redacted, [])

    def test_redacts_openai_key(self):
        r = sanitize("Your key is sk-ABCDEFGHIJKLMNOPQRSTUV and that is it.")
        self.assertFalse(r.clean)
        self.assertIn("[REDACTED:openai_key]", r.text)
        self.assertNotIn("sk-ABCDEFGHIJKLMNOPQRSTUV", r.text)
        self.assertEqual(r.redacted[0]["label"], "openai_key")

    def test_redacts_aws_key(self):
        r = sanitize("AKIAIOSFODNN7EXAMPLE was the access key.")
        self.assertFalse(r.clean)
        self.assertIn("[REDACTED:aws_key]", r.text)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", r.text)

    def test_redacts_pem_block(self):
        r = sanitize("Begin block:\n-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n-----END RSA PRIVATE KEY-----\n")
        self.assertFalse(r.clean)
        self.assertIn("[REDACTED:private_key]", r.text)
        # The header is gone; the body is preserved (we only redact the marker).
        self.assertIn("MIIE", r.text)

    def test_redacts_github_pat(self):
        r = sanitize("token: ghp_abcdefghijklmnopqrstuvwxyz0123456789")
        self.assertFalse(r.clean)
        self.assertIn("[REDACTED:github_pat]", r.text)

    def test_redacts_slack_token(self):
        r = sanitize("Slack: xoxb-1234567890-12345")
        self.assertFalse(r.clean)
        self.assertIn("[REDACTED:slack_token]", r.text)

    def test_multiple_redactions_counted(self):
        text = "sk-ABCDEFGHIJKLMNOPQRSTUV and AKIAIOSFODNN7EXAMPLE in the same string"
        r = sanitize(text)
        self.assertFalse(r.clean)
        self.assertEqual(len(r.redacted), 2)
        self.assertIn("[REDACTED:openai_key]", r.text)
        self.assertIn("[REDACTED:aws_key]", r.text)

    def test_summary_format(self):
        r = sanitize("sk-ABCDEFGHIJKLMNOPQRSTUV and AKIAIOSFODNN7EXAMPLE")
        s = r.summary()
        self.assertIn("aws_key", s)
        self.assertIn("openai_key", s)
        self.assertIn("2 match", s)

    def test_extra_patterns(self):
        r = sanitize("my phone is 555-123-4567",
                     extra_patterns=[r"555-\d{3}-\d{4}"])
        self.assertFalse(r.clean)
        self.assertIn("[REDACTED:secret]", r.text)
        self.assertEqual(r.redacted[0]["label"], "secret")  # unknown label

    def test_never_aborts(self):
        # A text full of secrets should still return a string, just redacted.
        text = "sk-ABCDEFGHIJKLMNOPQRSTUV " * 3 + " " + "AKIA" + "B" * 16
        r = sanitize(text)
        self.assertIsInstance(r.text, str)
        self.assertGreater(len(r.redacted), 0)
        self.assertNotIn("sk-A", r.text)


class TestSanitizeInAgentLoop(unittest.TestCase):
    """Verify that run_turn applies sanitization to the final reply."""
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="pi_san_"))
        from runtime.soft.assembler import load_contract
        self.contract = load_contract(Path("contracts/agent-contract.json"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_secret_in_llm_reply_gets_redacted_not_aborted(self):
        from runtime.soft.agent import run_turn
        from runtime.soft.llm import LLMResponse

        def fake_llm(system, user, model=""):
            # LLM echoes a secret in its reply. We expect sanitization to
            # redact, NOT to block the whole turn.
            return LLMResponse(
                text="Sure, your key is sk-ABCDEFGHIJKLMNOPQRSTUV. Bye.",
                model="stub", tokens_in=0, tokens_out=0)

        contents = {
            "persona": "Iris.", "hard_policies": "- no secrets.",
            "long_term_mem": "",
            "plan": "", "scratchpad": "",
            "tool_results": json.dumps({"tool": "none", "ok": True, "data": []}),
            "history": "", "user_input": "hola",
        }
        rec = run_turn(self.contract, contents, audit_dir=self.tmp,
                       llm_callable=fake_llm)
        self.assertEqual(rec.outcome, "ok")
        self.assertIn("[REDACTED:openai_key]", rec.user_message)
        self.assertNotIn("sk-ABCDEFGHIJKLMNOPQRSTUV", rec.user_message)
        # Audit log records the redaction.
        self.assertIn("sanitization", rec.audit_record)
        self.assertEqual(rec.audit_record["sanitization"]["redacted"][0]["label"],
                         "openai_key")


