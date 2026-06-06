"""Tests for the audit-log aggregator (runtime/hard/audit_report.py)."""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

from runtime.hard.audit_report import format_summary, summarize_audit


def _write(d: Path, ts: int, record: dict) -> None:
    (d / f"turn-{ts}.json").write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8")


class TestSummarizeAudit(unittest.TestCase):
    def _tmp(self):
        return Path(tempfile.mkdtemp(prefix="pi_audit_"))

    def test_empty_dir(self):
        s = summarize_audit(self._tmp())
        self.assertEqual(s["turns"], 0)
        self.assertEqual(s["outcomes"], {})

    def test_counts_outcomes_and_gates(self):
        d = self._tmp()
        _write(d, 1, {"outcome": "ok", "guardrails_passed": True,
                      "tool_log": [{"name": "search", "result": {}}],
                      "memory_delta": {"summary": "x"}})
        _write(d, 2, {"outcome": "blocked_by_guardrail",
                      "guardrails_passed": False,
                      "guardrail_verdicts": [{"id": "no-secrets", "passed": False}]})
        _write(d, 3, {"outcome": "ok", "guardrails_passed": True,
                      "sanitization": {"redacted": [
                          {"label": "openai_key"}, {"label": "aws_key"}]}})
        _write(d, 4, {"outcome": "tool_error",
                      "tool_log": [{"name": "calc", "error": "boom"}]})

        s = summarize_audit(d)
        self.assertEqual(s["turns"], 4)
        self.assertEqual(s["outcomes"],
                         {"ok": 2, "blocked_by_guardrail": 1, "tool_error": 1})
        self.assertEqual(s["guardrail_blocks"], 1)
        self.assertEqual(s["blocks_by_guardrail"], {"no-secrets": 1})
        self.assertEqual(s["redaction_matches"], 2)
        self.assertEqual(s["redactions_by_label"],
                         {"openai_key": 1, "aws_key": 1})
        self.assertEqual(s["tool_calls"], 1)
        self.assertEqual(s["tool_calls_by_name"], {"search": 1})
        self.assertEqual(s["tool_errors"], 1)
        self.assertEqual(s["tool_errors_by_name"], {"calc": 1})
        self.assertEqual(s["memory_writes"], 1)

    def test_ignores_unparseable_files(self):
        d = self._tmp()
        _write(d, 1, {"outcome": "ok"})
        (d / "turn-2.json").write_text("{ not json", encoding="utf-8")
        (d / "notes.txt").write_text("ignore me", encoding="utf-8")
        s = summarize_audit(d)
        self.assertEqual(s["turns"], 1)  # torn file and non-turn file skipped

    def test_format_summary_is_readable(self):
        d = self._tmp()
        _write(d, 1, {"outcome": "ok", "guardrails_passed": True})
        out = format_summary(summarize_audit(d))
        self.assertIn("turns:", out)
        self.assertIn("outcomes:", out)
        self.assertIn("guardrail blocks:", out)


if __name__ == "__main__":
    unittest.main()
