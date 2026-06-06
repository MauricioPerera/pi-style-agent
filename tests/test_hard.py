"""Hard-layer tests. No LLM, no network. Stdlib unittest."""
from runtime.hard.budget import SlotSpec, allocate, estimate_tokens, truncate_to_tokens, tokeniser_name
from runtime.hard.guardrails import run_guardrails
from runtime.hard.secrets import compile_patterns, find_secrets
import json
import unittest


class TestTokenEstimation(unittest.TestCase):
    def test_zero_for_empty(self):
        self.assertEqual(estimate_tokens(""), 0)

    def test_positive_for_nonempty(self):
        # The exact count depends on whether tiktoken is installed; just
        # assert the function returns a positive int and is monotonic
        # with text length.
        self.assertGreater(estimate_tokens("hello world"), 0)
        self.assertLess(estimate_tokens("a"), estimate_tokens("a" * 100))

    def test_truncate_round_trip(self):
        # Truncating a non-trivial string at any positive max_tokens
        # should give back a non-empty string no longer than the input.
        text = "the quick brown fox jumps over the lazy dog " * 10
        for k in (1, 5, 20, 100):
            out = truncate_to_tokens(text, k)
            self.assertLessEqual(len(out), len(text))
            if k > 0:
                self.assertGreater(len(out), 0)


class TestSecretPatterns(unittest.TestCase):
    def test_catches_openai_key(self):
        c = compile_patterns([r"sk-[A-Za-z0-9]{20,}"])
        self.assertEqual(len(find_secrets("here is sk-ABCDEFGHIJKLMNOPQRSTUV leaked", c)), 1)

    def test_catches_aws_key(self):
        c = compile_patterns([r"AKIA[0-9A-Z]{16}"])
        self.assertEqual(len(find_secrets("AKIAIOSFODNN7EXAMPLE", c)), 1)

    def test_catches_pem_block(self):
        c = compile_patterns([r"-----BEGIN [A-Z ]*PRIVATE KEY-----"])
        self.assertEqual(len(find_secrets("-----BEGIN RSA PRIVATE KEY-----", c)), 1)

    def test_clean_text_has_no_hits(self):
        c = compile_patterns([r"sk-[A-Za-z0-9]{20,}", r"AKIA[0-9A-Z]{16}"])
        self.assertEqual(find_secrets("this is clean documentation, no keys", c), [])


class TestBudgetAllocation(unittest.TestCase):
    PERSONA = SlotSpec("persona",       priority=0, kind="static",  compaction="none",      min_tokens=10)
    POLICIES = SlotSpec("hard_policies", priority=0, kind="static",  compaction="none",      min_tokens=20)
    MEM = SlotSpec("long_term_mem",     priority=1, kind="dynamic", compaction="summarize", min_tokens=10, max_tokens=200)
    TOOLS = SlotSpec("tool_results",    priority=3, kind="dynamic", compaction="truncate",  max_tokens=100)
    HISTORY = SlotSpec("history",       priority=4, kind="dynamic", compaction="truncate",  max_tokens=500)

    def test_critical_slots_never_truncated(self):
        r = allocate(
            [self.PERSONA, self.POLICIES],
            {"persona": "a" * 40, "hard_policies": "b" * 80},
            available=50,
        )
        self.assertIsNone(r.aborted)
        actions = {a.spec.id: a.action for a in r.slots}
        self.assertEqual(actions, {"persona": "full", "hard_policies": "full"})

    def test_aborts_when_critical_does_not_fit(self):
        r = allocate(
            [self.PERSONA, self.POLICIES],
            {"persona": "a" * 200, "hard_policies": "b" * 200},
            available=50,
        )
        self.assertIsNotNone(r.aborted)
        # Whichever critical is processed first gets reported.
        self.assertTrue(r.aborted.startswith("slot critico"))

    def test_low_priority_dropped_under_pressure(self):
        r = allocate(
            [self.PERSONA, self.POLICIES, self.TOOLS, self.HISTORY],
            {"persona": "p" * 40, "hard_policies": "q" * 80,
             "tool_results": "t" * 4000, "history": "h" * 8000},
            available=200,
        )
        self.assertIsNone(r.aborted)
        actions = {a.spec.id: a.action for a in r.slots}
        # criticals get full budget; low-priority cannot fit
        self.assertEqual(actions["persona"], "full")
        self.assertEqual(actions["hard_policies"], "full")
        self.assertIn(actions["tool_results"], ("truncate", "drop"))
        self.assertIn(actions["history"], ("truncate", "drop"))

    def test_priority_order_respected(self):
        # Declared OUT of priority order on purpose. The allocator should still
        # process persona+hard_policies first; history gets what is left.
        # Make history too big to fit in whatever is left after the criticals.
        r = allocate(
            [self.HISTORY, self.PERSONA, self.POLICIES],
            {"persona": "p" * 40, "hard_policies": "q" * 80,
             "history": "h" * 4000},  # 1000 tok, way more than the leftover
            available=80,
        )
        self.assertIsNone(r.aborted)
        actions = {a.spec.id: a.action for a in r.slots}
        self.assertEqual(actions["persona"], "full")
        self.assertEqual(actions["hard_policies"], "full")
        self.assertIn(actions["history"], ("truncate", "drop"))


class TestGuardrails(unittest.TestCase):
    def test_secret_in_assembled_payload_blocks(self):
        slots = {"user_input": "my key is sk-ABCDEFGHIJKLMNOPQRSTUV please help"}
        g = [{"id": "no-secrets", "type": "regex_deny", "scope": "all",
              "patterns": [r"sk-[A-Za-z0-9]{20,}"], "on_fail": "abort"}]
        r = run_guardrails(g, slots)
        self.assertFalse(r.passed)
        self.assertEqual(r.failed()[0].id, "no-secrets")

    def test_clean_payload_passes(self):
        slots = {"user_input": "what is the weather in Madrid?"}
        g = [{"id": "no-secrets", "type": "regex_deny", "scope": "all",
              "patterns": [r"sk-[A-Za-z0-9]{20,}"], "on_fail": "abort"}]
        r = run_guardrails(g, slots)
        self.assertTrue(r.passed)

    def test_json_schema_guardrail_validates_tool_output(self):
        slots = {"tool_results": json.dumps({"tool": "search", "ok": True, "data": ["a"]})}
        g = [{"id": "shape", "type": "json_schema", "scope": "slot:tool_results",
              "schema_inline": {"type": "object", "required": ["tool", "ok"]},
              "on_fail": "abort"}]
        r = run_guardrails(g, slots)
        self.assertTrue(r.passed)

    def test_json_schema_guardrail_rejects_malformed(self):
        slots = {"tool_results": json.dumps({"tool": "search"})}     # missing "ok"
        g = [{"id": "shape", "type": "json_schema", "scope": "slot:tool_results",
              "schema_inline": {"type": "object", "required": ["tool", "ok"]},
              "on_fail": "abort"}]
        r = run_guardrails(g, slots)
        self.assertFalse(r.passed)
        self.assertIn("ok", r.failed()[0].detail)

    def test_unknown_guardrail_type_fails_closed(self):
        slots = {"x": "y"}
        g = [{"id": "mystery", "type": "magic_check", "on_fail": "abort"}]
        r = run_guardrails(g, slots)
        self.assertFalse(r.passed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
