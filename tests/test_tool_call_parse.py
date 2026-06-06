"""Unit tests for the tool-call extractor in the turn loop.

Locks in the balanced-brace parser that replaced the old `rfind("}")`
heuristic. The old version grabbed the LAST brace in the whole reply, so
nested args could merge and any trailing prose containing a `}` made the
JSON slice unparseable. These tests pin the correct behavior.
"""
from __future__ import annotations
import unittest

from runtime.soft.agent import _extract_tool_call, _matching_brace


class TestExtractToolCall(unittest.TestCase):
    def test_simple_call(self):
        body = '<<<TOOL_CALL>>>\n{"name": "search", "args": {"q": "madrid"}}\n<<<END>>>'
        self.assertEqual(
            _extract_tool_call(body),
            {"name": "search", "args": {"q": "madrid"}},
        )

    def test_deeply_nested_args(self):
        body = '<<<TOOL_CALL>>> {"name": "t", "args": {"a": {"b": {"c": 1}}}}'
        out = _extract_tool_call(body)
        self.assertEqual(out["args"]["a"]["b"]["c"], 1)

    def test_trailing_prose_with_stray_brace(self):
        # The killer case for the old rfind heuristic: a '}' after the call.
        body = ('<<<TOOL_CALL>>>\n{"name": "search", "args": {"q": "x"}}\n'
                '<<<END>>>\nGlad to help :} cheers')
        self.assertEqual(
            _extract_tool_call(body),
            {"name": "search", "args": {"q": "x"}},
        )

    def test_brace_inside_string_value(self):
        body = '<<<TOOL_CALL>>> {"name": "search", "args": {"q": "a } b { c"}}'
        self.assertEqual(_extract_tool_call(body)["args"]["q"], "a } b { c")

    def test_no_tag_returns_none(self):
        self.assertIsNone(_extract_tool_call("just a normal reply, no tool"))

    def test_unclosed_object_returns_none(self):
        self.assertIsNone(_extract_tool_call('<<<TOOL_CALL>>> {"name": "x"'))

    def test_missing_name_returns_none(self):
        self.assertIsNone(_extract_tool_call('<<<TOOL_CALL>>> {"args": {}}'))


class TestMatchingBrace(unittest.TestCase):
    def test_finds_simple_close(self):
        s = "{}"
        self.assertEqual(_matching_brace(s, 0), 1)

    def test_nested(self):
        s = '{"a": {"b": 1}}'
        self.assertEqual(_matching_brace(s, 0), len(s) - 1)

    def test_respects_escaped_quote_in_string(self):
        s = '{"q": "a \\" } still inside"}'
        self.assertEqual(_matching_brace(s, 0), len(s) - 1)

    def test_unclosed_returns_minus_one(self):
        self.assertEqual(_matching_brace('{"a": 1', 0), -1)


if __name__ == "__main__":
    unittest.main()
