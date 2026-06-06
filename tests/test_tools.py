"""Tool registry tests. Hard layer, no LLM, no network."""
import unittest
from typing import Any

from runtime.hard.tools import (
    ToolError, ToolSpec, specs_from_contract, tool_descriptions, validate_response, _validate,
)


SEARCH = ToolSpec(
    name="search",
    description="Search a small corpus.",
    schema={
        "type": "object",
        "required": ["tool", "ok", "data"],
        "properties": {
            "tool": {"type": "string", "minLength": 1},
            "ok":   {"type": "boolean"},
            "data": {"type": "array",  "items": {"type": "string"}},
        },
    },
)


class TestSchemaValidation(unittest.TestCase):
    def test_object_ok(self):
        v = _validate({"a": 1, "b": "x"}, {"type": "object", "required": ["a"]})
        self.assertIsNone(v)

    def test_missing_required(self):
        v = _validate({"a": 1}, {"type": "object", "required": ["a", "b"]})
        self.assertIn("b", v)

    def test_wrong_type(self):
        v = _validate("hello", {"type": "object"})
        self.assertIn("object", v)

    def test_array_items_validated(self):
        v = _validate([1, "x", 2], {"type": "array", "items": {"type": "integer"}})
        self.assertIn("items[1]", v)

    def test_integer_rejects_bool(self):
        self.assertIsNotNone(_validate(True, {"type": "integer"}))
        self.assertIsNotNone(_validate(False, {"type": "number"}))
        self.assertIsNone(_validate(1, {"type": "integer"}))
        self.assertIsNone(_validate(1.5, {"type": "number"}))

    def test_nested_object(self):
        v = _validate({"x": {"y": 5}}, {
            "type": "object",
            "properties": {"x": {"type": "object",
                                  "properties": {"y": {"type": "integer"}}}},
        })
        self.assertIsNone(v)
        v2 = _validate({"x": {"y": "oops"}}, {
            "type": "object",
            "properties": {"x": {"type": "object",
                                  "properties": {"y": {"type": "integer"}}}},
        })
        self.assertIn("x.y", v2)


class TestValidateResponse(unittest.TestCase):
    def test_search_passes(self):
        ok = {"tool": "search", "ok": True, "data": ["a", "b"]}
        self.assertEqual(validate_response(SEARCH, ok), ok)

    def test_search_rejects_missing_data(self):
        with self.assertRaises(ToolError):
            validate_response(SEARCH, {"tool": "search", "ok": True})

    def test_search_rejects_wrong_data_type(self):
        with self.assertRaises(ToolError):
            validate_response(SEARCH, {"tool": "search", "ok": True, "data": "not an array"})


class TestContractIntegration(unittest.TestCase):
    def test_specs_from_contract_loads(self):
        import json
        from pathlib import Path
        contract = json.loads(Path("contracts/agent-contract.json").read_text(encoding="utf-8"))
        specs = specs_from_contract(contract)
        self.assertEqual([s.name for s in specs], ["search", "calculator"])
        # Every spec can validate a hand-crafted response
        for s in specs:
            sample = {"tool": s.name, "ok": True, "data": []}
            if s.name == "calculator":
                sample["data"] = {"value": 42}
            validate_response(s, sample)

    def test_tool_descriptions_is_deterministic(self):
        import json
        from pathlib import Path
        contract = json.loads(Path("contracts/agent-contract.json").read_text(encoding="utf-8"))
        a = tool_descriptions(specs_from_contract(contract))
        b = tool_descriptions(specs_from_contract(contract))
        self.assertEqual(a, b)
        self.assertIn("search", a)
        self.assertIn("calculator", a)


if __name__ == "__main__":
    unittest.main(verbosity=2)
