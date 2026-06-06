"""Deterministic, pre-inference guardrails. Fail closed.

Mirrors CCDD''s `regex_deny` and `json_schema` guardrails, but lives in the
hard layer: it runs before the LLM is ever called. If something is wrong with
the input the agent refuses / reroutes without paying for inference.
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Any

from .secrets import compile_patterns, find_secrets


@dataclass
class GuardrailVerdict:
    id: str
    passed: bool
    detail: str
    reroute_to: str | None = None


@dataclass
class GuardrailResult:
    passed: bool
    verdicts: list[GuardrailVerdict]

    def failed(self) -> list[GuardrailVerdict]:
        return [v for v in self.verdicts if not v.passed]


def run_guardrails(contract_guardrails: list[dict],
                   assembled_slots: dict[str, str]) -> GuardrailResult:
    """Run every guardrail declared in the contract against the assembled slots.

    `assembled_slots` maps slot id -> text (post-truncation).
    """
    verdicts: list[GuardrailVerdict] = []
    overall_ok = True

    for g in contract_guardrails:
        gid = g["id"]
        gtype = g["type"]
        on_fail = g.get("on_fail", "abort")
        scope = g.get("scope", "all")

        if gtype == "regex_deny":
            patterns = compile_patterns(g.get("patterns", []))
            target = _scope_text(scope, assembled_slots)
            hits = find_secrets(target, patterns)
            if hits:
                v = GuardrailVerdict(
                    id=gid, passed=False,
                    detail=f"patron(es) prohibido(s) detectado(s): {len(hits)} match(s)",
                )
                overall_ok = False
            else:
                v = GuardrailVerdict(id=gid, passed=True, detail="limpio")
            verdicts.append(v)

        elif gtype == "json_schema":
            target_slot = scope.split(":", 1)[1] if scope.startswith("slot:") else None
            if not target_slot or target_slot not in assembled_slots:
                v = GuardrailVerdict(id=gid, passed=False, detail=f"slot '{target_slot}' no existe")
                overall_ok = False
                verdicts.append(v)
                continue
            raw = assembled_slots[target_slot]
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                v = GuardrailVerdict(id=gid, passed=False, detail=f"slot '{target_slot}' no es JSON: {e}")
                overall_ok = False
                verdicts.append(v)
                continue
            schema = g.get("schema_inline")
            if schema is None:
                v = GuardrailVerdict(id=gid, passed=False, detail="json_schema sin schema_inline")
                overall_ok = False
                verdicts.append(v)
                continue
            err = _validate(data, schema)
            if err:
                v = GuardrailVerdict(id=gid, passed=False, detail=f"slot '{target_slot}': {err}")
                if on_fail == "reroute":
                    v.reroute_to = g.get("reroute_to")
                overall_ok = False
            else:
                v = GuardrailVerdict(id=gid, passed=True, detail=f"slot '{target_slot}' valido")
            verdicts.append(v)
        else:
            # Unknown guardrail type: fail closed, never silently pass.
            v = GuardrailVerdict(
                id=gid, passed=False,
                detail=f"tipo de guardrail no soportado: {gtype}",
            )
            overall_ok = False
            verdicts.append(v)

    return GuardrailResult(passed=overall_ok, verdicts=verdicts)


def _scope_text(scope: str, assembled: dict[str, str]) -> str:
    if scope == "all":
        return "\n".join(assembled.values())
    if scope.startswith("slot:"):
        return assembled.get(scope.split(":", 1)[1], "")
    return ""


def _validate(data: Any, schema: dict) -> str | None:
    """Minimal JSON-schema-ish validator for the two shapes the contract uses.
    Good enough for a tool-output guardrail; swap for `jsonschema` if needed."""
    if not isinstance(schema, dict):
        return "schema no es un objeto"
    t = schema.get("type")
    if t == "object" and not isinstance(data, dict):
        return f"esperaba object, recibi {type(data).__name__}"
    if t == "array" and not isinstance(data, list):
        return f"esperaba array, recibi {type(data).__name__}"
    if t == "string" and not isinstance(data, str):
        return f"esperaba string, recibi {type(data).__name__}"
    if t == "boolean" and not isinstance(data, bool):
        return f"esperaba boolean, recibi {type(data).__name__}"
    for req in schema.get("required", []):
        if not isinstance(data, dict) or req not in data:
            return f"falta campo requerido '{req}'"
    return None
