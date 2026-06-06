"""Tool registry: schema-declared tools, validated at the ingestion seam.

Tools are declared in the contract. The hard layer is the source of truth for
their JSON shape. Two things this module does:

  1. `validate_response(name, payload)`: run a tool response through the
     declared schema. Raises ToolError on failure. This is what the agent's
     tool-calling code wraps every tool result with, before stuffing it into
     the `tool_results` slot.

  2. `tool_descriptions()`: emit a deterministic, prompt-friendly summary of
     all tools + their schemas. The LLM uses this to decide which tool to
     call and with which arguments. Living here (not in the soft layer) means
     there is exactly one source of truth: the contract.

JSON schema support is intentionally minimal: type, required, properties
(string/number/integer/boolean/object/array). Enough for the two shipped
tools; swap for `jsonschema` if you need full Draft 2020-12.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any


class ToolError(RuntimeError):
    """Raised when a tool response fails its declared schema."""


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict        # JSON-Schema-ish; see _validate for the supported subset
    soft_fail: bool = False   # if True, schema violations are returned to the LLM instead of aborting


def specs_from_contract(contract: dict) -> list[ToolSpec]:
    out: list[ToolSpec] = []
    for t in contract.get("tools", []):
        out.append(ToolSpec(
            name=t["name"],
            description=t["description"],
            schema=t.get("response_schema", {"type": "object"}),
            soft_fail=bool(t.get("soft_fail", False)),
        ))
    return out


def is_confirmable(contract: dict, tool_name: str) -> bool:
    """Return True iff the named tool is declared as `confirm: true` in the
    contract. The runner uses this to decide whether to dispatch a tool
    call directly or to queue a confirmation request.
    """
    for t in contract.get("tools", []):
        if t.get("name") == tool_name and t.get("confirm") is True:
            return True
    return False


def list_confirmable(contract: dict) -> list[str]:
    """Names of all tools that require human confirmation."""
    return [t["name"] for t in contract.get("tools", []) if t.get("confirm") is True]


def validate_response(spec: ToolSpec, payload: Any) -> Any:
    """Raise ToolError if `payload` does not match `spec.schema`."""
    err = _validate(payload, spec.schema)
    if err is not None:
        raise ToolError(f"tool '{spec.name}': respuesta invalida: {err}")
    return payload


def tool_descriptions(specs: list[ToolSpec]) -> str:
    """Deterministic text block the LLM sees to know what tools exist."""
    if not specs:
        return "(no tools declared)"
    lines: list[str] = []
    for s in specs:
        lines.append(f"- {s.name}: {s.description}")
        lines.append(f"  response_schema: {json.dumps(s.schema, ensure_ascii=False)}")
    return "\n".join(lines)


# --- minimal validator -----------------------------------------------------

def _validate(data: Any, schema: dict, path: str = "$") -> str | None:
    if not isinstance(schema, dict):
        return f"{path}: schema no es un objeto"

    t = schema.get("type")
    if t == "object":
        if not isinstance(data, dict):
            return f"{path}: esperaba object, recibi {type(data).__name__}"
        for req in schema.get("required", []):
            if req not in data:
                return f"{path}: falta campo requerido '{req}'"
        for key, sub in schema.get("properties", {}).items():
            if key in data:
                err = _validate(data[key], sub, f"{path}.{key}")
                if err is not None:
                    return err
    elif t == "array":
        if not isinstance(data, list):
            return f"{path}: esperaba array, recibi {type(data).__name__}"
        items = schema.get("items")
        if isinstance(items, dict):
            for i, v in enumerate(data):
                err = _validate(v, items, f"{path}.items[{i}]")
                if err is not None:
                    return err
    elif t == "string":
        if not isinstance(data, str):
            return f"{path}: esperaba string, recibi {type(data).__name__}"
    elif t == "integer":
        if isinstance(data, bool) or not isinstance(data, int):
            return f"{path}: esperaba integer, recibi {type(data).__name__}"
    elif t == "number":
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            return f"{path}: esperaba number, recibi {type(data).__name__}"
    elif t == "boolean":
        if not isinstance(data, bool):
            return f"{path}: esperaba boolean, recibi {type(data).__name__}"
    else:
        # unknown type -> permissive (caller can layer jsonschema on top)
        return None
    return None



def tool_spec_from_contract(contract: dict, tool_name: str) -> ToolSpec | None:
    """Return the ToolSpec for the named tool, or None if not declared.

    Used by the runner to look up per-tool settings (soft_fail, schema)
    before each call. The spec list is built fresh each time so contract
    edits are reflected immediately; for high-throughput paths you''d
    cache it.
    """
    for spec in specs_from_contract(contract):
        if spec.name == tool_name:
            return spec
    return None
