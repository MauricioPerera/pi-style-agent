"""Single source of truth for the minimal JSON-Schema-ish validator.

Both the tool-response check (`tools.validate_response`) and the input
`json_schema` guardrail (`guardrails.run_guardrails`) validate against the
same shapes, so the validator lives here, once, in the hard layer. Before
this module there were two near-identical `_validate` copies — one recursive
(tools), one flat (guardrails) — which could disagree on the same schema.

Supported subset: type (object / array / string / integer / number /
boolean), `required`, `properties`, and array `items`. Validation recurses
into properties and items. Unknown types are permissive — layer `jsonschema`
on top if you need full Draft 2020-12.

Returns `None` on success, or a human-readable error string (with a `$`-rooted
path) on the first failure. Pure stdlib, deterministic, no model dependency.
"""
from __future__ import annotations
from typing import Any


def validate(data: Any, schema: dict, path: str = "$") -> str | None:
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
                err = validate(data[key], sub, f"{path}.{key}")
                if err is not None:
                    return err
    elif t == "array":
        if not isinstance(data, list):
            return f"{path}: esperaba array, recibi {type(data).__name__}"
        items = schema.get("items")
        if isinstance(items, dict):
            for i, v in enumerate(data):
                err = validate(v, items, f"{path}.items[{i}]")
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
