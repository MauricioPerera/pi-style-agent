"""Assemble the prompt from slots, in declaration order.

The seam between hard and soft. Deterministic. Fully unit-testable.

What it does:
  1. Reads the contract and the current slot contents.
  2. Asks the hard layer to allocate budget by priority.
  3. Returns the assembled prompt (role-tagged), the dropped/failed slots, and
     a per-slot summary. No LLM call happens here.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime.hard.budget import (
    AllocationResult, SlotSpec, allocate, estimate_tokens,
)
from runtime.hard.guardrails import run_guardrails


@dataclass
class AssembledTurn:
    contract: dict
    system: str                  # persona + hard_policies + (optional) long_term_mem
    user_blocks: list[str]       # the dynamic / runtime slots, in priority order
    payload: str                 # the full payload, as one string
    payload_sha256: str
    allocation: AllocationResult
    guardrails_passed: bool
    guardrail_verdicts: list[dict]


def load_contract(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assemble(contract: dict, contents: dict[str, str],
             memory_retriever=None, memory_k: int | None = None) -> AssembledTurn:
    """Build the prompt for one turn.

    `memory_retriever` + `memory_k` (optional): if the `long_term_mem` slot is
    backed by a `Memory` object (passed in `contents["_memory"]`) and both
    are supplied, render only the summary + top-k items by similarity to the
    current user_input. Otherwise the slot is treated as opaque text.
    """
    specs = [_spec(s) for s in contract["slots"]]
    budget = contract["budget"]
    available = budget["max_input_tokens"] - budget.get("reserve_output_tokens", 0)

    # Optional: if the caller passed a Memory in contents["_memory"], render
    # it through the retriever and overwrite the long_term_mem slot's text.
    mem = contents.pop("_memory", None)
    if mem is not None and (memory_retriever is not None and memory_k is not None):
        contents = dict(contents)
        query = contents.get("user_input", "")
        contents["long_term_mem"] = mem.render(
            query=query, retriever=memory_retriever, k=memory_k)

    result = allocate(specs, contents, available)
    if result.aborted:
        raise AssemblyError(result.aborted)

    by_id = {a.spec.id: a for a in result.slots}

    # System = the critical static slots (persona, hard_policies) + long-term
    # memory, if it survived. Anything dropped is omitted, not silently nulled.
    system_parts: list[str] = []
    for sid in ("persona", "hard_policies", "long_term_mem"):
        if sid in by_id and by_id[sid].action != "drop":
            system_parts.append(f"<<{sid}>>\n{by_id[sid].text}")

    # Persona + policies are the only ones that MUST be present.
    persona = contract.get("persona", {})
    if persona:
        system_parts.insert(0, (
            f"Eres {persona.get('name', 'el agente')}. "
            f"Tono: {persona.get('tone', '')}. "
            f"Postura: {persona.get('stance', '')}."
        ))

    system_text = "\n\n".join(p for p in system_parts if p.strip())

    # User-side blocks: everything else, in declared order, with priority tags
    # so the model can see what was truncated.
    user_blocks: list[str] = []
    for slot in contract["slots"]:
        sid = slot["id"]
        if sid in ("persona", "hard_policies"):
            continue
        a = by_id.get(sid)
        if a is None or a.action == "drop":
            user_blocks.append(f"<<{sid}>> [dropped: no budget]")
            continue
        tag = "[truncated]" if a.action == "truncate" else ""
        user_blocks.append(f"<<{sid}>> {tag}\n{a.text}")

    payload = system_text + "\n\n" + "\n\n".join(user_blocks) if system_text else "\n\n".join(user_blocks)
    payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # Run guardrails against the truncated slots. If the contract is well-formed
    # this is cheap; we already have the data in memory.
    assembled_map = {sid: (by_id[sid].text if sid in by_id else "") for sid in contents}
    g = run_guardrails(contract.get("guardrails", []), assembled_map)

    return AssembledTurn(
        contract=contract,
        system=system_text,
        user_blocks=user_blocks,
        payload=payload,
        payload_sha256=payload_hash,
        allocation=result,
        guardrails_passed=g.passed,
        guardrail_verdicts=[{
            "id": v.id, "passed": v.passed, "detail": v.detail,
            **({"reroute_to": v.reroute_to} if v.reroute_to else {}),
        } for v in g.verdicts],
    )


def _spec(s: dict) -> SlotSpec:
    return SlotSpec(
        id=s["id"],
        priority=s["priority"],
        kind=s["kind"],
        compaction=s["compaction"],
        min_tokens=s.get("min_tokens", 0),
        max_tokens=s.get("max_tokens"),
    )


class AssemblyError(RuntimeError):
    pass

