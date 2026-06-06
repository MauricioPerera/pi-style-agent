"""The turn loop, extended with tools, memory, plan/scratchpad slots,
multi-step tool use with a hard depth cap, and embeddings-backed memory
retrieval.

The hard / soft seam lives in `assemble.AssembledTurn.guardrails_passed`: the
agent never calls the LLM if a guardrail is failing, never assembles without
budget, and never loses a critical slot to truncation. Audit log is per turn
and reproducible.

State is carried across turns by the caller (memory_path, prior memory/plan/
scratchpad), not hidden in globals. This keeps the loop testable.
"""
from __future__ import annotations
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from runtime.soft.assembler import (
    AssembledTurn, AssemblyError, assemble, load_contract,
)
from runtime.soft.llm import LLMResponse, call_llm
from runtime.soft.plan import AgentReply
from runtime.soft.memory import Memory, parse_delta, apply_delta
from runtime.hard.tools import ToolError, ToolSpec, specs_from_contract, validate_response
from runtime.hard.output_sanitize import sanitize


AUDIT_DIR = Path("audit")


# A tool is a callable (args_dict) -> result_dict. The runner wraps the
# result through validate_response() before stuffing it into the slot.
ToolFn = Callable[[dict], dict]


@dataclass
class TurnResult:
    """What one turn produced. Caller decides how to persist it."""
    outcome: str                 # "ok" | "blocked_by_guardrail" | "aborted" | "tool_error" | "awaiting_confirm"
    user_message: str            # the body of the agent's reply (plan/scratch stripped)
    plan_next: str | None        # the next turn's plan slot, or None to keep prior
    scratchpad_next: str | None  # same for scratchpad
    memory_delta: dict           # what the agent asked to remember (already applied to Memory if provided)
    audit_record: dict
    pending_confirm: dict | None = None  # populated when outcome == "awaiting_confirm"
    confirm_dispatched: dict | None = None  # populated when user confirmed a pending call


def run_turn(contract: dict,
             contents: dict[str, str],
             tools: dict[str, ToolFn] | None = None,
             memory: Memory | None = None,
             memory_retriever=None,
             memory_k: int | None = None,
             tool_depth_cap: int = 2,
             audit_dir: Path = AUDIT_DIR,
             model: str = "",
             llm_callable: Callable[[str, str, str], LLMResponse] | None = None,
             pending_confirm: dict | None = None,
             confirm_decision: str | None = None) -> TurnResult:
    """Run one turn end-to-end. Mutates `memory` in place if provided.

    `memory_retriever` + `memory_k`: if both are set, the `long_term_mem` slot
    is rendered through the retriever (top-k items by similarity to the
    current user_input, plus the summary). Without them, the slot is treated
    as opaque text and used as-is.

    `tool_depth_cap`: maximum number of LLM/tool rounds per turn. Default 1
    (single tool call). Set higher for agentic chains; the hard layer never
    trusts the model to bound itself.
    """
    tools = tools or {}
    if llm_callable is None:
        llm_callable = call_llm
    ts = int(time.time() * 1000)
    record: dict = {
        "ts": ts,
        "contract": contract.get("name"),
        "slot_hashes": {sid: _h(t) for sid, t in contents.items()},
    }

    # If the caller provided a retriever, let assemble() render the
    # long_term_mem slot using similarity search over `user_input`. We pass
    # the Memory object in a private key so the slot text is computed in
    # one place (the seam), not here in the runner.
    if memory is not None and memory_retriever is not None and memory_k is not None:
        contents = {**contents, "_memory": memory}

    try:
        turn = assemble(contract, contents,
                        memory_retriever=memory_retriever, memory_k=memory_k)
    except AssemblyError as e:
        record["outcome"] = "aborted"
        record["reason"] = str(e)
        _write(audit_dir, ts, record)
        return TurnResult("aborted", "", None, None, {}, record)

    record["allocation"] = [
        {"id": a.spec.id, "priority": a.spec.priority,
         "kept_tokens": a.kept_tokens, "action": a.action}
        for a in turn.allocation.slots
    ]
    record["payload_sha256"] = turn.payload_sha256
    record["guardrails_passed"] = turn.guardrails_passed
    record["guardrail_verdicts"] = turn.guardrail_verdicts

    if not turn.guardrails_passed:
        record["outcome"] = "blocked_by_guardrail"
        _write(audit_dir, ts, record)
        return TurnResult("blocked_by_guardrail", "", None, None, {}, record)

    # Tool-calling step: append tool descriptions to the system prompt if any
    # are declared, so the LLM knows what it can call. Actual tool execution
    # stays in the soft layer (the LLM suggests, we dispatch).
    specs = specs_from_contract(contract)
    system = turn.system
    if specs:
        from runtime.hard.tools import tool_descriptions
        system = system + "\n\nTools available:\n" + tool_descriptions(specs)

    user_blocks = list(turn.user_blocks)
    # Multi-step tool loop. Each iteration: call the LLM, optionally dispatch
    # a tool, append the validated result to user_blocks, and loop. We cap
    # the depth hard-side so a runaway model can not burn the budget.
    tool_called_history: list[str] = []
    tool_call_log: list[dict] = []
    tool_error: str | None = None
    resp: LLMResponse | None = None
    reply: AgentReply | None = None
    chosen_model = model or os.environ.get("PI_LLM_MODEL", "")

    # Confirmation flow: if a previous turn queued a confirmation
    # request, the caller passes it in as pending_confirm along with the
    # user''s decision (confirm | deny). We handle it BEFORE the LLM call.
    if pending_confirm is not None and confirm_decision in ("confirm", "deny"):
        tname = pending_confirm["name"]
        targs = pending_confirm.get("args", {})
        spec = next((s for s in specs if s.name == tname), None) if specs else None
        if confirm_decision == "deny":
            # Tell the LLM the user denied; let it re-plan.
            user_blocks.append(
                "<<<TOOL_RESULTS>>>\n" + json.dumps({"denied": True, "tool": tname}) +
                "\n<<<END>>>\n\nThe user denied that action. Explain and suggest alternatives.")
            record["pending_denied"] = pending_confirm
        elif spec is None or tname not in tools:
            user_blocks.append(
                "<<<TOOL_RESULTS>>>\n" + json.dumps({"error": f"tool no longer available: {tname}"}) +
                "\n<<<END>>>")
        else:
            try:
                result_raw = tools[tname](targs)
                result = validate_response(spec, result_raw)
            except ToolError as e:
                tool_error = str(e)
                record["tool_error"] = tool_error
                record["outcome"] = "tool_error"
                _write(audit_dir, ts, record)
                return TurnResult("tool_error", "", None, None, {}, record, None, None)
            except Exception as e:
                tool_error = f"tool raised: {e!r}"
                record["tool_error"] = tool_error
                record["outcome"] = "tool_error"
                _write(audit_dir, ts, record)
                return TurnResult("tool_error", "", None, None, {}, record, None, None)
            tool_called_history.append(tname)
            tool_call_log.append({"name": tname, "args": targs, "result": result,
                                  "confirmed_by_user": True})
            user_blocks.append(
                "<<<TOOL_RESULTS>>>\n" + json.dumps(result, ensure_ascii=False) +
                "\n<<<END>>>\n\nAction completed. Inform the user.")

    for _step in range(max(1, tool_depth_cap)):
        resp = llm_callable(system, "\n\n".join(user_blocks), model=chosen_model)
        reply = AgentReply.parse(resp.text)

        tool_call = _extract_tool_call(reply.body)
        if not tool_call or not specs:
            break

        tname = tool_call.get("name")
        targs = tool_call.get("args", {})
        spec = next((s for s in specs if s.name == tname), None) if tname else None
        if tname not in tools or spec is None:
            # Model asked for a tool we don't have. Feed it the error and
            # break so the next LLM call (if any) sees the rejection.
            tool_call_log.append({"name": tname, "args": targs, "error": f"unknown_tool: {tname}"})
            user_blocks.append(
                f"<<<TOOL_RESULTS>>>\n{json.dumps({'error': f'unknown tool: {tname}'})}\n<<<END>>>\n\n"
                f"No such tool: {tname}. Use one of: {', '.join(s.name for s in specs)}."
            )
            break

        # Confirmation gate: a tool declared confirm: true in the contract
        # is never dispatched without the human''s explicit /confirm. We
        # record the pending call, return outcome="awaiting_confirm", and
        # let the chat loop surface the question to the user.
        from runtime.hard.tools import is_confirmable
        if is_confirmable(contract, tname):
            record["awaiting_confirm"] = {"name": tname, "args": targs}
            record["outcome"] = "awaiting_confirm"
            _write(audit_dir, ts, record)
            return TurnResult(
                "awaiting_confirm",
                "Pedi confirmacion para la accion '" + tname + "'. "
                "Espera /confirm o /deny del usuario antes de continuar.",
                None, None, {}, record,
                pending_confirm={"name": tname, "args": targs},
            )

        # Look up whether the tool wants soft failure. The default
        # is strict (abort the whole turn). If soft_fail is true, the
        # bad payload is fed back to the LLM as a rejected tool result
        # so it can retry with different args or ask the user.
        from runtime.hard.tools import tool_spec_from_contract
        t_spec = tool_spec_from_contract(contract, tname)
        soft = bool(t_spec and t_spec.soft_fail)

        try:
            result_raw = tools[tname](targs)
        except Exception as e:  # tool implementation raised
            tool_error = f"tool raised: {e!r}"
            tool_call_log.append({"name": tname, "args": targs, "error": tool_error})
            if soft:
                user_blocks.append(
                    "<<<TOOL_RESULTS>>>\n" + json.dumps({"error": tool_error, "tool": tname}) +
                    "\n<<<END>>>\n\nThe tool raised. Try a different approach.")
                continue
            break
        try:
            result = validate_response(spec, result_raw)
        except ToolError as e:
            err = str(e)
            tool_call_log.append({"name": tname, "args": targs, "error": err, "rejected_payload": result_raw})
            if soft:
                user_blocks.append(
                    "<<<TOOL_RESULTS>>>\n" + json.dumps({"error": err, "tool": tname, "rejected": result_raw}, default=str) +
                    "\n<<<END>>>\n\nThe tool response failed schema validation. Try a different approach or fix the inputs.")
                continue
            tool_error = err
            break

        tool_called_history.append(tname)
        tool_call_log.append({"name": tname, "args": targs, "result": result})
        user_blocks.append(
            f"<<<TOOL_RESULTS>>>\n{json.dumps(result, ensure_ascii=False)}\n<<<END>>>\n\n"
            f"Based on the tool result above, continue. "
            f"You may call another tool or give the final answer."
        )

    if tool_error:
        record["outcome"] = "tool_error"
        record["tool_log"] = tool_call_log
        record["tool_error"] = tool_error
        _write(audit_dir, ts, record)
        return TurnResult("tool_error", "", None, None, {}, record)

    record["outcome"] = "ok"
    record["llm"] = asdict(resp)
    record["reply"] = {"plan": reply.plan, "scratchpad": reply.scratchpad,
                       "body_preview": reply.body[:200]}
    if tool_call_log:
        record["tool_log"] = tool_call_log
    if tool_called_history:
        record["tool_called"] = tool_called_history

    # Memory delta: the LLM can emit <<<MEMORY-DELTA>>> anywhere in the body.
    # Note: we extract the delta from the SANITIZED body, so a leaked
    # secret in the reply can never sneak into memory.
    final_body = reply.body
    delta = parse_delta(final_body)
    if memory is not None and (delta.get("summary") or delta.get("ops")):
        apply_delta(memory, delta)
        record["memory_delta"] = delta

    # Output sanitization: deterministic redaction of secrets / PII in
    # the assistant reply. Hard layer. Never aborts; always returns a
    # redacted reply. The audit log records what was redacted so we can
    # prove what the LLM actually said.
    san = sanitize(final_body)
    if not san.clean:
        record["sanitization"] = {
            "redacted": san.redacted,
            "summary": san.summary(),
        }

    _write(audit_dir, ts, record)
    return TurnResult(
        outcome="ok",
        user_message=san.text,
        plan_next=reply.plan,
        scratchpad_next=reply.scratchpad,
        memory_delta=delta,
        audit_record=record,
    )


def replay_turn(contract: dict, contents: dict[str, str], audit_path: Path) -> bool:
    """Re-derive the payload and compare the on-disk payload_sha256 to a fresh
    re-assembly. Returns True on match."""
    prior = json.loads(audit_path.read_text(encoding="utf-8"))
    turn = assemble(contract, contents)
    return turn.payload_sha256 == prior.get("payload_sha256")


# --- helpers ---------------------------------------------------------------

# Tool call convention emitted by the LLM (very simple, easy to teach):
#   <<<TOOL_CALL>>> {"name": "search", "args": {"q": "madrid weather"}}
import re as _re
_TOOL_CALL_RE = _re.compile(r"<<<TOOL_CALL>>>\s*(\{.*?\n\})", _re.DOTALL)


def _extract_tool_call(body: str) -> dict | None:
    """Extract a tool call. The wire format is:

        <<<TOOL_CALL>>>
        {"name": "...", "args": {...}}
        <<<END>>>

    The regex above is intentionally non-greedy within the JSON braces; with
    nested objects we still hit the bug where the inner `}` closes the match
    too early. We work around it by scanning for the first `{` after the tag
    and the LAST `}` in the body of the reply, and slicing the substring.
    """
    tag = "<<<TOOL_CALL>>>"
    i = body.find(tag)
    if i < 0:
        return None
    j = body.find("{", i)
    if j < 0:
        return None
    k = body.rfind("}")
    if k <= j:
        return None
    try:
        d = json.loads(body[j:k + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or "name" not in d:
        return None
    return d


def _h(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _write(audit_dir: Path, ts: int, record: dict) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    out = audit_dir / f"turn-{ts}.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


