"""Interactive chat loop for the agent.

The canonical entry point. Reads user input from stdin, drives run_turn,
carries plan / scratchpad / memory across turns, persists state.

Commands (start with /):
  /quit   exit the loop
  /reset  clear history + memory + index, start fresh
  /memory show what''s in long-term memory
  /tools  list registered tools + their response_schemas
  /audit  path of the latest audit log entry
  /help   print this list
  /config show the current contract name, model, and embedder
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Callable

from runtime.soft.agent import TurnResult, run_turn
from runtime.soft.assembler import load_contract
from runtime.soft.embeddings import build_lmstudio_retriever
from runtime.soft.lms import warmup_embeddings
from runtime.soft.memory import (
    HashingRetriever, Memory, load_index, save_index,
)
from runtime.hard.statelock import state_lock


class ChatState:
    """Holds the bits the loop carries across turns. Mutated in place."""

    def __init__(self, contract, memory, retriever, tools,
                 plan: str = "", scratch: str = "",
                 history: list[dict] | None = None,
                 state_dir: Path | None = None,
                 rag_index=None, rag_k: int = 4,
                 passphrase: str | None = None):
        """Build a chat state.

        `rag_index`: optional RAGIndex. When set, every user turn
        automatically queries it with the user''s text and the rendered
        context replaces the default empty tool_results slot. This is
        how the chat loop auto-injects relevant docs without changing
        the contract.
        `rag_k`: how many chunks to retrieve (default 4).
        """
        self.contract = contract
        self.memory = memory
        self.retriever = retriever
        self.tools = tools
        self.plan = plan
        self.scratch = scratch
        self.history = history or []
        self.state_dir = state_dir
        self.last_audit_path: Path | None = None
        self.pending_confirm: dict | None = None  # populated by run_turn when outcome == "awaiting_confirm"
        self._pending_decision: str | None = None  # "confirm" or "deny" for the next turn
        self.rag_index = rag_index
        self.rag_k = rag_k
        # Encryption at rest for persisted state. Reads PI_STATE_PASSPHRASE by
        # default so callers get it for free; None means plaintext (unchanged).
        self.passphrase = passphrase if passphrase is not None else os.environ.get("PI_STATE_PASSPHRASE")

    def persist(self) -> None:
        if self.state_dir is None:
            return
        self.state_dir.mkdir(parents=True, exist_ok=True)
        mem_path = self.state_dir / "memory.json"
        idx_path = self.state_dir / "index.json"
        # Serialize the whole write against other processes sharing this
        # state dir; each file is also written atomically (temp + replace),
        # so a crash mid-persist never leaves a torn JSON file.
        with state_lock(self.state_dir):
            self.memory.save(mem_path, passphrase=self.passphrase)
            # Sync the retriever with whatever the model just wrote.
            self.retriever._items.clear()
            self.retriever._vecs.clear()
            for it in self.memory.items:
                self.retriever.add(it)
            save_index(idx_path, self.retriever._items.values(), passphrase=self.passphrase)


def _format_reply(result: TurnResult) -> str:
    """Format a turn result for the user. The reply is already sanitized."""
    if result.outcome == "blocked_by_guardrail":
        who = "guardrail"
        for v in result.audit_record.get("guardrail_verdicts", []):
            if not v["passed"]:
                who = v["id"]
        return f"[blocked by {who}]"
    if result.outcome == "aborted":
        return f"[aborted: {result.audit_record.get('reason', '?')}]"
    if result.outcome == "tool_error":
        return f"[tool error: {result.audit_record.get('tool_error', '?')}]"
    if result.outcome == "awaiting_confirm":
        pc = result.pending_confirm or {}
        return ("[awaiting confirm] " + str(pc.get("name", "?")) + "(" + json.dumps(pc.get("args", {}), ensure_ascii=False) + ") -- reply /confirm or /deny")
    return result.user_message or "(no reply)"


def _format_debug_line(result: TurnResult) -> str:
    """One-line summary of what happened, for the (debug) line after a reply."""
    rec = result.audit_record
    bits: list[str] = []
    if rec.get("tool_called"):
        bits.append("tools=" + ",".join(rec["tool_called"]))
    if rec.get("sanitization"):
        bits.append("sanitized=" + rec["sanitization"]["summary"])
    if rec.get("memory_delta"):
        d = rec["memory_delta"]
        ops = len(d.get("ops") or [])
        bits.append("memory_ops=" + str(ops))
    payload = (rec.get("payload_sha256") or "")[:8]
    if payload:
        bits.append("payload=" + payload)
    return "  (" + "  ".join(bits) + ")" if bits else ""


def handle_command(state: ChatState, line: str, read_fn) -> bool:
    """Handle a /command. Return True to exit the loop, False to continue."""
    cmd = line.strip().lower().split()
    if not cmd:
        return False
    name = cmd[0]

    if name in ("/confirm", "/yes"):
        if state.pending_confirm is None:
            print("(no pending tool call to confirm)")
            return False
        state._pending_decision = "confirm"
        return False
    if name in ("/deny", "/no"):
        if state.pending_confirm is None:
            print("(no pending tool call to deny)")
            return False
        state._pending_decision = "deny"
        return False

    if name in ("/quit", "/exit", "/q"):
        return True

    if name == "/help":
        print("/quit   exit")
        print("/reset  clear history + memory + index, start fresh")
        print("/memory show long-term memory")
        print("/tools  list registered tools")
        print("/audit  path of the latest audit log entry")
        print("/config show contract name, model, embedder")
        print("/confirm confirm a pending tool call")
        print("/deny    deny a pending tool call")
        print("/help    this list")
        return False

    if name == "/reset":
        state.memory = Memory()
        state.retriever._items.clear()
        state.retriever._vecs.clear()
        state.history = []
        state.plan = ""
        state.scratch = ""
        if state.state_dir:
            for p in (state.state_dir / "memory.json",
                      state.state_dir / "index.json"):
                if p.exists():
                    p.unlink()
        print("(memory cleared)")
        return False

    if name == "/memory":
        if not state.memory.summary and not state.memory.items:
            print("(empty)")
        else:
            if state.memory.summary:
                print("summary: " + state.memory.summary)
            for it in state.memory.items:
                print("  " + it.key + ": " + it.value)
        return False

    if name == "/tools":
        from runtime.hard.tools import specs_from_contract
        specs = specs_from_contract(state.contract)
        if not specs:
            print("(no tools declared in the contract)")
        for s in specs:
            print("- " + s.name + ": " + s.description)
            print("  schema: " + json.dumps(s.schema, ensure_ascii=False))
        return False

    if name == "/audit":
        if state.last_audit_path and state.last_audit_path.exists():
            print(str(state.last_audit_path))
        else:
            print("(no audit entry yet)")
        return False

    if name == "/config":
        print("contract: " + state.contract.get("contract", {}).get("name", "?"))
        print("model:    " + (os.environ.get("PI_LLM_MODEL", "stub")))
        print("embed:    " + (os.environ.get("PI_EMBED_MODEL", "(hashing)")))
        print("dim:      " + str(getattr(state.retriever, "_embed_dim", "(full)")))
        return False

    print("(unknown command: " + name + "; try /help)")
    return False


def build_turn_input(state: ChatState, user_input: str) -> dict:
    """Build the slot contents for one turn, including history and prior plan/scratch.

    If `state.rag_index` is set, the user''s input is queried against the
    RAG index and the rendered context (with source provenance) is
    dropped into the `tool_results` slot. The LLM reads it as if it
    came from a search tool and cites the source by name.
    """
    history_str = "\n".join(
        m["role"] + ": " + m["content"] for m in state.history[-20:]
    )

    # RAG: query the index unless the user is typing a /command (which
    # the caller handles before reaching here) or the input is a
    # confirmation/denial (in which case we don''t want to re-query on
    # the user''s behalf).
    if state.rag_index is not None and user_input and not user_input.startswith("/"):
        rendered = state.rag_index.render_for_prompt(user_input, k=state.rag_k)
        if rendered:
            # Wrap the rendered text as a synthetic tool_results payload
            # so the LLM treats it as a search response.
            tool_results = json.dumps({
                "tool": "rag_search",
                "ok": True,
                "data": [{"query": user_input, "context": rendered}],
            })
        else:
            tool_results = json.dumps({"tool": "none", "ok": True, "data": []})
    else:
        tool_results = json.dumps({"tool": "none", "ok": True, "data": []})
    return {
        "persona": _persona_str(state.contract),
        "hard_policies": "\n".join(
            "- " + p for p in state.contract.get("contract", {}).get("hard_policies", [])),
        "long_term_mem": state.memory.render(),
        "plan": state.plan,
        "scratchpad": state.scratch,
        "tool_results": tool_results,
        "history": history_str,
        "user_input": user_input,
    }


def _persona_str(contract: dict) -> str:
    p = contract.get("contract", {}).get("persona", {})
    if not p:
        return ""
    return (f"Eres {p.get('name', 'el agente')}. "
            f"Tono: {p.get('tone', '')}. "
            f"Postura: {p.get('stance', '')}.")


def run_forever(state: ChatState,
                read_fn: Callable[[str], str] = input,
                write_fn: Callable[[str], None] = print,
                audit_dir: Path | None = None,
                memory_k: int = 4) -> None:
    """Drive the chat loop until /quit or EOF.

    `read_fn` and `write_fn` are injectable for tests. `audit_dir` defaults
    to `state.state_dir / "audit"`. `memory_k` controls the retriever top-k.
    """
    audit_dir = audit_dir or (state.state_dir / "audit" if state.state_dir else None)
    if audit_dir:
        audit_dir.mkdir(parents=True, exist_ok=True)

    write_fn("pi-style-agent chat. Type /help for commands, /quit to exit.")

    while True:
        try:
            line = read_fn("you> ")
        except EOFError:
            write_fn("(EOF)")
            return
        if line is None:
            return
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("/"):
            if handle_command(state, line, read_fn):
                return
            continue

        # If the user typed /confirm or /deny, we need to use the
        # pending_confirm from the previous turn. The user''s typed
        # command is the trigger; we don''t need a separate "user input".
        pending = state.pending_confirm
        decision = getattr(state, "_pending_decision", None)
        if decision is not None and pending is not None:
            state._pending_decision = None
            state.pending_confirm = None
            # We use the user''s "line" (the /confirm or /deny text) as
            # the user_input for this turn so the LLM has something to
            # respond to. The runner will dispatch the pending call.
            contents = build_turn_input(state, line)
            result = run_turn(
                state.contract, contents,
                tools=state.tools,
                memory=state.memory,
                memory_retriever=state.retriever,
                memory_k=memory_k,
                tool_depth_cap=3,
                audit_dir=audit_dir,
                pending_confirm=pending,
                confirm_decision=decision,
            )
        else:
            contents = build_turn_input(state, line)
            result = run_turn(
                state.contract, contents,
                tools=state.tools,
                memory=state.memory,
                memory_retriever=state.retriever,
                memory_k=memory_k,
                tool_depth_cap=3,
                audit_dir=audit_dir,
            )
        # Carry plan / scratch from this turn into the next.
        state.plan = result.plan_next or ""
        state.scratch = result.scratchpad_next or ""
        # If the turn produced a pending confirm, surface it for the user.
        if result.pending_confirm is not None:
            state.pending_confirm = result.pending_confirm
        state.history.append({"role": "user", "content": line})
        if result.user_message:
            state.history.append({"role": "assistant", "content": result.user_message})
        # Persist + record audit path.
        state.persist()
        # Audit file written by run_turn; find the most recent one.
        if audit_dir:
            files = sorted(audit_dir.glob("turn-*.json"))
            if files:
                state.last_audit_path = files[-1]
        write_fn("iris> " + _format_reply(result))
        debug = _format_debug_line(result)
        if debug:
            write_fn(debug)
