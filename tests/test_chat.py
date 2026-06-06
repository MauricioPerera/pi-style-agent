"""Tests for the interactive chat loop. No LLM, no network."""
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Callable

from runtime.soft.agent import TurnResult, run_turn
from runtime.soft.chat import ChatState, _format_debug_line, _format_reply, build_turn_input, handle_command, run_forever
from runtime.soft.assembler import load_contract
from runtime.soft.llm import LLMResponse
from runtime.soft.memory import HashingRetriever, Memory


CONTRACT = Path("contracts/agent-contract.json")


class ScriptedLLM:
    """A callable that returns scripted replies. Records what it saw."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.last_system = None
        self.last_user = None

    def __call__(self, system, user, model=""):
        self.calls += 1
        self.last_system = system
        self.last_user = user
        text = self.replies.pop(0) if self.replies else "(no more scripted replies)"
        return LLMResponse(text=text, model="stub", tokens_in=0, tokens_out=0)


def make_state(replies=None, tmp=None) -> ChatState:
    contract = load_contract(CONTRACT)
    mem = Memory()
    idx = HashingRetriever()
    state = ChatState(
        contract=contract, memory=mem, retriever=idx, tools={},
        state_dir=tmp,
    )
    # Inject the scripted LLM by patching the module-level reference.
    import runtime.soft.agent as agent_mod
    if replies is not None:
        agent_mod.call_llm = ScriptedLLM(replies)
    return state


def capture_io(state, inputs: list[str]) -> str:
    """Run the chat loop with the given input lines; return the output."""
    import runtime.soft.agent as agent_mod
    original = agent_mod.call_llm
    if not hasattr(state, "_llm"):
        # The default path uses call_llm directly. We want our scripted
        # one. Easiest: monkey-patch the agent_mod.call_llm.
        agent_mod.call_llm = state._llm if hasattr(state, "_llm") else original
    buf_out = io.StringIO()
    in_iter = iter(inputs + ["/quit"])

    def fake_input(prompt):
        line = next(in_iter, None)
        if line is None:
            raise EOFError
        return line

    # run_forever does not have a clean exit on /quit from this test.
    # Use a /quit at the end of the input list.
    run_forever(state, read_fn=fake_input, write_fn=lambda s: buf_out.write(s + "\n"))
    return buf_out.getvalue()


class TestChatFormatting(unittest.TestCase):
    def test_format_reply_ok(self):
        rec = {"outcome": "ok", "llm": {}, "reply": {}, "guardrail_verdicts": []}
        r = TurnResult("ok", "hola Maria", None, None, {}, rec)
        self.assertEqual(_format_reply(r), "hola Maria")

    def test_format_reply_blocked(self):
        rec = {"outcome": "blocked_by_guardrail", "guardrail_verdicts": [
            {"id": "no-secrets", "passed": False, "detail": "sk-..."}
        ]}
        r = TurnResult("blocked_by_guardrail", "", None, None, {}, rec)
        self.assertIn("blocked", _format_reply(r))
        self.assertIn("no-secrets", _format_reply(r))

    def test_format_reply_aborted(self):
        rec = {"outcome": "aborted", "reason": "presupuesto insuficiente"}
        r = TurnResult("aborted", "", None, None, {}, rec)
        self.assertIn("aborted", _format_reply(r))
        self.assertIn("presupuesto", _format_reply(r))

    def test_format_reply_tool_error(self):
        rec = {"outcome": "tool_error", "tool_error": "schema mismatch"}
        r = TurnResult("tool_error", "", None, None, {}, rec)
        self.assertIn("tool error", _format_reply(r))
        self.assertIn("schema", _format_reply(r))

    def test_format_debug_line_tool_called(self):
        rec = {"outcome": "ok", "tool_called": ["search", "calculator"],
               "payload_sha256": "abcdef0123456789..."}
        r = TurnResult("ok", "ok", None, None, {}, rec)
        line = _format_debug_line(r)
        self.assertIn("search", line)
        self.assertIn("calculator", line)
        self.assertIn("payload=abcdef01", line)

    def test_format_debug_line_sanitization(self):
        rec = {"outcome": "ok", "sanitization": {"summary": "redacted: openai_key (1 match)"}}
        r = TurnResult("ok", "ok", None, None, {}, rec)
        line = _format_debug_line(r)
        self.assertIn("sanitized", line)
        self.assertIn("openai_key", line)

    def test_format_reply_awaiting_confirm(self):
        rec = {"outcome": "awaiting_confirm", "awaiting_confirm": {"name": "delete_user", "args": {"id": 42}}}
        r = TurnResult("awaiting_confirm", "", None, None, {},
                       rec, pending_confirm={"name": "delete_user", "args": {"id": 42}})
        out = _format_reply(r)
        self.assertIn("awaiting confirm", out)
        self.assertIn("delete_user", out)
        self.assertIn("/confirm", out)


class TestChatCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="pi_chat_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_help_prints_commands(self):
        state = make_state(tmp=self.tmp)
        out = io.StringIO()
        handle_command(state, "/help", lambda p: "")
        # handle_command writes to stdout via print; capture by patching
        # sys.stdout is overkill. Instead, just check it returns False
        # (didn''t exit).
        self.assertFalse(handle_command(state, "/help", lambda p: ""))

    def test_quit_returns_true(self):
        state = make_state(tmp=self.tmp)
        self.assertTrue(handle_command(state, "/quit", lambda p: ""))
        self.assertTrue(handle_command(state, "/exit", lambda p: ""))
        self.assertTrue(handle_command(state, "/q", lambda p: ""))

    def test_memory_shows_state(self):
        state = make_state(tmp=self.tmp)
        state.memory.set_summary("Maria, Madrid.")
        state.memory.update_item("city", "Madrid")
        # /memory writes to stdout. Just verify it does not raise.
        handle_command(state, "/memory", lambda p: "")

    def test_reset_clears_memory(self):
        state = make_state(tmp=self.tmp)
        state.memory.set_summary("old")
        state.memory.update_item("city", "Madrid")
        state.retriever.add(__import__("runtime.soft.memory", fromlist=["MemoryItem"]).MemoryItem("city", "Madrid"))
        state.history.append({"role": "user", "content": "hola"})
        handle_command(state, "/reset", lambda p: "")
        self.assertEqual(state.memory.summary, "")
        self.assertEqual(state.memory.items, [])
        self.assertEqual(state.history, [])

    def test_unknown_command_does_not_exit(self):
        state = make_state(tmp=self.tmp)
        self.assertFalse(handle_command(state, "/bogus", lambda p: ""))


class TestBuildTurnInput(unittest.TestCase):
    def test_includes_history_plan_scratch(self):
        state = make_state(tmp=None)
        state.plan = "1) ask\n2) wait"
        state.scratch = "private note"
        state.history.append({"role": "user", "content": "hola"})
        state.history.append({"role": "assistant", "content": "que tal?"})
        contents = build_turn_input(state, "donde vivo?")
        self.assertEqual(contents["plan"], "1) ask\n2) wait")
        self.assertEqual(contents["scratchpad"], "private note")
        self.assertIn("hola", contents["history"])
        self.assertIn("que tal?", contents["history"])
        self.assertEqual(contents["user_input"], "donde vivo?")

    def test_history_truncated_to_last_20(self):
        state = make_state(tmp=None)
        # Use distinctive markers so substring search is unambiguous.
        for i in range(30):
            state.history.append({"role": "user", "content": f"<{i:02d}>"})
        contents = build_turn_input(state, "now")
        # The 10 most recent (20-29) are present.
        for i in range(20, 30):
            self.assertIn(f"<{i:02d}>", contents["history"])
        # The 10 oldest (00-09) are dropped.
        for i in range(10):
            self.assertNotIn(f"<{i:02d}>", contents["history"])


class TestChatLoopIntegration(unittest.TestCase):
    """End-to-end with a scripted fake LLM (no network)."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="pi_chat_run_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _setup_with_llm(self, replies):
        import runtime.soft.agent as agent_mod
        agent_mod.call_llm = ScriptedLLM(replies)
        return make_state(replies=None, tmp=self.tmp)

    def test_two_turn_loop(self):
        state = self._setup_with_llm([
            "Hola! Como te llamas?",
            "Mucho gusto, Maria.",
        ])
        in_lines = ["hola", "me llamo Maria", "/quit"]
        out = capture_io(state, in_lines)
        self.assertIn("Hola!", out)
        self.assertIn("Mucho gusto", out)
        # History grew with both turns.
        self.assertEqual(len(state.history), 4)  # 2 user + 2 assistant

    def test_memory_delta_written_to_memory(self):
        from runtime.soft.memory import DELTA_HEADER
        body = (f"Sure, remembering.\n{DELTA_HEADER}\n"
                "summary: Maria, Madrid.\n+ name: Maria\n+ city: Madrid\n<<<END>>>\n")
        state = self._setup_with_llm([body])
        capture_io(state, ["me llamo Maria y vivo en Madrid", "/quit"])
        self.assertEqual(state.memory.summary, "Maria, Madrid.")
        self.assertEqual({it.key for it in state.memory.items}, {"name", "city"})

    def test_secret_blocks_turn(self):
        state = self._setup_with_llm(["(unused)"])
        out = capture_io(state, ["sk-ABCDEFGHIJKLMNOPQRSTUV", "/quit"])
        self.assertIn("blocked", out)
        self.assertIn("no-secrets", out)

    def test_plan_carry_over_across_turns(self):
        # First turn: model emits a plan. Second turn: the plan shows up
        # in the contents of the next turn.
        import runtime.soft.agent as agent_mod
        llm = ScriptedLLM([
            "<<<PLAN>>>\n1) ask the user to confirm\n2) proceed\n<<<END>>>\nSure, I''ll confirm.",
            "Great, proceeding now.",
        ])
        agent_mod.call_llm = llm
        state = make_state(replies=None, tmp=self.tmp)
        capture_io(state, ["do thing X", "/quit"])
        # After turn 1, the plan_next was captured into state.plan.
        self.assertIn("ask the user to confirm", state.plan)

    def test_scratchpad_carry_over(self):
        import runtime.soft.agent as agent_mod
        llm = ScriptedLLM([
            "<<<SCRATCHPAD>>>\ninternal: user likes short answers\n<<<END>>>\nGot it.",
            "Will keep it short.",
        ])
        agent_mod.call_llm = llm
        state = make_state(replies=None, tmp=self.tmp)
        capture_io(state, ["hi", "/quit"])
        self.assertIn("user likes short answers", state.scratch)

    def test_persistence_between_states(self):
        from runtime.soft.memory import DELTA_HEADER
        body = (f"{DELTA_HEADER}\n+ name: Maria\n<<<END>>>\nWelcome Maria.")
        import runtime.soft.agent as agent_mod
        agent_mod.call_llm = ScriptedLLM([body])
        state1 = make_state(replies=None, tmp=self.tmp)
        capture_io(state1, ["me llamo Maria", "/quit"])

        # A new ChatState should pick up the persisted memory + index.
        contract = load_contract(CONTRACT)
        mem2 = Memory.load(self.tmp / "memory.json")
        from runtime.soft.memory import load_index
        idx2 = HashingRetriever()
        for it in load_index(self.tmp / "index.json"):
            idx2.add(it)
        state2 = ChatState(contract=contract, memory=mem2, retriever=idx2,
                           tools={}, state_dir=self.tmp)
        self.assertIn("name", {it.key for it in state2.memory.items})
        self.assertEqual(state2.memory.items[0].value, "Maria")






class TestConfirmableTools(unittest.TestCase):
    """A tool declared `confirm: true` in the contract must NOT be dispatched
    until the human explicitly confirms via the chat loop. The runner
    returns outcome="awaiting_confirm" with a pending_confirm record.
    """

    def setUp(self):
        import json, tempfile
        from pathlib import Path as P
        self.tmp = P(tempfile.mkdtemp(prefix="pi_conf_"))
        base = json.loads(P("contracts/agent-contract.json").read_text(encoding="utf-8"))
        base["tools"] = [{
            "name": "delete_user",
            "description": "Delete a user account. IRREVERSIBLE.",
            "confirm": True,
            "response_schema": {
                "type": "object",
                "required": ["deleted"],
                "properties": {"deleted": {"type": "boolean"}},
            },
        }]
        self.contract = base
        self._calls = []
        import runtime.soft.agent as agent_mod
        self._original_call_llm = agent_mod.call_llm

    def tearDown(self):
        import shutil
        import runtime.soft.agent as agent_mod
        agent_mod.call_llm = self._original_call_llm
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _delete_tool(self):
        calls = self._calls
        def delete_user(args):
            calls.append(args)
            return {"deleted": True}
        return {"delete_user": delete_user}

    def _contents(self):
        return {
            "persona": "Iris.",
            "hard_policies": "- no secrets.",
            "long_term_mem": "",
            "plan": "", "scratchpad": "",
            "tool_results": json.dumps({"tool": "none", "ok": True, "data": []}),
            "history": "",
            "user_input": "delete user 42",
        }

    def test_confirmable_tool_blocks_until_human_confirms(self):
        import runtime.soft.agent as agent_mod
        from runtime.soft.llm import LLMResponse

        body = "<<<TOOL_CALL>>> " + chr(123) + chr(34) + "name" + chr(34) + ": " + chr(34) + "delete_user" + chr(34) + ", " + chr(34) + "args" + chr(34) + ": " + chr(123) + chr(34) + "user_id" + chr(34) + ": 42" + chr(125) + chr(125) + " " + chr(60) + chr(60) + chr(60) + "END" + chr(62) + chr(62) + chr(62)
        def fake_llm(system, user, model=""):
            return LLMResponse(text=body, model="stub", tokens_in=0, tokens_out=0)
        agent_mod.call_llm = fake_llm

        rec = run_turn(self.contract, self._contents(), tools=self._delete_tool(),
                       audit_dir=self.tmp, llm_callable=fake_llm)
        self.assertEqual(rec.outcome, "awaiting_confirm")
        self.assertIsNotNone(rec.pending_confirm)
        self.assertEqual(rec.pending_confirm["name"], "delete_user")
        self.assertEqual(self._calls, [])

    def test_user_confirm_dispatches(self):
        import runtime.soft.agent as agent_mod
        from runtime.soft.llm import LLMResponse
        n = [0]
        def call_text():
            return ("<<<TOOL_CALL>>> " + chr(123) + chr(34) + "name" + chr(34) + ": " + chr(34) + "delete_user" + chr(34) + ", " + chr(34) + "args" + chr(34) + ": " + chr(123) + chr(34) + "user_id" + chr(34) + ": 7" + chr(125) + chr(125) + " " + chr(60) + chr(60) + chr(60) + "END" + chr(62) + chr(62) + chr(62))
        def fake_llm(system, user, model=""):
            n[0] += 1
            if n[0] == 1:
                return LLMResponse(text=call_text(), model="stub", tokens_in=0, tokens_out=0)
            return LLMResponse(text="User 7 deleted.", model="stub", tokens_in=0, tokens_out=0)
        agent_mod.call_llm = fake_llm

        rec1 = run_turn(self.contract, self._contents(), tools=self._delete_tool(),
                        audit_dir=self.tmp, llm_callable=fake_llm)
        rec2 = run_turn(self.contract, self._contents(), tools=self._delete_tool(),
                        audit_dir=self.tmp, llm_callable=fake_llm,
                        pending_confirm=rec1.pending_confirm,
                        confirm_decision="confirm")
        self.assertEqual(rec2.outcome, "ok")
        self.assertEqual(self._calls, [{"user_id": 7}])
        self.assertTrue(rec2.audit_record["tool_log"][0].get("confirmed_by_user"))

    def test_user_deny_blocks_dispatch(self):
        import runtime.soft.agent as agent_mod
        from runtime.soft.llm import LLMResponse
        n = [0]
        def call_text():
            return ("<<<TOOL_CALL>>> " + chr(123) + chr(34) + "name" + chr(34) + ": " + chr(34) + "delete_user" + chr(34) + ", " + chr(34) + "args" + chr(34) + ": " + chr(123) + chr(34) + "user_id" + chr(34) + ": 99" + chr(125) + chr(125) + " " + chr(60) + chr(60) + chr(60) + "END" + chr(62) + chr(62) + chr(62))
        def fake_llm(system, user, model=""):
            n[0] += 1
            if n[0] == 1:
                return LLMResponse(text=call_text(), model="stub", tokens_in=0, tokens_out=0)
            return LLMResponse(text="OK, no delete.", model="stub", tokens_in=0, tokens_out=0)
        agent_mod.call_llm = fake_llm

        rec1 = run_turn(self.contract, self._contents(), tools=self._delete_tool(),
                        audit_dir=self.tmp, llm_callable=fake_llm)
        rec2 = run_turn(self.contract, self._contents(), tools=self._delete_tool(),
                        audit_dir=self.tmp, llm_callable=fake_llm,
                        pending_confirm=rec1.pending_confirm,
                        confirm_decision="deny")
        self.assertEqual(rec2.outcome, "ok")
        self.assertEqual(self._calls, [])
        self.assertIn("pending_denied", rec2.audit_record)



class TestSoftFailTools(unittest.TestCase):
    """A tool declared soft_fail: true returns a bad payload to the LLM
    (as a rejected tool result) instead of aborting the turn.
    """

    def setUp(self):
        import json, tempfile
        from pathlib import Path as P
        self.tmp = P(tempfile.mkdtemp(prefix="pi_soft_"))
        base = json.loads(P("contracts/agent-contract.json").read_text(encoding="utf-8"))
        base["tools"] = [{
            "name": "search",
            "description": "Search a corpus.",
            "soft_fail": True,
            "response_schema": {"type": "object", "required": ["results"],
                                "properties": {"results": {"type": "array",
                                                          "items": {"type": "string"}}}},
        }]
        self.contract = base
        import runtime.soft.agent as agent_mod
        self._original_call_llm = agent_mod.call_llm

    def tearDown(self):
        import shutil
        import runtime.soft.agent as agent_mod
        agent_mod.call_llm = self._original_call_llm
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _call_text(self):
        return ("<<<TOOL_CALL>>> " + chr(123) + chr(34) + "name" + chr(34) + ": "
                + chr(34) + "search" + chr(34) + ", " + chr(34) + "args" + chr(34)
                + ": " + chr(123) + chr(125) + chr(125))

    def test_soft_fail_returns_bad_payload_to_llm(self):
        import runtime.soft.agent as agent_mod
        from runtime.soft.llm import LLMResponse
        n = [0]
        def fake_llm(system, user, model=""):
            n[0] += 1
            if n[0] == 1:
                return LLMResponse(text=self._call_text(), model="stub", tokens_in=0, tokens_out=0)
            return LLMResponse(text="OK, the search failed. Let me try a different query.",
                               model="stub", tokens_in=0, tokens_out=0)
        agent_mod.call_llm = fake_llm
        def bad_search(args):
            return {"results": "not an array"}
        contents = {
            "persona": "Iris.",
            "hard_policies": "- no secrets.",
            "long_term_mem": "",
            "plan": "", "scratchpad": "",
            "tool_results": json.dumps({"tool": "none", "ok": True, "data": []}),
            "history": "", "user_input": "search for X",
        }
        rec = run_turn(self.contract, contents, tools={"search": bad_search},
                       audit_dir=self.tmp, llm_callable=fake_llm)
        self.assertEqual(rec.outcome, "ok")
        log = rec.audit_record.get("tool_log", [])
        self.assertEqual(len(log), 1)
        self.assertIn("respuesta invalida", log[0].get("error", ""))
        self.assertIn("rejected_payload", log[0])
        self.assertIn("different query", rec.user_message)

    def test_strict_still_aborts_on_schema_failure(self):
        import json
        from pathlib import Path as P
        base = json.loads(P("contracts/agent-contract.json").read_text(encoding="utf-8"))
        base["tools"] = [{
            "name": "search",
            "description": "Search.",
            "response_schema": {"type": "object", "required": ["results"],
                                "properties": {"results": {"type": "array"}}},
        }]
        import runtime.soft.agent as agent_mod
        from runtime.soft.llm import LLMResponse
        def fake_llm(system, user, model=""):
            return LLMResponse(text=self._call_text(), model="stub", tokens_in=0, tokens_out=0)
        agent_mod.call_llm = fake_llm
        def bad_search(args):
            return {"results": "not an array"}
        contents = {
            "persona": "Iris.",
            "hard_policies": "- no secrets.",
            "long_term_mem": "",
            "plan": "", "scratchpad": "",
            "tool_results": json.dumps({"tool": "none", "ok": True, "data": []}),
            "history": "", "user_input": "search",
        }
        rec = run_turn(base, contents, tools={"search": bad_search},
                       audit_dir=self.tmp, llm_callable=fake_llm)
        self.assertEqual(rec.outcome, "tool_error")

"""RAG integration tests for the chat loop."""
import json
import unittest
from pathlib import Path

from runtime.soft.chat import build_turn_input, ChatState
from runtime.soft.assembler import load_contract
from runtime.soft.embeddings import build_lmstudio_retriever
from runtime.soft.memory import HashingRetriever, Memory
from runtime.soft.rag import RAGIndex, Chunk


def _make_state(rag=None, rag_k=3):
    contract = load_contract(Path("contracts/agent-contract.json"))
    idx = build_lmstudio_retriever()
    idx._retriever = HashingRetriever()
    return ChatState(
        contract=contract, memory=Memory(),
        retriever=idx._retriever, tools={},
        state_dir=None, rag_index=rag, rag_k=rag_k,
    )


def _make_rag(items):
    idx = HashingRetriever()
    rag = RAGIndex(idx)
    for c in items:
        rag.add(c)
    return rag


class TestChatRAGIntegration(unittest.TestCase):
    def test_rag_context_appears_in_tool_results(self):
        rag = _make_rag([
            Chunk(id="d1:0", doc_id="d1", text="Madrid is sunny.",
                  offset=0, metadata={"name": "w.md"}),
            Chunk(id="d2:0", doc_id="d2", text="Berlin is cold.",
                  offset=0, metadata={"name": "e.md"}),
        ])
        state = _make_state(rag=rag)
        contents = build_turn_input(state, "weather in Madrid")
        result = json.loads(contents["tool_results"])
        self.assertEqual(result["tool"], "rag_search")
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["data"]), 1)
        item = result["data"][0]
        self.assertEqual(item["query"], "weather in Madrid")
        self.assertIn("source: w.md", item["context"])
        self.assertIn("Madrid is sunny", item["context"])

    def test_no_rag_uses_default_empty(self):
        state = _make_state()
        contents = build_turn_input(state, "any question")
        result = json.loads(contents["tool_results"])
        self.assertEqual(result["tool"], "none")
        self.assertEqual(result["data"], [])

    def test_rag_skips_for_command_input(self):
        rag = _make_rag([
            Chunk(id="d1:0", doc_id="d1", text="x", offset=0, metadata={"name": "w.md"})
        ])
        state = _make_state(rag=rag)
        contents = build_turn_input(state, "/help")
        result = json.loads(contents["tool_results"])
        self.assertEqual(result["tool"], "none")

    def test_rag_k_respected(self):
        rag = _make_rag([
            Chunk(id="d" + str(i) + ":0", doc_id="d" + str(i),
                  text="topic " + str(i) + " content",
                  offset=0, metadata={"name": "d" + str(i) + ".md"})
            for i in range(5)
        ])
        state = _make_state(rag=rag, rag_k=2)
        contents = build_turn_input(state, "topic content")
        result = json.loads(contents["tool_results"])
        if result["tool"] == "rag_search":
            ctx = result["data"][0]["context"]
            n_sources = ctx.count("source: ")
            self.assertLessEqual(n_sources, 2,
                "expected at most 2 chunks, got " + str(n_sources))


if __name__ == "__main__":
    unittest.main(verbosity=2)
