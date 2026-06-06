# runtime/soft/

The soft layer. Calls the LLM, interprets its output, carries state
across turns. All the code that *reacts* to what the model says.

## Modules

- [agent.py](agent.py) — the turn loop. Assembles the prompt, runs
  the guardrails, calls the LLM, dispatches tools (with confirmation
  + soft-fail gates), sanitizes the output, applies the memory
  delta, writes the audit log. The only file that calls
  `run_turn()` and the only one that bridges hard + soft.
- [assembler.py](assembler.py) — the seam. `assemble(contract,
  contents)` turns the contract + slot contents into one string the
  LLM consumes. Calls into the retriever for memory rendering.
  Testable without an LLM.
- [chat.py](chat.py) — the interactive loop. `ChatState` carries
  plan, scratchpad, history, and pending confirmations across
  turns. Supports `/quit`, `/reset`, `/memory`, `/tools`,
  `/audit`, `/config`, `/confirm`, `/deny`, `/help`. Auto-queries
  a `RAGIndex` if one is attached.
- [llm.py](llm.py) — the LLM wrapper. `provider=stub` for tests,
  `openai` for the official client, `lmstudio` for a local server
  via `urllib` (no extra deps). Selection via `PI_LLM_PROVIDER`.
- [embeddings.py](embeddings.py) — the embeddings client and
  retriever factory for LM Studio''s `/v1/embeddings`. Supports
  Matryoshka truncation dim via `PI_EMBED_DIM` (default 256).
- [lms.py](lms.py) — thin wrapper around the `lms` CLI. Pre-loads
  a model and pings it to warm the connection. The demo calls
  this at startup to avoid the cold-start on the first turn.
- [memory.py](memory.py) — `Memory` (summary + items), `MemoryItem`
  (with `last_accessed`), `HashingRetriever`, `EmbeddingRetriever`,
  `DecayingRetriever`. The chat loop touches items after every
  retrieval so the decay wrapper keeps them fresh.
- [plan.py](plan.py) — parses `<<<PLAN>>>` and `<<<SCRATCHPAD>>>`
  tags out of the LLM''s reply. `AgentReply.plan` and
  `AgentReply.scratchpad` are what the chat loop carries to the
  next turn.
- [rag.py](rag.py) — `RAGIndex`. Wraps any retriever with a
  parallel dict of chunk metadata. `render_for_prompt(query, k)`
  returns a string with `[source: <doc_name>, score=…]` headers
  for attribution.

## State that lives here vs state that doesn''t

- **In `ChatState`:** plan, scratchpad, history, the most-recent
  audit path, the pending confirmation, the retriever, the RAG
  index, the `Memory`, the contract. All carried across turns.
- **In `Memory`:** the long-term memory (summary + items) that
  survives across sessions, persisted to `state/`.
- **In `RAGIndex`:** the indexed chunks; loaded from `docs/` (or
  wherever) at startup.
- **NOT here:** the LLM''s reply text (lives one turn), the model
  itself (in `llm.py` only), the audit log (in `audit/` as JSON).

## How to add a new tool

1. Add a `tools[]` entry in `contracts/agent-contract.json` with a
   `name`, `description`, and `response_schema`.
2. Optionally set `confirm: true` for irreversible actions.
3. Optionally set `soft_fail: true` if a malformed payload should
   return to the LLM (not abort).
4. Register the Python callable in the `tools=` arg of
   `ChatState(...)`.
5. The LLM discovers the tool through the system prompt (the
   `Tools available:` section added by `agent.py`).

The wire format is `<<<TOOL_CALL>>>{"name": "...", "args": {...}}`.
The runner extracts it, dispatches, validates, and feeds the
result back as `<<<TOOL_RESULTS>>>...<<<END>>>`.
