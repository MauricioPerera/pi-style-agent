# Changelog

This project follows a soft-semver. The major version is bumped when
the contract or the hard-layer guarantees change in a way that
breaks older contracts. Minor bumps add features. Patches are
documentation and tests only.

## v0.5.1 — current

Patch: perimeter consistency. No change to the hard-layer guarantees or
the contract semantics; the sealed boundary and the turn loop are
untouched.

### Fixed
- **Offline demo ran broken.** `demo.py` asserted `tool_called == "search"`,
  but the turn loop records `tool_called` as a **list** (`["search"]`)
  since it went multi-step (the `tool_depth_cap` gate). The demo — a
  documented entry point — crashed on turn 2 and no test caught it.
  The audit log (the deterministic source of truth) was always correct;
  the demo was reading the old scalar format.

### Added
- **`tests/test_demo.py`** — subprocess smoke test that runs `demo.py`
  with the stub provider and asserts a clean exit. Binds the demo to the
  suite so a future audit-record format change fails loudly instead of
  silently rotting the showcase.

### Docs
- README test count synced (`154 → 166`).
- `agent-contract.json` `model` field corrected from a stale
  `claude-opus-4-8` to the `google/gemma-4-12b` the project actually runs.

### Tests
166 stdlib `unittest` tests (165 unit + 1 demo smoke).

## v0.5.0

### Hard layer
- **Budget:** tiktoken-based token estimation with `chars/4` fallback.
- **Guardrails:** `regex_deny` + `json_schema` (input side, fails closed).
- **Output sanitization:** deterministic redaction of secrets in
  the assistant''s reply.
- **Tools:** registry with schema validation, `confirm: true`
  gate for irreversible actions, `soft_fail: true` for graceful
  schema failures.
- **Sealed boundary:** the hard layer never imports from the
  soft layer. See [ARCHITECTURE.md](ARCHITECTURE.md).

### Soft layer
- **Turn loop:** assemble → guardrails → confirmation gate → LLM
  → tools (with the three gates) → sanitize → memory delta → audit.
- **Memory:** summary + items, ASCII-normalized keys, persistable
  to JSON. Three retrievers: `HashingRetriever` (tests),
  `EmbeddingRetriever` (real embedder, Matryoshka-aware),
  `DecayingRetriever` (wraps any retriever, time-decay).
- **RAG:** `RAGIndex` over any retriever, with chunking,
  provenance rendering, and chat-loop auto-injection.
- **Plan / scratchpad:** tags in the LLM''s reply, carried across
  turns by the chat loop.
- **Confirmation flow:** `outcome="awaiting_confirm"` with the
  pending call in `TurnResult.pending_confirm`; the chat loop
  surfaces `/confirm` and `/deny` commands.
- **Multilingual memory:** `embeddinggemma-300m-qat` is
  multilingual; cross-language retrieval is verified by a live test.
- **Matryoshka:** `embed_dim` constructor arg on
  `EmbeddingRetriever`; default 256 (1/3 of full 768, no measurable
  loss on small corpora).

### Surface
- **Chat loop** in `runtime/soft/chat.py` with `/quit`, `/reset`,
  `/memory`, `/tools`, `/audit`, `/config`, `/confirm`, `/deny`,
  `/help` commands.
- **Live demo** (`demo_live.py`) with 8 scripted turns; 2 of
  them (greeting, memory write) drive the long-term memory across
  sessions; 1 of them (turn 5b) drives a real RAG retrieval
  against `docs/`.

### Tests
165 stdlib `unittest` tests, 88s end-to-end (with the live server
tests). No LLM in the deterministic suite.

## v0.1.0 — first cut

- Slots with priority.
- Single LLM call per turn.
- `HashingRetriever` only (no real embedder yet).
- `regex_deny` input guardrail.
- Chat loop with `/quit` and `/memory`.

## How to read the project at v0.5

If you''re new, the order is:

1. [README.md](README.md) — what it is, how to run it.
2. [ARCHITECTURE.md](ARCHITECTURE.md) — why it is the way it is.
3. [runtime/soft/agent.py](runtime/soft/agent.py) — the turn loop.
4. [runtime/soft/assembler.py](runtime/soft/assembler.py) — the seam.
5. Anything else as needed.
