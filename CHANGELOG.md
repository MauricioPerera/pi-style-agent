# Changelog

This project follows a soft-semver. The major version is bumped when
the contract or the hard-layer guarantees change in a way that
breaks older contracts. Minor bumps add features. Patches are
documentation and tests only.

## v0.5.6 — current

Adds an adversarial security regression corpus (Nivel 3). Tests only; no
code change.

### Added
- **`tests/test_adversarial.py`** (11 tests). Three kinds:
  - *Guarantees:* every known secret format is blocked on input (any slot)
    and redacted on output, end-to-end through the turn loop (a secret on
    input blocks before any LLM call; a secret in the reply is redacted and
    logged, not aborted).
  - *Documented gaps:* explicit assertions that the deterministic layer does
    NOT catch unrecognized secret formats (Google key, JWT, AWS secret key),
    trivial obfuscation (a space inside the token), or prompt injection. If a
    gap is later closed, the test fails and gets updated — the limitation
    stays tracked instead of silent.
  - *No false positives:* near-miss strings below each pattern's minimum
    length do not trip the guardrail.
- ARCHITECTURE.md's honest list now points at this corpus.

### Tests
219 stdlib `unittest` tests.

## v0.5.5

Adds state integrity under crashes and concurrency (Nivel 2, third piece).
Completes Nivel 2.

### Added
- **`runtime/hard/statelock.py`** — `write_atomic` (temp file + `os.replace`,
  so a crash mid-write never leaves a torn JSON file) and `state_lock` (an
  advisory `O_EXCL` file lock that serializes concurrent writers and breaks
  stale locks). Pure stdlib, cross-platform.
- `Memory.save` / `save_index` now write atomically; `ChatState.persist`
  takes the lock around the whole read-modify-write.
- `tests/test_statelock.py` (8 tests: atomic write, mutual exclusion,
  re-acquire, stale-lock breaking).

### Honest scope
This prevents *corruption* (torn files, interleaved writes). It does NOT
prevent *lost updates* between two long-lived agents that each loaded the
state at startup — the later writer wins. Real multi-writer correctness
needs per-tenant state. ARCHITECTURE.md's honest list says so.

### Tests
208 stdlib `unittest` tests.

## v0.5.4

Adds encryption at rest for persisted state (Nivel 2, second piece).
Opt-in; default behavior (plaintext) is unchanged.

### Added
- **`runtime/hard/crypto.py`** — `encrypt_str` / `decrypt_str` /
  `is_encrypted`. scrypt KDF (stdlib `hashlib.scrypt`) derives a key from a
  passphrase; Fernet (authenticated AES-128-CBC + HMAC, via `cryptography`)
  encrypts. Self-describing JSON envelope carries the salt + KDF params.
  Tamper-evident: a modified or truncated blob fails closed on decrypt.
- **`PI_STATE_PASSPHRASE`** — when set, `ChatState.persist` and the demo
  loaders encrypt `memory.json` / `index.json`. `Memory.save/load` and
  `save_index/load_index` gained an optional `passphrase` parameter.
- `tests/test_crypto.py` (11 tests: primitive round-trip / wrong key /
  tamper / no-leak, and encrypted persistence + plaintext migration).

### Notes
- **No home-grown crypto, no silent plaintext.** `cryptography` is an
  optional lazy import (like tiktoken), but if you ask to encrypt and it is
  missing, it raises — it never quietly persists plaintext.
- **Backward compatible.** Existing plaintext state still loads even with a
  passphrase set; the next save re-writes it encrypted.
- **Out of scope:** key management. The passphrase lives in your env; lose
  it and the state is unrecoverable by design.

### Tests
200 stdlib `unittest` tests.

## v0.5.3

Adds deterministic execution bounds for tool calls (the first piece of
"Nivel 2"). Hard-layer only; the sealed boundary is intact.

### Added
- **`runtime/hard/sandbox.py` — `run_guarded(fn, args, *, timeout_s,
  isolated)`.** A tool can hang or crash; the turn cannot. Default mode
  runs the tool in a worker thread and bounds the *wait* (a hung thread
  lingers but the turn regains control). Opt-in `isolated=True` runs it
  in a separate `spawn` process that can be killed and that contains a
  segfault. Timeouts and crashes surface as `ToolTimeout` / `ToolCrashed`.
- **Contract keys `timeout_s` and `isolated` per tool**, read by
  `tools.tool_exec_opts` and applied at both tool-dispatch sites in the
  turn loop.
- `tests/test_sandbox.py` (11 tests: threaded + isolated success / crash /
  timeout, opts parsing).

### Honest scope
This bounds *liveness and blast radius*, not *privilege*. An isolated tool
still runs with the agent's OS permissions; it is not a security sandbox.
ARCHITECTURE.md's honest list is updated to say so.

### Tests
189 stdlib `unittest` tests.

## v0.5.2

Patch: hard-layer cleanup. No change to the contract or the sealed
boundary; both fixes make existing behavior more correct, not different
in the happy path.

### Changed
- **Single schema validator.** Extracted `runtime/hard/schema.py`. The
  tool-response check and the `json_schema` guardrail had two
  near-identical `_validate` copies — one recursive, one flat — that
  could disagree on the same schema. They now share one recursive
  implementation, so the guardrail also catches wrong types on *nested*
  properties (previously only top-level shape was checked).

### Fixed
- **Tool-call parser.** `_extract_tool_call` used `rfind("}")`, grabbing
  the last brace in the whole reply. Trailing prose containing a `}`
  made the JSON slice unparseable, and adjacent objects could merge.
  Replaced with a balanced-brace scan that respects JSON string literals
  and escapes. Correct for nested args and trailing text.

### Added
- `tests/test_tool_call_parse.py` (parser + brace matcher) and a nested-
  property case in `test_hard.py`.

### Tests
178 stdlib `unittest` tests.

## v0.5.1

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
