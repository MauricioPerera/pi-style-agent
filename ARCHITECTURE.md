# Architecture

This document explains the design decisions behind `pi-style-agent`.
The README tells you **what** the project is; this tells you **why** it
is that way and **what guarantees** you get.

The single most important idea in the project: **the agent loop is a
state machine that only ever calls the LLM at well-defined seams,
and every state transition is either deterministic (in the hard
layer) or model-driven (in the soft layer).** Knowing which is which
is the difference between an agent you can reason about and one you
cannot.

## The hard/soft split

```
   hard layer (deterministic)             soft layer (LLM-driven)
   ───────────────────────────             ───────────────────────
   no model calls                          calls the LLM
   pure stdlib (yaml, json, hashlib)      depends on the model
   unit-tested in milliseconds            tested live with the model
   fail-closed on bad input               best-effort on model output
   the contract                           the conversation
```

The hard layer is everything that *can* be done without a model: budget
allocation, priority truncation, secret detection, tool schema
validation, audit log writing, memory persistence, plan/scratchpad
parsing. None of it requires picking up a temperature. It is the part
of the system you can audit byte-by-byte.

The soft layer is everything that *requires* a model: emitting a
tool call, writing a memory delta, deciding what to say. These are
best-effort: the LLM might lie, omit, or confabulate. The system
treats every soft-layer output as untrusted input to the next hard-
layer check.

The two meet at exactly one place: the `assemble()` step. The runner
hands the soft layer (the LLM) a string, the LLM returns a string,
and the runner applies the next set of hard-layer checks before the
next call. There is no other crossing point.

## The agent loop, as a state machine

`runtime/soft/agent.py:run_turn` is the only place hard and soft code
touch. Every step is one of three kinds:

| Step | Kind | What it guarantees |
| --- | --- | --- |
| `assemble(contract, contents)` | hard | Slots fit the budget; critical slots are full; render ordering is correct. |
| Guardrail check on assembled slots | hard | No `sk-...` in the prompt. No malformed JSON in `tool_results`. Caller-side failures abort before any LLM call. |
| RAG query | soft | Top-k documents retrieved by similarity. The render is a string; the LLM decides what to do with it. |
| Confirmation gate | hard | If the tool is declared `confirm: true`, no dispatch. The runner returns `outcome="awaiting_confirm"` and the chat loop asks the human. |
| `llm_callable(system, user)` | soft | The model produces a reply. The reply is opaque until the next hard step parses it. |
| Tool-call extraction | hard | `<<<TOOL_CALL>>>{...}` is parsed by a regex + a JSON-load; the result is a dict or `None`. The LLM cannot inject anything that bypasses this parser. |
| Tool dispatch + schema validation | hard | The tool runs; its return is checked against the declared schema. Bad shape → abort (`tool_error`) or feed back as a rejected result (`soft_fail: true`). The LLM cannot smuggle a tool result past the validator. |
| Output sanitization | hard | The assistant''s reply is scanned for the same secret patterns as the input guardrail. Matches are replaced with `[REDACTED:<label>]`; the turn is never aborted on output. The LLM cannot exfiltrate a credential without it being redacted. |
| Memory delta apply | soft | The LLM emits `<<<MEMORY-DELTA>>>`. The runner parses it and applies the operations to the in-memory `Memory`. Keys are normalized to ASCII snake_case; collisions overwrite instead of duplicating. The LLM cannot store an un-normalized key. |
| Audit log write | hard | Per-turn JSON written to `audit/turn-<ms>.json`. The file contains hashes of every slot, the guardrail verdicts, the tool call log, the sanitization record, the memory delta, and a `payload_sha256` of the exact prompt. Reproducible byte-a-byte. |

Notice the pattern: every time the soft layer produces something, the
next hard step *validates* it before letting the model see the result
of its own action. The LLM can never be the last word on a state
transition.

## What we explicitly do not trust the LLM to do

This is the inverse of the previous section. Knowing what the LLM
*can''t* do is half the architecture.

- **Bound the number of tool calls.** The runner has a `tool_depth_cap`
  hard-side. A model in a loop calling the same tool forever will be
  cut off at N rounds.
- **Decide if a tool is reversible.** The contract declares
  `confirm: true` for irreversible tools. The runner never dispatches
  those without an explicit human signal. The model can ask; the
  human says yes or no.
- **Sanitize its own output.** Output sanitization runs *after* the
  model replies. The runner never trusts the model to redact itself.
- **Validate the shape of tool results.** Every tool declares a
  `response_schema`. The runner validates before the result is allowed
  into the next prompt. A tool that returns HTML when JSON is expected
  is caught here, not by the model on the next round.
- **Bump timestamps.** `DecayingRetriever.touch()` is called by the
  chat loop, not by the model. The LLM cannot decide "this memory
  item is fresh now"; only observed retrieval can.
- **Sign the audit log.** The audit log is written by the hard layer.
  The model cannot edit its own history.

## Slot priority and budget

Slots are declared in `contract/agent-contract.json` with a `priority`
integer. **Lower number = higher retention priority.** A budget
allocator (`runtime/hard/budget.py`) walks the slots in priority
order, gives each one its full size up to its `max_tokens`, and stops
when the budget runs out.

Critical slots (`compaction: none`) are special: they MUST enter the
prompt whole, or the turn aborts. The allocator never truncates a
critical slot. This is what guarantees that **the system prompt is
never dropped because of a long user message**.

The priority layout used by the demo:

```
priority 0  persona, hard_policies     ← never dropped
priority 1  long_term_mem              ← memory survives long histories
priority 2  plan, scratchpad           ← the agent''s working memory
priority 3  tool_results               ← dropped first under pressure
priority 4  history, user_input        ← dropped first
```

If you add a slot, pick a priority that matches its retention
semantics, not its size. A `user_input` slot should be priority 4
even if it''s small, because long histories are exactly when the
budget is tight.

## The seam: `assemble()`

`runtime/soft/assembler.py:assemble` is the function that turns
contract + slot contents into a single string the LLM consumes. It is
the **only** place in the project where:

1. The retriever is queried for memory (via `memory.render(query=...)`).
2. The RAG index is queried (via `rag_index.render_for_prompt(...)`).
3. The slot text is truncated by priority.
4. The slots are concatenated into the final payload.

Everything that needs to know "what does the model see" calls this
function. Tests that need to verify a guardrail works call it
directly, with a fake retriever and a fake RAG index, and assert on
the returned `AssembledTurn`. The model is never involved.

```python
turn = assemble(contract, contents, memory_retriever=retriever, memory_k=4)
assert turn.guardrails_passed
assert "sk-..." not in turn.payload
assert "<<<TOOL_RESULTS>>>" in turn.payload  # if a tool ran
```

## The retriever interface

A retriever is anything that implements:

```python
class Retriever(Protocol):
    def add(self, item: MemoryItem) -> None: ...
    def remove(self, key: str) -> None: ...
    def search(self, query: str, k: int = 5) -> list[tuple[MemoryItem, float]]: ...
```

`HashingRetriever` is a dependency-free stub for tests. `EmbeddingRetriever`
wraps an external embedder. `DecayingRetriever` wraps any retriever
and applies an exponential time-decay to the cosine score. `RAGIndex`
wraps any retriever and holds a parallel dict of chunk metadata for
provenance.

The same interface serves memory (small, frequently-touched) and
RAG (large, infrequent) — the chat loop does not care which is which,
it just calls `retriever.search(query, k)`.

## Multi-tenancy and policy

The system is single-tenant by construction. There is one `Memory`,
one `RAGIndex`, one chat loop. If you need multi-tenant:

- Per-tenant contracts → keep multiple `agent-contract.json` files,
  load by tenant ID.
- Per-tenant memory + RAG → use separate `Memory` and `RAGIndex`
  instances, key the chat state by tenant.
- Per-tenant audit log → point `state_dir` at per-tenant
  directories.

The hard/soft split is preserved across tenants because the
boundaries are about *what the model sees*, not *which user is
asking*.

## What is NOT covered (the honest list)

- **Jailbreak resistance.** CCDD guarantees the policies are *in* the
  prompt. The model can still be convinced to ignore them. Output
  sanitization catches the most common leak (a credential in the
  reply) but cannot prevent all misuse.
- **Tool execution security (partial).** Tool calls now run under
  `runtime/hard/sandbox.py`: a wall-clock timeout, crash containment,
  and an opt-in `isolated` mode that runs the tool in a separate,
  killable process. That bounds *liveness and blast radius* — a hung
  or crashing tool no longer takes down the turn. It does **not** bound
  *privilege*: an isolated tool still runs with the agent''s OS
  permissions. For untrusted code that shells out, you still need
  OS-level confinement (containers, seccomp, a restricted user). The
  hard layer validates a tool''s return shape and bounds its execution;
  it does not jail it.
- **RAG factuality.** The RAG retriever returns the top-k by
  similarity. The LLM may still misread them. If the corpus has
  stale or wrong docs, the agent will use them.
- **Memory consistency across forks.** A single Memory object is
  sequential; if you run two agents in parallel that share memory,
  they will race. Use external locking or split by tenant.
- **Encrypted at rest.** The persisted `state/` is plain JSON. Add
  encryption at the filesystem layer if you need it.

## Cross-references

- [README.md](README.md) — what it is, how to run it.
- [`runtime/soft/agent.py`](runtime/soft/agent.py) — the turn loop, with
  comments on each step.
- [`runtime/soft/assembler.py`](runtime/soft/assembler.py) — the seam.
- [`runtime/soft/memory.py`](runtime/soft/memory.py) — Memory, retrievers,
  decay.
- [`runtime/soft/rag.py`](runtime/soft/rag.py) — RAG chunking + retrieval.
- [`runtime/hard/budget.py`](runtime/hard/budget.py) — token estimation,
  priority truncation.
- [`runtime/hard/guardrails.py`](runtime/hard/guardrails.py) — input
  regex + json_schema.
- [`runtime/hard/output_sanitize.py`](runtime/hard/output_sanitize.py) —
  output regex redaction.
- [`runtime/hard/tools.py`](runtime/hard/tools.py) — tool schema
  validation, confirmable + soft_fail flags.
- [`docs/ARCHITECTURE.md`](../ccdd/.../ARCHITECTURE.md) (upstream) — the
  CCDD methodology this project was distilled from.
