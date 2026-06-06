# pi-style-agent

A personal/async agent skeleton that borrows the **durable ideas** from
[CCDD](../ccdd) — slots with explicit priority, deterministic guardrails, the
hard/soft split, and bit-a-bit auditability — without adopting its ceremony
(versioned `context.yaml`, Ed25519 attestations, CI gate R1–R9, quorum). Those
are designed for **teams shipping agents through PRs**; this skeleton is
designed for **a single long-running agent whose context mutates every turn**.

Runs end-to-end against [LM Studio](https://lmstudio.ai) with `gemma-4-12b`
for chat and `embeddinggemma-300m-qat` for memory retrieval. The default model
choices reflect what the project was built and tested against; both are
configurable via env vars.

## Documentation

- [README.md](README.md) — what the project is, how to run it (you are here).
- [ARCHITECTURE.md](ARCHITECTURE.md) — the hard/soft split, the seam,
  what the LLM is and isn''t trusted to do. **Read this second.**
- [CHANGELOG.md](CHANGELOG.md) — what changed in each version and the
  rationale.
- [CONTRIBUTING.md](CONTRIBUTING.md) — how to add code, where each
  piece of the system lives, what NOT to do.
- [runtime/README.md](runtime/README.md) — top-level map of the
  runtime layer.
- [runtime/hard/README.md](runtime/hard/README.md) — the hard layer
  (deterministic, no LLM).
- [runtime/soft/README.md](runtime/soft/README.md) — the soft layer
  (LLM-facing, chat loop, retrievers, RAG).
- [docs/README.md](docs/README.md) — the RAG knowledge base format.

## What we kept from CCDD

- **Slots with priority** (`runtime/hard/budget.py`): persona, hard policies,
  long-term memory, plan, scratchpad, tool results, history, user input —
  truncated in that order when the budget runs out.
- **Deterministic pre-inference guardrails** (`runtime/hard/guardrails.py`):
  `regex_deny` (catches `sk-…`, AWS keys, PEM private keys, GitHub PATs, Slack
  tokens) and `json_schema` (validates a slot''s contents). Fails closed.
- **Tool registry with schema-validated responses** (`runtime/hard/tools.py`):
  every tool declares its response shape; the runner validates before the
  result is allowed into the prompt. Bad shape → `tool_error`, no LLM call.
- **Output sanitization** (`runtime/hard/output_sanitize.py`): mirror of the
  input regex_deny, but on the assistant''s reply. Redacts leaked secrets
  with `[REDACTED:<label>]` instead of aborting the turn — a redacted reply
  is still useful, and the audit log records what was redacted.
- **Multi-step tool loop** with a hard depth cap. The LLM can chain
  tool calls; the hard layer stops it at `tool_depth_cap` rounds.
- **Soft-fail tool results** (v0.4): a tool declared `soft_fail: true`
  returns a malformed payload to the LLM as a rejected tool result
  (instead of aborting the turn), so the model can adapt: retry with
  different args, ask the user, or fall back to a different tool.
- **Confirmation gate for irreversible actions**: a tool declared
  `confirm: true` in the contract is never dispatched without the
  human''s explicit `/confirm`. The runner returns `outcome="awaiting_confirm"`
  and the chat loop surfaces the question. The next turn dispatches or
  rejects.
- **Bit-a-bit audit log** (`audit/turn-*.json`): per turn, the hashes of every
  slot, the assembled payload hash, the guardrail verdict, the tool call
  log, the sanitization record, the model''s reply preview, and the memory
  delta. Reproducible byte-a-byte.

## What we deliberately did NOT take

- No versioned `context.yaml` + `lint --sign` (context mutates every turn).
- No Ed25519 attestations / quorum (no reviewer registry makes sense for one
  user).
- No CI gate R1–R9 (most context changes are made by the agent, not a human).
- No advisory LLM (`review_assist.py`) — the LLM is already in the loop.

## Layout

```
pi-style-agent/
  contracts/
    agent-contract.json     # persona, slots with priority, tools, guardrails
  runtime/
    hard/                   # deterministic, no LLM (the seam)
      budget.py             #   token estimation, truncation by priority
      guardrails.py         #   regex_deny, json_schema (fails closed)
      output_sanitize.py    #   secret redaction on the LLM reply
      secrets.py            #   secret patterns
      tools.py              #   tool registry + schema validation
    soft/                   # calls the LLM
      assembler.py          #   the hard/soft seam (deterministic, testable)
      llm.py                #   thin wrapper (stub / OpenAI / LM Studio)
      agent.py              #   the turn loop (tools, memory, confirm, soft_fail)
      plan.py               #   <<<PLAN>>>/<<<SCRATCHPAD>>> tag parsing
      memory.py             #   two-level memory + delta parser + retrievers
      embeddings.py         #   LM Studio embeddings client + retriever factory
      lms.py                #   lms CLI wrapper (preload model + warmup)
      rag.py                #   RAG: chunk documents, retrieve top-k
      chat.py               #   interactive chat loop with /commands
  audit/                    # one JSON per turn (also audit_demo* from runs)
  state_demo/               # persisted memory.json + index.json
  config/
  tests/                    # 227 stdlib unittest tests, 0 LLM calls in the loop
    test_hard.py            #   budget, secrets, guardrails, output sanitize
    test_tools.py           #   tool schema validation
    test_memory.py          #   memory, plan/scratch, retrievers, Matryoshka dim
    test_embeddings.py      #   embedder/retriever (live tests gated on server)
    test_output_sanitize.py #   redaction in isolation + in the agent loop
    test_assembler.py       #   assembler + agent loop end-to-end (faked LLM)
    test_lms.py             #   lms CLI wrapper + multilingual live tests
    test_chat.py            #   chat loop, command handling, confirmation flow
    test_rag.py             #   RAG: chunking, retrieval, rendering for prompt
  demo.py                   # scripted end-to-end demo (no LLM)
  demo_live.py              # live demo against LM Studio
```

## Memory: two levels + retrieval

The agent has a structured long-term memory that survives across sessions:

- **Summary** (1 paragraph about the user) — high signal, low churn.
- **Items** (list of `{key, value}` preferences) — high churn, structured.

The LLM emits updates as a `<<<MEMORY-DELTA>>>` tag at the end of its reply;
the runner parses and applies them. Keys are normalized to ASCII
snake_case by the hard layer, so a model that writes `Ubicación`,
`ubicacion`, or `Ubicación  ` on different turns collides on the same
canonical key.

For retrieval, the `long_term_mem` slot is rendered through an
`EmbeddingRetriever`. The default is `embeddinggemma-300m-qat` served by
LM Studio, truncated to **Matryoshka dim 256** (1/3 the storage of full 768,
no measurable quality loss on small corpora). Override with
`PI_EMBED_DIM=128` or `768`.

**Multilingual:** embeddinggemma is multilingual out of the box. A
Spanish query `"donde vive Maria"` matches an English doc `"Maria lives
in Madrid"` with cosine 0.75.

**Decay (optional):** wrap the retriever in `DecayingRetriever`
(see [`runtime/soft/memory.py`](runtime/soft/memory.py)) to demote memory
items that haven\'t been accessed in a while. Each `MemoryItem` carries a
`last_accessed` epoch-ms; the wrapper multiplies the cosine score by
`exp(-ln(2) * age / half_life)`. Items never touched (or just touched)
get full weight. The chat loop calls `touch(keys)` after every search
to keep recently-used items fresh. Default half_life is 7 days.

## RAG: retrieval over a knowledge base

`runtime/soft/rag.py` adds document ingestion and chunk-level retrieval
on top of the same retriever interface:

```python
from pathlib import Path
from runtime.soft.embeddings import build_lmstudio_retriever
from runtime.soft.rag import RAGIndex, load_directory

idx = build_lmstudio_retriever()
rag = RAGIndex(idx)
rag.add_documents(load_directory(Path("docs/")))   # default: **/*.md
rendered = rag.render_for_prompt("donde vive Maria", k=5)
# "[source: spain.md, score=0.832]\nMadrid es la capital...\n\n---\n\n..."
```

The chat loop drops `rendered` into the `tool_results` slot; the LLM
reads the chunks as if they came from a search tool and cites them by
name. `RAGIndex` holds a parallel dict of chunk metadata so
`render_for_prompt` includes `<source>: <doc_name>` headers for
attribution.

## Tools: schema-validated, multi-step, confirmable, soft-fail

Each tool is declared in the contract with:

```json
{
  "name": "delete_user",
  "description": "Delete a user account. IRREVERSIBLE.",
  "confirm": true,                  // or false (default)
  "soft_fail": true,                // or false (default)
  "response_schema": { ... }
}
```

The runner:

1. Parses the LLM''s `<<<TOOL_CALL>>>{"name":..., "args":{...}}`.
2. If the tool is confirmable, **does not dispatch** — returns
   `outcome="awaiting_confirm"` with the pending call. The chat loop
   shows `[awaiting confirm] delete_user(...)` and the user types
   `/confirm` or `/deny`. Next turn dispatches or rejects.
3. Calls the tool, validates the response against the schema.
4. If the schema fails AND `soft_fail: true`, feeds the rejection back
   to the LLM as a tool_results block (with `error` + `rejected` fields)
   so the model can retry with different args. If `soft_fail: false`
   (default), aborts the turn with `tool_error`.
5. If valid, feeds the result back as `<<<TOOL_RESULTS>>>` and lets the
   LLM continue. Loop repeats up to `tool_depth_cap` rounds (default 2,
   hard-side).

This gives you the chainable tool use of ReAct/Voyager without
dragging in a framework, plus the safety properties of CCDD (human
in the loop for irreversible actions, no LLM trust on its own output).

## Hard/soft split (the seam)

The only place hard and soft code meet is in the agent loop:

```
1. assemble(contract, contents)              ← hard: budget, priorities, guardrails
2. if !guardrails_passed: return blocked    ← hard: no LLM call
3. render long_term_mem through retriever    ← soft: similarity search
4. if pending_confirm: dispatch or reject   ← hard: gate
5. llm_callable(system, user)                ← soft: first round
6. parse tool calls (if any)                 ← hard: regex, JSON parse
7. dispatch tool, validate shape             ← hard: schema check
8. if schema failed and soft_fail: feed      ← hard: feed back, don''t abort
   rejection to LLM
9. llm_callable(followup)                   ← soft: next round
10. sanitize(final reply)                     ← hard: redact secrets, don''t abort
11. apply_delta(memory, delta)               ← soft: model decides what to remember
12. write audit/turn-<ts>.json               ← hard: append-only log
```

Step 1 is fully unit-testable without an LLM. Steps 6/7/8/10/12 are
deterministic. The agent loop never trusts the LLM to bound itself,
sanitize its own output, or remember correctly.

## Interactive chat loop

`runtime/soft/chat.py` is the canonical entry point:

```python
from pathlib import Path
from runtime.soft.assembler import load_contract
from runtime.soft.chat import ChatState, run_forever
from runtime.soft.embeddings import build_lmstudio_retriever
from runtime.soft.lms import warmup_embeddings
from runtime.soft.memory import Memory
from runtime.soft.rag import RAGIndex, load_directory

warmup_embeddings()
contract = load_contract(Path("contracts/agent-contract.json"))
idx = build_lmstudio_retriever()
rag = RAGIndex(idx)
rag.add_documents(load_directory(Path("docs/")))

state = ChatState(contract=contract, memory=Memory(), retriever=idx, tools={}, state_dir=Path("state/"))
run_forever(state)
```

Commands (start with `/`):
- `/quit` exit
- `/reset` clear memory + index
- `/memory` show long-term memory
- `/tools` list registered tools + schemas
- `/audit` path of the latest audit log entry
- `/config` show contract name, model, embedder
- `/confirm` / `/deny` confirm or reject a pending tool call
- `/help` this list

Plan and scratchpad carry over across turns automatically (the
runner extracts them from the model''s reply and the chat loop feeds
them back as slot contents in the next turn).

## Persistence

After every turn, the demo persists `Memory` (JSON) and the
retriever''s items (JSON, vectors re-computed on load) to
`state_dir/`. The next run loads them and the agent starts with
prior context. Re-run the demo to see this in action.

## Run

```bash
# Tests (no LLM, ~20s for the deterministic suite; ~85s with live
# server tests; 227 tests total)
python -m unittest discover -s tests -p "test_*.py" -v

# Offline demo (deterministic, no LM Studio needed)
python demo.py

# Live demo against LM Studio
$env:PI_LLM_PROVIDER = "lmstudio"           # PowerShell; use export on bash
$env:PI_LLM_MODEL    = "google/gemma-4-12b"
$env:PYTHONIOENCODING = "utf-8"
python demo_live.py

# Interactive chat
python -c "from runtime.soft.chat import run_forever, ChatState; ..."
```

`demo_live.py` runs 7 scripted turns: greeting, intro, tool call, tool
chain, retrieval test, secret injection (blocked), output sanitization
test. Each turn is logged to `audit_live/turn-<ts>.json`. The whole run
takes ~10–15 min on a 12B model.

### Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `PI_LLM_PROVIDER` | `stub` | `stub` (deterministic), `openai` (official client), `lmstudio` (urllib) |
| `PI_LLM_MODEL` | `google/gemma-4-12b` | LLM model identifier |
| `PI_LLM_ENDPOINT` | `http://localhost:1234/v1/chat/completions` | LM Studio chat URL |
| `PI_EMBED_MODEL` | `text-embedding-embeddinggemma-300m-qat` | Embedding model (Q4_0; switch to no-suffix for Q8_0) |
| `PI_EMBED_DIM` | `256` | Matryoshka truncation dim (128/256/768) |
| `PI_LLM_TIMEOUT` | `300` | HTTP timeout in seconds |
| `LMS_CLI` | (auto) | Path to the `lms` binary (for model preloading) |
| `PI_STATE_PASSPHRASE` | (unset) | If set, persisted `state/` (memory + index) is encrypted at rest (scrypt + Fernet). Requires `cryptography`. Unset = plaintext. |

## Why embeddinggemma + Matryoshka

- **Small.** Q4_0-QAT is 219 MB. Fits alongside a 12B chat model in
  modest VRAM. Cold load ~2s, warm inference sub-100ms.
- **Multilingual.** Same semantic space across 100+ languages. A user
  who types in Spanish and a doc base in English work without
  translation.
- **Matryoshka.** The vector is trained so the first 256 dimensions
  carry nearly all the signal. We default to 256 to save 2/3 of
  storage and cosine cost; quality is identical to 768 on small
  corpora. Bump to 128 if you need hot-path speed; bump to 768 if
  your memory grows past 1000s of items.
- **Q4_0-QAT vs Q8_0.** The two are nearly identical on retrieval
  (cosine ~0.97 per text, identical top-1 ranking). Q4_0-QAT is 30%
  smaller. Default to Q4_0; switch to the no-suffix model id if you
  want the reference.

## Next steps (when you need them)

- **Per-tool `on_fail: reroute`** (CDD-style): pipe the bad payload to
  a recovery slot the LLM can use to reformulate. The `soft_fail`
  flag is the lighter-weight version of this.
- **Webhook of audit events**: the audit log is a directory of JSONs;
  a `tail` mode that follows the latest file and emits events (to
  Slack, to a logger) is ~50 lines.
- **Plan editor / visualizer**: dado que plan y scratchpad son strings con estructura
- **Multi-step tool chains with depth tracking**: the runner already
  supports depth; adding per-step budget tracking and a separate
  scratchpad for the chain is straightforward.
