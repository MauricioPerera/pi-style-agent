# Contributing

The project is small (14 modules, 165 tests) and opinionated. Most
contributions are either:

1. A new piece in the hard layer (a new guardrail, a new budget
   strategy, a new tool validation).
2. A new piece in the soft layer that talks to a new LLM feature
   (streaming, function-calling, multimodal input).
3. A new corpus in `docs/` or a new demo scenario.

## Before you write a line

Read [ARCHITECTURE.md](ARCHITECTURE.md) first. It is short and
explains the rules. The most important rule:

> Every time the soft layer produces something, the next hard step
> validates it.

If your change crosses the seam in the other direction (hard layer
imports from soft), it''s wrong. If your change has the LLM decide
something that should be hard-decided, it''s wrong.

## Where to add code

- **New guardrail (input side)** → `runtime/hard/guardrails.py`. The
  pattern is a function that takes a string and returns a verdict.
  The contract''s `guardrails[]` array is where it gets wired in.
- **New output filter** → `runtime/hard/output_sanitize.py`. The
  pattern is a regex + a label; the default patterns are in
  `runtime/hard/secrets.py`.
- **New retriever** → `runtime/soft/memory.py`. Implement the
  `Retriever` protocol (or subclass `EmbeddingRetriever` if you have
  an external embedder). It must compose with `DecayingRetriever`.
- **New tool validation rule** → `runtime/hard/tools.py` /
  `runtime/soft/agent.py`. The dispatcher in `agent.py` calls
  `validate_response(spec, payload)`; if your rule is more than a
  JSON-Schema check, add it there.
- **New RAG chunker** → `runtime/soft/rag.py`. The default is
  `chunk_document(doc, max_tokens=200, overlap_tokens=20)`. Override
  via a subclass of `RAGIndex`.
- **New wire format with the LLM** → `runtime/soft/agent.py` and
  `runtime/soft/plan.py`. The current formats are `<<<PLAN>>>`,
  `<<<SCRATCHPAD>>>`, `<<<TOOL_CALL>>>`, `<<<TOOL_RESULTS>>>`,
  `<<<MEMORY-DELTA>>>`. New tags should follow the same shape: a
  single line opener, a body, `<<<END>>>`.

## Tests are required

Every change ships with tests. The pattern:

- **Hard layer change:** a `unittest.TestCase` in the matching
  `tests/test_*.py`. No LLM, no network. The test should be
  deterministic and finish in <100ms.
- **Soft layer change:** a `unittest.TestCase` with a `ScriptedLLM`
  (or a fake `llm_callable`) that returns canned replies. The test
  exercises the agent loop without a real model.
- **Real-model integration (optional):** a test class gated on
  `LM Studio no esta corriendo` via `unittest.skipUnless(...)`.
  These run only when the server is reachable; they are not
  required for CI to pass.

The bar is: every test in `tests/` must pass on a clean checkout
without any external service, and the live tests must pass when the
service is available. Currently 165 tests, ~88s with the live suite
included.

## Style

- Stdlib only by default. `tiktoken` is an optional dependency
  (the import is inside a function); if you add a hard dependency,
  explain why in the module docstring.
- Docstrings on every public function. The seam modules
  (`agent.py`, `assembler.py`) are heavily commented; new
  soft-layer files should match.
- Prefer composition over modification. `DecayingRetriever` wraps
  any retriever instead of adding a `decay` flag to
  `EmbeddingRetriever`. The same pattern applies to the RAG index
  (it wraps any retriever) and the chat loop (it accepts an
  optional `rag_index`).

## Running the suite

```bash
# All tests (88s with live server; ~20s offline).
python -m unittest discover -s tests -p "test_*.py" -v

# Just the hard layer (sub-second).
python -m unittest tests.test_hard tests.test_tools -v

# Live demo against LM Studio.
$env:PI_LLM_PROVIDER = "lmstudio"
$env:PI_LLM_MODEL    = "google/gemma-4-12b"
python demo_live.py
```

## What NOT to do

- **Don''t** add a `class AgentConfig: …` mega-config. The contract
  is JSON; the chat state is dataclasses; the retriever is a
  protocol. Adding a config class is a smell.
- **Don''t** make the LLM responsible for sanitization, validation,
  or any state transition. Every transition is hard.
- **Don''t** add a feature that the model has to remember to use
  (e.g. a magic prefix in the user message). If the LLM can
  forget it, the feature is unreliable.
