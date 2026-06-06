# runtime/

The runtime layer. Two subdirectories split by what each does
*without* the LLM:

- [hard/](hard/) — deterministic, no model calls. The things you can
  audit byte-by-byte.
- [soft/](soft/) — the LLM-facing layer. The chat loop, the turn
  loop, the retriever, the RAG index.

The two meet at one place: `assemble()` in
[soft/assembler.py](soft/assembler.py). See
[ARCHITECTURE.md](../ARCHITECTURE.md) for the full picture.

## When to add a file here

- **In `hard/`** if the new code can be unit-tested without an LLM
  call. Examples: a new regex guardrail, a new compaction policy, a
  different token-budget strategy.
- **In `soft/`** if the new code either calls the LLM or interprets
  its output. Examples: a new wire format, a new tool-call convention,
  a new retriever.

## Cross-references

- [ARCHITECTURE.md](../ARCHITECTURE.md)
- [README.md](../README.md)
