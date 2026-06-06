# runtime/hard/

The hard layer. Deterministic, no LLM calls. All the code that
*decides* what the agent can and cannot do.

## Modules

- [budget.py](budget.py) — token estimation, priority-based truncation.
  The allocator walks slots in priority order; critical slots
  (`compaction: none`) enter the prompt whole or the turn aborts.
  Uses `tiktoken`''s `cl100k_base` if installed, falls back to
  `chars/4`.
- [guardrails.py](guardrails.py) — pre-inference `regex_deny` and
  `json_schema` checks. Fails closed. The `tool-output-shape` guard
  validates the contents of the `tool_results` slot.
- [output_sanitize.py](output_sanitize.py) — deterministic redaction
  of secrets / PII in the assistant''s reply. Mirrors the input
  `regex_deny` but redacts-and-continues instead of aborting.
- [secrets.py](secrets.py) — the regex patterns for known secret
  formats (`sk-…`, AWS, PEM, GitHub PAT, Slack). Centralised so the
  input guardrail and the output sanitizer stay in sync.
- [schema.py](schema.py) — the single recursive JSON-Schema-ish
  validator (`validate`). Both `tools.validate_response` and the
  `json_schema` guardrail use it, so a tool response and a slot are
  checked against the same logic. Centralised so the two can never
  drift apart.
- [tools.py](tools.py) — tool registry. Each tool declares a
  `response_schema`; the runner validates the return. Supports
  `confirm: true` (irreversible actions need `/confirm`) and
  `soft_fail: true` (schema violations return to the LLM as
  rejected tool results).

## How the layers meet

The soft layer (chat loop, turn loop) calls into the hard layer
*only* through these public functions:

| Function | Module | What it guarantees |
| --- | --- | --- |
| `estimate_tokens`, `truncate_to_tokens` | `budget.py` | Cheap approximation; swap to a real tokeniser without touching the call sites. |
| `run_guardrails(contract, slots)` | `guardrails.py` | Fails closed. The only place secrets are filtered on the input side. |
| `sanitize(text)` | `output_sanitize.py` | Redacts. Never aborts. |
| `specs_from_contract`, `validate_response`, `is_confirmable`, `tool_spec_from_contract` | `tools.py` | Schema validation is hard; routing decisions (which tool, which mode) are soft. |

The hard layer never calls into the soft layer. If you find yourself
adding an import from `runtime/soft/` inside `runtime/hard/`, stop —
you''re crossing the seam in the wrong direction.
