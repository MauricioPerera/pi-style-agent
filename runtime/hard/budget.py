"""Token estimation and priority-based truncation.

The hard layer never sees the LLM. It just decides how much of each slot fits
in the budget. Replacement-friendly: swap `estimate_tokens` for a real
tokeniser without touching anything else.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable


def estimate_tokens(text: str) -> int:
    """Estimate token count. Uses tiktoken''s cl100k_base if available (accurate
    for OpenAI / Claude / Llama-3 tokenisers); falls back to chars/4.

    The fallback is the same as the original approximation. tiktoken is a
    pure-Python-friendly package, ~1 MB, no required system deps; we lazy-
    import so the project still runs without it.
    """
    enc = _get_tokenizer()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    return max(0, (len(text) + 3) // 4)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to at most `max_tokens` tokens.

    With a real tokeniser this cuts cleanly at a token boundary (no
    half-tokens slipping into the prompt). With the chars/4 fallback
    it''s a 4x approximation, good enough for budgeting.
    """
    enc = _get_tokenizer()
    if enc is not None:
        ids = enc.encode(text, disallowed_special=())[:max_tokens]
        return enc.decode(ids)
    return text[: max_tokens * 4]


_TOKENIZER = None
_TOKENIZER_NAME = None


def _get_tokenizer():
    """Lazy-load a real tokeniser. Returns None if tiktoken is not installed.

    Callers must not mutate the return value.
    """
    global _TOKENIZER, _TOKENIZER_NAME
    if _TOKENIZER is not None or _TOKENIZER_NAME == "unavailable":
        return _TOKENIZER
    try:
        import tiktoken
        _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        _TOKENIZER_NAME = "cl100k_base"
    except ImportError:
        _TOKENIZER_NAME = "unavailable"
    return _TOKENIZER


def tokeniser_name() -> str:
    """For diagnostics: the name of the active tokeniser, or "chars/4"."""
    n = _get_tokenizer()
    return n.name if n is not None else "chars/4"


@dataclass
class SlotSpec:
    id: str
    priority: int                # lower = more important
    kind: str                    # "static" | "dynamic" | "runtime"
    compaction: str              # "none" | "truncate" | "summarize"
    min_tokens: int = 0
    max_tokens: int | None = None

    @property
    def is_critical(self) -> bool:
        return self.compaction == "none"


@dataclass
class AllocatedSlot:
    spec: SlotSpec
    text: str
    kept_tokens: int
    action: str                  # "full" | "truncate" | "drop"


@dataclass
class AllocationResult:
    slots: list[AllocatedSlot]
    used_tokens: int
    available_tokens: int
    aborted: str | None          # reason if we could not honour a critical slot


def allocate(slots: Iterable[SlotSpec], contents: dict[str, str], available: int) -> AllocationResult:
    """Allocate budget to slots in priority order.

    - Critical slots (compaction=none) MUST enter whole or we abort.
    - Non-critical slots are truncated or dropped, never raised in priority.
    - `min_tokens` is a floor: if a slot would fall below it, we abort
      (a 0-token persona is worse than refusing to answer).
    - A critical slot that declares a floor (`min_tokens > 0`) but receives
      EMPTY content aborts. Without this, `floor = min(want, min_tokens)`
      collapses to 0 when `want == 0`, so an empty persona/hard_policies slot
      would slip through silently — exactly the "0-token persona" case above.
      Short-but-present content still passes (the floor stays lenient for it).
    """
    remaining = available
    out: list[AllocatedSlot] = []

    # Strict priority order: ties resolved by declaration order (stable).
    ordered = sorted(slots, key=lambda s: s.priority)

    for spec in ordered:
        text = contents.get(spec.id, "")
        want = estimate_tokens(text)
        cap = spec.max_tokens if spec.max_tokens is not None else want
        grant = min(want, cap, remaining)

        if spec.is_critical:
            if want > remaining:
                return AllocationResult(
                    slots=out, used_tokens=available - remaining,
                    available_tokens=available,
                    aborted=f"slot critico '{spec.id}' no entra ({want} tok pedidos, {remaining} disponibles)",
                )
            if spec.min_tokens > 0 and want == 0:
                return AllocationResult(
                    slots=out, used_tokens=available - remaining,
                    available_tokens=available,
                    aborted=f"slot critico '{spec.id}' vacio (requiere min_tokens={spec.min_tokens})",
                )
            grant = want
            action = "full"
        else:
            if grant < want:
                kept_text = text if grant == want else truncate_to_tokens(text, grant)
                action = "truncate" if grant > 0 else "drop"
                text = kept_text
            else:
                action = "full"

        floor = min(want, spec.min_tokens)
        if grant < floor:
            return AllocationResult(
                slots=out, used_tokens=available - remaining,
                available_tokens=available,
                aborted=f"slot '{spec.id}' truncado bajo su piso ({grant} < {floor} tok)",
            )

        used = estimate_tokens(text) if action != "drop" else 0
        out.append(AllocatedSlot(spec=spec, text=text, kept_tokens=used, action=action))
        remaining -= used

    return AllocationResult(
        slots=out,
        used_tokens=available - remaining,
        available_tokens=available,
        aborted=None,
    )
