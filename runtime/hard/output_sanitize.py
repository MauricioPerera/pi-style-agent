"""Output sanitization: deterministic redaction of secrets / PII in the
LLM''s final reply.

Symmetric to the input `regex_deny` guardrail, but with one crucial difference:
we do NOT abort the turn. A redacted reply is still useful to the user
("I think you may have leaked a credential; I''ve removed it from my reply
so it does not propagate to logs") and cheaper than re-running inference.

Why this is a HARD layer concern: the patterns are well-known (sk-, AKIA,
PEM headers, etc.), they are stable across models, and false positives are
cheap (we just substitute the match with a `[REDACTED:secret]` token). The
LLM is never trusted to spot its own leaks.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Iterable

from .secrets import DEFAULT_PATTERNS, compile_patterns


# Mapping from a regex group to the label that will appear in the redaction
# marker. Order matters: more specific patterns first, so a PEM block is not
# mis-reported as a generic secret.
_LABEL_BY_PATTERN: dict[str, str] = {
    r"sk-[A-Za-z0-9]{20,}": "openai_key",
    r"AKIA[0-9A-Z]{16}": "aws_key",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----": "private_key",
    r"ghp_[A-Za-z0-9]{30,}": "github_pat",
    r"xox[baprs]-[A-Za-z0-9-]{10,}": "slack_token",
}


@dataclass
class SanitizationResult:
    text: str                  # the redacted text (always safe to display)
    redacted: list[dict]       # [{"label": "...", "match": "...[truncated]"}]
    clean: bool                # True iff no redactions happened

    def summary(self) -> str:
        if self.clean:
            return "ok"
        labels = sorted({r["label"] for r in self.redacted})
        return f"redacted: {', '.join(labels)} ({len(self.redacted)} match(s))"


def sanitize(text: str,
             extra_patterns: Iterable[str] = ()) -> SanitizationResult:
    """Apply the default secret patterns + any extras to `text` and return
    the redacted version. Each match is replaced by `[REDACTED:<label>]`.
    """
    patterns = list(DEFAULT_PATTERNS) + list(extra_patterns)
    compiled = compile_patterns(patterns)
    redacted: list[dict] = []
    out = text

    # Apply patterns in declared order, longest first within a class, so a
    # more specific match wins (e.g. PEM header before generic `sk-`).
    for raw, compiled_re in zip(patterns, compiled):
        label = _LABEL_BY_PATTERN.get(raw, "secret")
        def _sub(m, _lbl=label):
            s = m.group(0)
            redacted.append({"label": _lbl,
                             "match": s[:8] + "..." if len(s) > 12 else s})
            return f"[REDACTED:{_lbl}]"
        out = compiled_re.sub(_sub, out)

    return SanitizationResult(text=out, redacted=redacted, clean=not redacted)
