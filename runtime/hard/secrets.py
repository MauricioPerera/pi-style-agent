"""Secret patterns used by the no-secrets guardrail.

Centralised so the patterns are easy to extend and easy to test.
"""
from __future__ import annotations
import re
from typing import Iterable

# Default patterns, copy-paste-friendly from CCDD with a couple of additions.
DEFAULT_PATTERNS: tuple[str, ...] = (
    r"sk-[A-Za-z0-9]{20,}",            # OpenAI / many SaaS API keys
    r"AKIA[0-9A-Z]{16}",               # AWS access key ID
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",  # PEM private key
    r"ghp_[A-Za-z0-9]{30,}",           # GitHub personal access token
    r"xox[baprs]-[A-Za-z0-9-]{10,}",   # Slack token
)


def compile_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in patterns]


def find_secrets(blob: str, compiled: list[re.Pattern[str]]) -> list[str]:
    """Return the list of secret matches in `blob`. Empty list means clean."""
    hits: list[str] = []
    for pat in compiled:
        hits.extend(m.group(0) for m in pat.finditer(blob))
    return hits
