"""Aggregate the per-turn audit log into observability. Hard layer.

The agent writes one `audit/turn-*.json` per turn — the deterministic,
append-only record of what happened (allocation, guardrail verdicts, tool
calls, sanitization, memory delta). This module reads that pile back and
summarizes it, so the security machinery we can't see at a glance — how often
a guardrail blocked, what got redacted, which tools erred or timed out —
becomes a number you can watch.

Pure stdlib, deterministic, model-independent: it only reads JSON the loop
already wrote. Run as a script:

    python -m runtime.hard.audit_report [audit_dir]
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _load_records(audit_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in sorted(Path(audit_dir).glob("turn-*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue  # a torn or unreadable file should not break the report
    return out


def summarize_audit(audit_dir: Path) -> dict[str, Any]:
    """Aggregate every `turn-*.json` in `audit_dir` into a stats dict."""
    records = _load_records(audit_dir)

    outcomes: Counter[str] = Counter()
    guardrail_blocks = 0
    by_guardrail: Counter[str] = Counter()
    redaction_matches = 0
    by_redaction_label: Counter[str] = Counter()
    tool_calls = 0
    by_tool: Counter[str] = Counter()
    tool_errors = 0
    by_tool_error: Counter[str] = Counter()
    memory_writes = 0

    for r in records:
        outcomes[r.get("outcome", "unknown")] += 1

        if r.get("guardrails_passed") is False:
            guardrail_blocks += 1
        for v in r.get("guardrail_verdicts", []):
            if isinstance(v, dict) and v.get("passed") is False:
                by_guardrail[v.get("id", "?")] += 1

        san = r.get("sanitization")
        if isinstance(san, dict):
            for red in san.get("redacted", []):
                redaction_matches += 1
                if isinstance(red, dict):
                    by_redaction_label[red.get("label", "secret")] += 1

        for entry in r.get("tool_log", []):
            if not isinstance(entry, dict):
                continue
            name = entry.get("name", "?")
            if "error" in entry:
                tool_errors += 1
                by_tool_error[name] += 1
            else:
                tool_calls += 1
                by_tool[name] += 1

        if r.get("memory_delta"):
            memory_writes += 1

    return {
        "turns": len(records),
        "outcomes": dict(outcomes),
        "guardrail_blocks": guardrail_blocks,
        "blocks_by_guardrail": dict(by_guardrail),
        "redaction_matches": redaction_matches,
        "redactions_by_label": dict(by_redaction_label),
        "tool_calls": tool_calls,
        "tool_calls_by_name": dict(by_tool),
        "tool_errors": tool_errors,
        "tool_errors_by_name": dict(by_tool_error),
        "memory_writes": memory_writes,
    }


def format_summary(summary: dict[str, Any]) -> str:
    """Render a summary dict as a compact human-readable block."""
    lines: list[str] = []
    lines.append(f"turns:            {summary['turns']}")

    def _kv(d: dict[str, Any]) -> str:
        return ", ".join(f"{k}={v}" for k, v in d.items()) if d else "-"

    lines.append(f"outcomes:         {_kv(summary['outcomes'])}")
    lines.append(f"guardrail blocks: {summary['guardrail_blocks']}"
                 + (f"  ({_kv(summary['blocks_by_guardrail'])})"
                    if summary['blocks_by_guardrail'] else ""))
    lines.append(f"redactions:       {summary['redaction_matches']}"
                 + (f"  ({_kv(summary['redactions_by_label'])})"
                    if summary['redactions_by_label'] else ""))
    lines.append(f"tool calls:       {summary['tool_calls']}"
                 + (f"  ({_kv(summary['tool_calls_by_name'])})"
                    if summary['tool_calls_by_name'] else ""))
    lines.append(f"tool errors:      {summary['tool_errors']}"
                 + (f"  ({_kv(summary['tool_errors_by_name'])})"
                    if summary['tool_errors_by_name'] else ""))
    lines.append(f"memory writes:    {summary['memory_writes']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    audit_dir = Path(argv[0]) if argv else Path("audit")
    if not audit_dir.exists():
        print(f"audit dir not found: {audit_dir}")
        return 1
    print(format_summary(summarize_audit(audit_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
