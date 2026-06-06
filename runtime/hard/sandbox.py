"""Deterministic execution bounds for tool calls. Hard layer.

The agent never trusts a tool to return promptly or to fail cleanly. This
module wraps a tool call so that a hang, a crash, or a runaway tool becomes a
structured error the runner can log and feed back to the LLM as a rejected
result — instead of blocking the turn forever or taking down the process.

Two modes:

- **in-process** (default): runs the tool in a worker thread and enforces a
  timeout on the *wait*. A hung tool's thread cannot be force-killed (a
  CPython limitation), so the thread may linger in the background, but the
  turn regains control immediately and records a `ToolTimeout`. Use for
  trusted, pure-Python tools (the common case).

- **isolated** (`isolated=True`): runs the tool in a separate process via
  multiprocessing "spawn". A hung or crashing tool is *contained* —
  `terminate()` reclaims it, and a segfault / `os._exit` in the tool does not
  take the agent down. The tool callable and its args/result must be
  picklable (module-level functions qualify; closures do not). Use for tools
  that shell out or touch fragile native code.

What this is NOT: a security sandbox. An isolated tool still runs with the
agent's own OS permissions — it bounds *liveness and blast radius*, not
*privilege*. For untrusted code you need OS-level confinement (containers,
seccomp, a restricted user). See the honest list in ARCHITECTURE.md.
"""
from __future__ import annotations
import multiprocessing as _mp
import queue as _queue
import threading
from typing import Any, Callable

DEFAULT_TIMEOUT_S: float = 10.0

ToolFn = Callable[[dict], Any]


class ToolTimeout(RuntimeError):
    """The tool did not return within its wall-clock budget."""


class ToolCrashed(RuntimeError):
    """The tool raised, or (isolated) its process died without a result."""


def run_guarded(fn: ToolFn, args: dict, *,
                timeout_s: float | None = DEFAULT_TIMEOUT_S,
                isolated: bool = False) -> Any:
    """Run `fn(args)` under a wall-clock bound. Returns the tool's result, or
    raises `ToolTimeout` / `ToolCrashed`. Never blocks past `timeout_s`
    (plus a short reap window in isolated mode).
    """
    if timeout_s is None or timeout_s <= 0:
        timeout_s = DEFAULT_TIMEOUT_S
    if isolated:
        return _run_isolated(fn, args, timeout_s)
    return _run_threaded(fn, args, timeout_s)


def _run_threaded(fn: ToolFn, args: dict, timeout_s: float) -> Any:
    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            box["ok"] = fn(args)
        except BaseException as e:  # contain everything the tool can throw
            box["err"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise ToolTimeout(f"tool did not return within {timeout_s:g}s")
    if "err" in box:
        raise ToolCrashed(f"tool raised: {box['err']!r}")
    return box.get("ok")


def _isolated_target(q: Any, fn: ToolFn, args: dict) -> None:
    try:
        q.put(("ok", fn(args)))
    except BaseException as e:
        q.put(("err", repr(e)))


def _run_isolated(fn: ToolFn, args: dict, timeout_s: float) -> Any:
    ctx = _mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_isolated_target, args=(q, fn, args), daemon=True)
    p.start()
    try:
        try:
            kind, payload = q.get(timeout=timeout_s)
        except _queue.Empty:
            raise ToolTimeout(f"tool did not return within {timeout_s:g}s")
    finally:
        if p.is_alive():
            p.terminate()
        p.join(1)
    if kind == "err":
        raise ToolCrashed(f"tool raised: {payload}")
    return payload
