"""Thin wrapper around the `lms` CLI shipped with LM Studio.

The CLI is the documented way to pre-load a model into the local server.
Calling it from Python is the cleanest way to avoid the 2-12s cold-start
hit on the first turn of the agent.

This module is small and pure. No imports from the rest of the project.
"""
from __future__ import annotations
import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


# Standard install paths for `lms`. The `LMS_CLI` env var is honored
# separately above; this list is the fallback chain.
_DEFAULT_LMS_PATHS = [
    Path.home() / ".lmstudio" / "bin" / "lms.exe",
    Path("C:/Users") / os.environ.get("USERNAME", "x") / ".lmstudio" / "bin" / "lms.exe",
    Path("/usr/local/bin/lms"),
    Path("/opt/lmstudio/bin/lms"),
]


def find_lms() -> str | None:
    """Return the path to the `lms` executable, or None if not found.

    Honors `LMS_CLI` env var first; then probes the standard install paths
    and finally `which lms` (via shutil.which).
    """
    import shutil
    explicit = os.environ.get("LMS_CLI", "").strip()
    if explicit:
        # User has spoken: LMS_CLI is exclusive. If they set it to a
        # non-existent path, fail loud rather than fall through to a
        # different binary the user didn''t ask for.
        return explicit if Path(explicit).exists() else None
    for p in _DEFAULT_LMS_PATHS:
        if not p:
            continue
        try:
            if Path(p).exists():
                return str(p)
        except OSError:
            continue
    on_path = shutil.which("lms")
    return on_path


def is_loaded(model_id: str) -> bool:
    """Return True iff `model_id` is currently loaded in the local server.

    Uses the LM Studio `/api/v0/models` endpoint (not the OpenAI-compatible
    /v1/models which has less detail). Returns False on any error.
    """
    try:
        with urllib.request.urlopen("http://localhost:1234/api/v0/models", timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
        return any(m.get("id") == model_id and m.get("state") == "loaded"
                   for m in data.get("data", []))
    except (urllib.error.URLError, OSError, ValueError):
        return False


def lms_load(model_id: str, yes: bool = True) -> tuple[int, str, str]:
    """Invoke `lms load <model_id>` to bring the model into memory.

    Returns (returncode, stdout, stderr). The CLI is interactive by
    default; pass yes=True (default) to auto-approve.

    If the model is already loaded, this is a fast no-op (lms still
    returns 0). If the server is not running, the CLI will report an
    error but we still return a non-zero exit code for the caller.
    """
    cli = find_lms()
    if not cli:
        return 127, "", "lms CLI not found on PATH or in standard install locations"
    args = [cli, "load", model_id]
    if yes:
        args.append("-y")
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=300)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "lms load timed out after 300s"
    except FileNotFoundError as e:
        return 127, "", str(e)


def warmup_embeddings(model_id: str = "text-embedding-embeddinggemma-300m-qat",
                      endpoint: str = "http://localhost:1234/v1/embeddings",
                      verbose: bool = True) -> bool:
    """Make sure the embedding model is loaded and warm.

    1. If the model is already loaded (per /api/v0/models), issue a tiny
       embedding call to wake it from IDLE state (~2s).
    2. If not loaded, call `lms load` and wait for it to finish, then
       issue the warmup call.
    3. If lms is not installed, fall back to just the warmup call; if
       that succeeds the model is loaded by some other means.

    Returns True on success (the model responds with a vector), False
    otherwise. Never raises; failures are logged when verbose=True.
    """
    def _ping() -> bool:
        body = json.dumps({"model": model_id, "input": "warmup"}).encode("utf-8")
        req = urllib.request.Request(endpoint, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read().decode("utf-8"))
                return bool(d.get("data"))
        except (urllib.error.URLError, OSError, ValueError):
            return False

    if is_loaded(model_id):
        if verbose:
            print(f"[lms] {model_id} already loaded; pinging to wake from IDLE")
        return _ping()

    if verbose:
        print(f"[lms] {model_id} not loaded; invoking lms load")
    rc, out, err = lms_load(model_id)
    if rc != 0:
        if verbose:
            print(f"[lms] lms load failed (rc={rc}): {err.strip() or out.strip()}")
        # Fall through to a ping anyway: maybe the model is loaded by
        # some other path and the CLI just got confused.
        return _ping()

    if verbose:
        print(f"[lms] loaded; pinging to confirm")
    return _ping()
