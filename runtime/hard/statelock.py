"""Cross-process state integrity: atomic writes + an advisory file lock.

Two failure modes the persistence layer must survive, both deterministic and
model-independent, so both live in the hard layer:

1. **Partial write.** A crash mid-write leaves a half-written JSON file that
   fails to parse on next load. `write_atomic` writes a temp file and then
   `os.replace`s it into place — atomic on POSIX and on Windows when source
   and destination share a filesystem. A reader sees either the old file or
   the new one, never a torn one.

2. **Concurrent writers.** Two agent processes persisting the same `state/`
   at once can interleave and clobber each other. `state_lock` serializes
   them with an advisory `O_EXCL` lock file: whoever creates it holds the
   lock; others spin with backoff until they get it or time out.

Honest limits:
- The lock is **advisory** — it only constrains processes that also go
  through `state_lock`. It is not an OS mandatory lock.
- A crashed holder leaves a **stale** lock file. We break locks older than
  `stale_after` so a crash cannot wedge the agent forever; tune to taste.
- This prevents *corruption* from concurrent writes. It does **not** prevent
  *lost updates* between two long-lived agents that each loaded the state at
  startup — the later writer still wins. For real multi-writer correctness,
  split state by tenant (see ARCHITECTURE.md's honest list).
"""
from __future__ import annotations
import contextlib
import os
import time
from pathlib import Path


class LockTimeout(RuntimeError):
    """Could not acquire the state lock within the timeout."""


def write_atomic(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write `text` to `path` atomically (temp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            if tmp.exists():
                tmp.unlink()


@contextlib.contextmanager
def state_lock(target_dir: Path, *, timeout: float = 10.0,
               poll: float = 0.05, stale_after: float = 60.0):
    """Advisory cross-process lock for a state directory.

    Acquires `<target_dir>/.state.lock` via `O_EXCL`, retrying with backoff
    until `timeout`. Breaks a lock older than `stale_after` (a crashed
    holder). Releases on exit. Raises `LockTimeout` if it cannot acquire.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    lock_path = target_dir / ".state.lock"
    start = time.monotonic()

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(fd, f"{os.getpid()} {time.time():.0f}".encode("ascii"))
            finally:
                os.close(fd)
            break
        except FileExistsError:
            # Held by someone. Break it if it is stale, else wait.
            try:
                age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue  # released between our open and our stat; retry
            if age > stale_after:
                with contextlib.suppress(FileNotFoundError):
                    lock_path.unlink()
                continue
            if time.monotonic() - start >= timeout:
                raise LockTimeout(
                    f"could not acquire {lock_path} within {timeout:g}s")
            time.sleep(poll)

    try:
        yield lock_path
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()
