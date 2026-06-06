"""Tests for cross-process state integrity (runtime/hard/statelock.py):
atomic writes and the advisory lock.
"""
from __future__ import annotations
import os
import tempfile
import time
import unittest
from pathlib import Path

from runtime.hard.statelock import LockTimeout, state_lock, write_atomic


class TestWriteAtomic(unittest.TestCase):
    def _tmp(self):
        return Path(tempfile.mkdtemp())

    def test_writes_content(self):
        d = self._tmp()
        p = d / "x.json"
        write_atomic(p, '{"a": 1}\n')
        self.assertEqual(p.read_text(encoding="utf-8"), '{"a": 1}\n')

    def test_no_tmp_file_left_behind(self):
        d = self._tmp()
        write_atomic(d / "x.json", "data")
        self.assertEqual(list(d.glob("*.tmp")), [])

    def test_overwrite_replaces(self):
        d = self._tmp()
        p = d / "x.json"
        write_atomic(p, "old")
        write_atomic(p, "new")
        self.assertEqual(p.read_text(encoding="utf-8"), "new")

    def test_creates_parent_dirs(self):
        d = self._tmp()
        p = d / "nested" / "deep" / "x.json"
        write_atomic(p, "ok")
        self.assertTrue(p.exists())


class TestStateLock(unittest.TestCase):
    def _tmp(self):
        return Path(tempfile.mkdtemp())

    def test_acquire_and_release(self):
        d = self._tmp()
        with state_lock(d) as lock_path:
            self.assertTrue(lock_path.exists())
        # released on exit
        self.assertFalse((d / ".state.lock").exists())

    def test_mutual_exclusion_times_out(self):
        d = self._tmp()
        with state_lock(d):
            with self.assertRaises(LockTimeout):
                with state_lock(d, timeout=0.2, poll=0.02):
                    pass

    def test_reacquire_after_release(self):
        d = self._tmp()
        with state_lock(d):
            pass
        # second acquisition must succeed now that the first released
        with state_lock(d, timeout=1.0) as lock_path:
            self.assertTrue(lock_path.exists())

    def test_breaks_stale_lock(self):
        d = self._tmp()
        lock = d / ".state.lock"
        lock.write_text("99999 0", encoding="ascii")
        old = time.time() - 120
        os.utime(lock, (old, old))
        # stale_after=60 -> the 120s-old lock is broken and we acquire
        with state_lock(d, stale_after=60, timeout=1.0) as lock_path:
            self.assertTrue(lock_path.exists())


if __name__ == "__main__":
    unittest.main()
