"""Tests for the deterministic tool-execution bounds (runtime/hard/sandbox.py).

Worker functions are module-level so the isolated (spawn) mode can pickle
them. Timeouts are kept short (sub-second) to keep the suite fast.
"""
from __future__ import annotations
import time
import unittest

from runtime.hard.sandbox import (
    DEFAULT_TIMEOUT_S, ToolCrashed, ToolTimeout, run_guarded,
)
from runtime.hard.tools import tool_exec_opts


# --- module-level tools (picklable for isolated mode) ----------------------
def _good(args):
    return {"tool": "t", "ok": True, "data": {"value": args["x"] + 1}}


def _boom(args):
    raise ValueError("intentional")


def _hang(args):
    time.sleep(5)


class TestThreadedMode(unittest.TestCase):
    def test_success(self):
        self.assertEqual(run_guarded(_good, {"x": 41})["data"]["value"], 42)

    def test_crash_becomes_toolcrashed(self):
        with self.assertRaises(ToolCrashed):
            run_guarded(_boom, {})

    def test_timeout_becomes_tooltimeout(self):
        with self.assertRaises(ToolTimeout):
            run_guarded(_hang, {}, timeout_s=0.3)

    def test_lambda_is_fine_in_process(self):
        # in-process mode does not pickle, so closures/lambdas work
        self.assertEqual(run_guarded(lambda a: a["n"] * 2, {"n": 5}), 10)

    def test_none_timeout_falls_back_to_default(self):
        # a fast tool with timeout_s=None must still return (uses the default)
        self.assertIsNotNone(run_guarded(_good, {"x": 0}, timeout_s=None))


class TestIsolatedMode(unittest.TestCase):
    def test_success(self):
        out = run_guarded(_good, {"x": 1}, isolated=True)
        self.assertEqual(out["data"]["value"], 2)

    def test_crash_is_contained(self):
        with self.assertRaises(ToolCrashed):
            run_guarded(_boom, {}, isolated=True)

    def test_hung_process_is_killed(self):
        start = time.monotonic()
        with self.assertRaises(ToolTimeout):
            run_guarded(_hang, {}, isolated=True, timeout_s=0.4)
        # the timeout must actually bound the wait (not run the full 5s sleep)
        self.assertLess(time.monotonic() - start, 3.0)


class TestToolExecOpts(unittest.TestCase):
    def test_reads_declared_bounds(self):
        contract = {"tools": [
            {"name": "shell", "timeout_s": 2.5, "isolated": True},
            {"name": "calc"},
        ]}
        self.assertEqual(tool_exec_opts(contract, "shell"), (2.5, True))

    def test_absent_keys_default_in_process(self):
        contract = {"tools": [{"name": "calc"}]}
        self.assertEqual(tool_exec_opts(contract, "calc"), (None, False))

    def test_unknown_tool(self):
        self.assertEqual(tool_exec_opts({"tools": []}, "nope"), (None, False))


if __name__ == "__main__":
    unittest.main()
