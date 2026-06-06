"""Smoke test: the offline demo must run to completion.

This closes the gap that let `demo.py` rot: the 165 unit tests exercise the
agent loop directly, but nobody verified that `demo.py` — a documented entry
point in the README — actually runs end-to-end. It crashed on turn 2 for a
while (a list/str mismatch on `tool_called`) and no test caught it.

The test runs the demo in a subprocess with the deterministic stub provider
(no LM Studio needed) and asserts a clean exit. If the demo's asserts drift
from the agent loop again, this fails loudly.
"""
from __future__ import annotations
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestDemoSmoke(unittest.TestCase):
    def test_offline_demo_runs_to_completion(self):
        env = dict(os.environ)
        env["PI_LLM_PROVIDER"] = "stub"
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            [sys.executable, str(ROOT / "demo.py")],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            proc.returncode, 0,
            msg=f"demo.py exited {proc.returncode}\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
