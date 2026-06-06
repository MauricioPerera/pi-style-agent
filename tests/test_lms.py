"""Tests for runtime.soft.lms (LM Studio CLI wrapper + warmup helper).

The lms-CLI tests use subprocess mocking (no real CLI calls). The
is_loaded + warmup_embeddings tests probe the local server and skip
cleanly if it''s not reachable.
"""
import json
import os
import subprocess
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

from runtime.soft.lms import (
    _DEFAULT_LMS_PATHS, find_lms, is_loaded, lms_load, warmup_embeddings,
)


# --- pure helpers, no network --------------------------------------------

class TestFindLms(unittest.TestCase):
    def test_explicit_env_wins(self):
        with patch.dict(os.environ, {"LMS_CLI": "/nonexistent/lms"}):
            self.assertIsNone(find_lms())

    def test_explicit_env_used_when_exists(self):
        # Create a tiny dummy file and point LMS_CLI at it.
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".exe") as fh:
            dummy = fh.name
        try:
            with patch.dict(os.environ, {"LMS_CLI": dummy}):
                self.assertEqual(find_lms(), dummy)
        finally:
            os.unlink(dummy)

    def test_default_paths_chain(self):
        # If no env var is set, the function still resolves one of the
        # standard install paths (if present) or falls back to None.
        with patch.dict(os.environ, {"LMS_CLI": ""}):
            result = find_lms()
            # The result is either a string (something we found) or None.
            self.assertTrue(result is None or isinstance(result, str))

    def test_default_paths_skip_empty(self):
        # Regression: the env-LMS_CLI entry used to come back as Path(".")
        # which exists() == True and broke the chain.
        for p in _DEFAULT_LMS_PATHS:
            self.assertNotEqual(str(p), ".")
            self.assertNotEqual(str(p), "")


class TestLmsLoad(unittest.TestCase):
    @patch("runtime.soft.lms.find_lms", return_value="lms")
    @patch("subprocess.run")
    def test_lms_load_invokes_cli(self, mock_run, mock_find):
        mock_run.return_value = MagicMock(returncode=0, stdout="loaded", stderr="")
        rc, out, err = lms_load("some-model", yes=True)
        self.assertEqual(rc, 0)
        self.assertIn("loaded", out)
        # Args: [cli, "load", "some-model", "-y"]
        args = mock_run.call_args[0][0]
        self.assertIn("load", args)
        self.assertIn("some-model", args)
        self.assertIn("-y", args)

    @patch("runtime.soft.lms.find_lms", return_value="lms")
    @patch("subprocess.run")
    def test_lms_load_returns_error_code(self, mock_run, mock_find):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="model not found")
        rc, _, err = lms_load("missing-model", yes=True)
        self.assertEqual(rc, 1)
        self.assertIn("not found", err)

    def test_lms_load_handles_missing_cli(self):
        # If find_lms returns None, lms_load should report 127 (command
        # not found) without raising.
        with patch("runtime.soft.lms.find_lms", return_value=None):
            rc, out, err = lms_load("any-model", yes=True)
            self.assertEqual(rc, 127)
            self.assertIn("not found", err)

    @patch("runtime.soft.lms.find_lms", return_value="lms")
    @patch("subprocess.run")
    def test_lms_load_handles_timeout(self, mock_run, mock_find):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="lms", timeout=300)
        rc, _, err = lms_load("any-model", yes=True)
        self.assertEqual(rc, 124)
        self.assertIn("timed out", err)


# --- live tests: gated on the server being up ----------------------------

def server_reachable(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen("http://localhost:1234/v1/models", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


skip_live = unittest.skipUnless(server_reachable(), "LM Studio no esta corriendo")


@skip_live
class TestIsLoadedLive(unittest.TestCase):
    def test_known_model_loaded(self):
        self.assertTrue(is_loaded("text-embedding-embeddinggemma-300m-qat"))

    def test_unknown_model_not_loaded(self):
        self.assertFalse(is_loaded("definitely-not-a-real-model-xyz-12345"))

    def test_returns_bool(self):
        # The function must always return bool, never raise.
        self.assertIsInstance(is_loaded("anything"), bool)


@skip_live
class TestWarmupLive(unittest.TestCase):
    def test_warmup_already_loaded_returns_true(self):
        # The qat model is currently loaded in this session.
        ok = warmup_embeddings("text-embedding-embeddinggemma-300m-qat", verbose=False)
        self.assertTrue(ok)

    def test_warmup_skips_lms_when_already_loaded(self):
        # Verify warmup does not invoke lms when not needed: mock lms_load
        # and assert it is not called.
        from runtime.soft import lms as lms_mod
        with patch.object(lms_mod, "lms_load") as mock_load:
            ok = warmup_embeddings("text-embedding-embeddinggemma-300m-qat",
                                   verbose=False)
        self.assertTrue(ok)
        mock_load.assert_not_called()


@skip_live
class TestMultilingualLive(unittest.TestCase):
    """End-to-end test of the multilingual property of the loaded embedder.

    These skip cleanly if LM Studio is not running. The skip is per-test
    via the module-level decorator, so the whole class is skipped together.
    """

    def setUp(self):
        from runtime.soft.embeddings import build_lmstudio_retriever
        from runtime.soft.memory import MemoryItem
        self.idx = build_lmstudio_retriever()
        corpus = [
            ("city_es",  "Maria vive en Madrid"),
            ("city_en",  "Maria lives in Madrid"),
            ("food_es",  "Le gusta la paella"),
            ("food_en",  "She likes paella"),
        ]
        for k, v in corpus:
            self.idx.add(MemoryItem(k, v))

    def test_english_query_ranks_a_city(self):
        # A query in English should rank a "city_*" item first,
        # regardless of which language the city sentence is in.
        # (Cross-language match is fine; the test guards the topic.)
        results = self.idx.search("where does Maria live", k=2)
        self.assertTrue(results[0][0].key.startswith("city_"),
                        msg="top-1 was " + results[0][0].key)

    def test_spanish_query_ranks_a_city(self):
        results = self.idx.search("donde vive Maria", k=2)
        self.assertTrue(results[0][0].key.startswith("city_"),
                        msg="top-1 was " + results[0][0].key)

    def test_cross_language_retrieval_top1_is_city(self):
        # A query in any of the two languages should rank a "city" item
        # above any "food" item, regardless of which language the city
        # sentence is in.
        for q in ["where does she live", "donde vive ella"]:
            results = self.idx.search(q, k=2)
            top1 = results[0][0].key
            self.assertTrue(top1.startswith("city_"),
                            f"query={q!r} top1={top1!r}, expected city_*")

    def test_matryoshka_dim_256_preserves_multilingual(self):
        # Re-run with embed_dim=256 and verify the same property holds.
        from runtime.soft.embeddings import build_lmstudio_embedder
        from runtime.soft.memory import EmbeddingRetriever, MemoryItem
        embed = build_lmstudio_embedder()
        idx = EmbeddingRetriever(embed, embed_dim=256)
        idx.add(MemoryItem("city_es", "Maria vive en Madrid"))
        idx.add(MemoryItem("city_en", "Maria lives in Madrid"))
        idx.add(MemoryItem("food_es", "Le gusta la paella"))
        results = idx.search("donde vive Maria", k=1)
        self.assertTrue(results[0][0].key.startswith("city_"))
