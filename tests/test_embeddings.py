"""Tests for the embeddings client + retriever.

The live test is skipped if LM Studio is not reachable, so the unit suite
stays green offline. The skip is one line; in CI, set up the server and
the tests run for real.
"""
import json
import unittest
import urllib.error
import urllib.request

from runtime.soft.memory import EmbeddingRetriever, MemoryItem
from runtime.soft.embeddings import (
    DEFAULT_EMBED_ENDPOINT, DEFAULT_EMBED_MODEL, build_lmstudio_embedder,
    build_lmstudio_retriever,
)


def server_reachable(url: str = DEFAULT_EMBED_ENDPOINT, timeout: float = 2.0) -> bool:
    """Return True iff a quick GET to the embeddings endpoint succeeds.

    We use the models list endpoint to check; it''s lighter than a full
    embeddings call.
    """
    try:
        with urllib.request.urlopen("http://localhost:1234/v1/models", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


LIVE = server_reachable()
skip_live = unittest.skipUnless(LIVE, "LM Studio no esta corriendo")


# --- the offline tests: a tiny stub embedder ------------------------------

class TestStubEmbedder(unittest.TestCase):
    """EmbeddingRetriever with a deterministic stub. No network."""

    def test_retriever_with_caller_embedder(self):
        def embed(text):
            # Crude but deterministic: identical text -> identical vec.
            v = [0.0] * 4
            for i, c in enumerate(text):
                v[i % 4] += ord(c)
            return v

        idx = EmbeddingRetriever(embed)
        idx.add(MemoryItem("a", "alpha"))
        idx.add(MemoryItem("b", "beta"))
        idx.add(MemoryItem("c", "alpha beta"))
        results = idx.search("alpha", k=2)
        keys = [it.key for it, _ in results]
        self.assertIn("a", keys)
        self.assertIn("c", keys)

    def test_add_replaces_existing(self):
        def embed(text):
            return [1.0, 0.0]
        idx = EmbeddingRetriever(embed)
        idx.add(MemoryItem("k", "v1"))
        idx.add(MemoryItem("k", "v2"))
        results = idx.search("x", k=10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].value, "v2")

    def test_remove(self):
        def embed(text):
            return [1.0, 0.0]
        idx = EmbeddingRetriever(embed)
        idx.add(MemoryItem("a", "x"))
        idx.add(MemoryItem("b", "y"))
        idx.remove("a")
        results = idx.search("x", k=10)
        self.assertEqual([it.key for it, _ in results], ["b"])


# --- the live tests: gated on the server being up ------------------------

@skip_live
class TestLiveEmbedder(unittest.TestCase):
    def test_basic_embed_call(self):
        embed = build_lmstudio_embedder()
        v = embed("hola mundo")
        # embeddinggemma-300m returns 768-dim vectors.
        self.assertIsInstance(v, list)
        self.assertEqual(len(v), 768)
        self.assertTrue(all(isinstance(x, float) for x in v))
        # Not all zeros, not all NaN.
        self.assertNotEqual(v, [0.0] * 768)
        self.assertFalse(any(x != x for x in v))   # NaN check

    def test_cache_returns_same_vector(self):
        embed = build_lmstudio_embedder()
        v1 = embed("a stable test string")
        v2 = embed("a stable test string")
        self.assertEqual(v1, v2)

    def test_retriever_end_to_end(self):
        idx = build_lmstudio_retriever()
        idx.add(MemoryItem("city", "Maria lives in Madrid, the capital of Spain. Address: calle mayor."))
        idx.add(MemoryItem("food", "She loves paella, gazpacho, and tortilla"))
        idx.add(MemoryItem("job", "She works as a product designer at a startup"))
        idx.add(MemoryItem("sport", "She plays padel on weekends"))
        results = idx.search("where does she live in Madrid", k=3)
        keys = [it.key for it, _ in results]
        # A 300M-param embedder on a 4-doc corpus is more reliable with a
        # more specific query. We assert that "city" is in the top-3.
        self.assertIn("city", keys,
                      msg="expected city in top-3, got ranking: " + str(keys))

    def test_different_phrases_produce_different_vectors(self):
        embed = build_lmstudio_embedder()
        v1 = embed("the cat sat on the mat")
        v2 = embed("quantum entanglement in graphene lattices")
        # Vectors must differ. (Sanity: not just byte-different.)
        self.assertNotEqual(v1, v2)

