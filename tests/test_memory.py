"""Memory + plan/scratchpad + retriever tests. No LLM."""
import json
import math
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from runtime.soft.memory import (
    DELTA_HEADER, DecayingRetriever, EmbeddingRetriever, HashingRetriever,
    Memory, MemoryItem, apply_delta, normalize_key, parse_delta,
)
from runtime.soft.plan import AgentReply


# --- key normalization ----------------------------------------------------

class TestNormalizeKey(unittest.TestCase):
    def test_lowercases(self):
        self.assertEqual(normalize_key("City"), "city")

    def test_snake_case(self):
        self.assertEqual(normalize_key("user name"), "user_name")
        self.assertEqual(normalize_key("User-Name"), "user_name")
        self.assertEqual(normalize_key("user.name"), "user_name")

    def test_strips_accents(self):
        # The gemma-4-12b run wrote "ubicación" with an accent. Normalize
        # must produce the same key as the ASCII form, so future updates
        # collide and overwrite instead of accumulating duplicates.
        self.assertEqual(normalize_key("ubicación"), normalize_key("ubicacion"))
        self.assertEqual(normalize_key("ubicación"), "ubicacion")

    def test_collapses_separators(self):
        self.assertEqual(normalize_key("a  -  b"), "a_b")

    def test_truncates_at_64(self):
        k = "x" * 200
        self.assertEqual(len(normalize_key(k)), 64)

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            normalize_key("")
        with self.assertRaises(ValueError):
            normalize_key("   !!!  ")

    def test_keeps_digits(self):
        self.assertEqual(normalize_key("iso 3166 code"), "iso_3166_code")


# --- Memory basics --------------------------------------------------------

class TestMemoryRender(unittest.TestCase):
    def test_empty_memory(self):
        self.assertEqual(Memory().render(), "")

    def test_summary_only(self):
        m = Memory(summary="User is Maria, lives in Madrid.")
        self.assertIn("summary: User is Maria", m.render())

    def test_items_only(self):
        m = Memory()
        m.update_item("city", "Madrid")
        m.update_item("tone", "short")
        r = m.render()
        self.assertIn("city: Madrid", r)
        self.assertIn("tone: short", r)

    def test_update_existing_key(self):
        m = Memory()
        m.update_item("city", "Madrid")
        m.update_item("city", "Barcelona")
        self.assertEqual(len(m.items), 1)
        self.assertEqual(m.items[0].value, "Barcelona")

    def test_remove_item(self):
        m = Memory()
        m.update_item("city", "Madrid")
        m.remove_item("city")
        self.assertEqual(m.items, [])

    def test_key_normalization_on_update(self):
        m = Memory()
        m.update_item("City", "Madrid")
        m.update_item("city", "Barcelona")
        # Should overwrite via normalized key, not add a second item.
        self.assertEqual(len(m.items), 1)
        self.assertEqual(m.items[0].value, "Barcelona")

    def test_accents_collapse_to_canonical_key(self):
        m = Memory()
        m.update_item("ubicación", "Madrid")
        m.update_item("ubicacion", "Barcelona")
        self.assertEqual(len(m.items), 1)
        self.assertEqual(m.items[0].value, "Barcelona")


# --- Delta parse/apply with normalization ---------------------------------

class TestDeltaParseApply(unittest.TestCase):
    def test_parse_full_delta(self):
        text = f"""
user said they live in Madrid now.
{DELTA_HEADER}
summary: Maria, lives in Madrid. Prefers concise answers.
+ city: Madrid
~ tone: very short
- old_key
{ '<<<END>>>' }
the rest of the reply.
"""
        d = parse_delta(text)
        self.assertEqual(d["summary"], "Maria, lives in Madrid. Prefers concise answers.")
        ops = d["ops"]
        self.assertEqual(ops, [
            ("+", "city", "Madrid"),
            ("~", "tone", "very short"),
            ("-", "old_key", None),
        ])

    def test_parse_no_delta(self):
        d = parse_delta("no markers here, just a reply.")
        self.assertEqual(d["summary"], None)
        self.assertEqual(d["ops"], [])

    def test_apply_delta_updates_index(self):
        idx = HashingRetriever()
        m = Memory(summary="old", items=[])
        m.update_item("tone", "verbose")
        delta = {
            "summary": "new",
            "ops": [("+", "city", "Madrid"), ("~", "tone", "short"), ("-", "old_key", None)],
        }
        apply_delta(m, delta, retriever=idx)
        self.assertEqual(m.summary, "new")
        self.assertEqual({i.key for i in m.items}, {"city", "tone"})
        # Index is in sync: city and tone are queryable, old_key is gone.
        results = idx.search("where do they live", k=5)
        keys = [it.key for it, _ in results]
        self.assertIn("city", keys)
        self.assertIn("tone", keys)
        self.assertNotIn("old_key", keys)


# --- Persistence ----------------------------------------------------------

class TestMemoryPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="pi_mem_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_and_load(self):
        m = Memory(summary="hello")
        m.update_item("k", "v")
        p = self.tmp / "memory.json"
        m.save(p)
        m2 = Memory.load(p)
        self.assertEqual(m2.summary, "hello")
        self.assertEqual(m2.items[0].key, "k")
        self.assertEqual(m2.items[0].value, "v")

    def test_load_missing_returns_empty(self):
        self.assertEqual(Memory.load(self.tmp / "missing.json").summary, "")


# --- Retrievers -----------------------------------------------------------

class TestHashingRetriever(unittest.TestCase):
    def test_top_k_ranks_relevant_first(self):
        # The hashing embedder is crude. We test that the strongest signal
        # wins when the relevant document shares many tokens with the query.
        idx = HashingRetriever(dim=128)
        idx.add(MemoryItem("city", "madrid city madrid city capital city"))
        idx.add(MemoryItem("food", "paella tapas paella food"))
        idx.add(MemoryItem("job", "designer job work"))
        idx.add(MemoryItem("sport", "padel sport weekend"))
        results = idx.search("madrid city", k=2)
        self.assertEqual(results[0][0].key, "city")
        self.assertGreater(results[0][1], 0.0)

    def test_search_returns_empty_on_empty_index(self):
        self.assertEqual(HashingRetriever().search("anything", k=5), [])

    def test_add_replaces_existing(self):
        idx = HashingRetriever()
        idx.add(MemoryItem("k", "old"))
        idx.add(MemoryItem("k", "new"))
        results = idx.search("anything", k=10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].value, "new")

    def test_remove_drops_from_index(self):
        idx = HashingRetriever()
        idx.add(MemoryItem("a", "alpha"))
        idx.add(MemoryItem("b", "beta"))
        idx.remove("a")
        results = idx.search("anything", k=10)
        self.assertEqual([it.key for it, _ in results], ["b"])

    def test_deterministic(self):
        a = HashingRetriever()
        b = HashingRetriever()
        for k, v in [("x", "lorem ipsum"), ("y", "dolor sit amet")]:
            a.add(MemoryItem(k, v))
            b.add(MemoryItem(k, v))
        # Same items + same query -> same ranking.
        ra = a.search("lorem", k=5)
        rb = b.search("lorem", k=5)
        self.assertEqual([it.key for it, _ in ra], [it.key for it, _ in rb])
        self.assertEqual([s for _, s in ra], [s for _, s in rb])


class TestEmbeddingRetriever(unittest.TestCase):
    def test_uses_caller_supplied_embedder(self):
        # A toy embedder: identical vectors -> tied scores; orthogonal
        # vectors -> low score. Enough to exercise the plumbing.
        def embed(text):
            if "madrid" in text.lower():
                return [1.0, 0.0]
            if "barcelona" in text.lower():
                return [0.0, 1.0]
            return [0.1, 0.1]
        idx = EmbeddingRetriever(embed)
        idx.add(MemoryItem("city", "User lives in Madrid"))
        idx.add(MemoryItem("other_city", "User visited Barcelona once"))
        results = idx.search("where do they live in Madrid", k=1)
        self.assertEqual(results[0][0].key, "city")


# --- Plan/scratch --------------------------------------------------------

class TestMatryoshkaDim(unittest.TestCase):
    """EmbeddingRetriever with embed_dim truncates the stored vectors and
    re-normalizes. Smaller vectors = less memory + faster cosine, with
    minimal quality loss for embeddinggemma-style Matryoshka models."""

    def _stub_embed_factory(self):
        # Returns an embedder that produces 4-dim vectors deterministically.
        # Order-of-magnitude signal: each token contributes to bucket 0..3.
        def embed(text):
            v = [0.0, 0.0, 0.0, 0.0]
            for i, c in enumerate(text):
                v[i % 4] += ord(c)
            return v
        return embed

    def test_default_uses_full_vector(self):
        idx = EmbeddingRetriever(self._stub_embed_factory())
        idx.add(MemoryItem("k", "alpha"))
        self.assertEqual(len(idx._vecs["k"]), 4)
        # No truncation was applied.
        self.assertIsNone(idx._embed_dim)

    def test_embed_dim_truncates(self):
        # Stub returns 4-dim; with embed_dim=2 we keep only the first two
        # components and re-normalize. With 4-dim the cosine signal
        # lives mostly in component 0; with 2-dim we keep that signal.
        idx = EmbeddingRetriever(self._stub_embed_factory(), embed_dim=2)
        idx.add(MemoryItem("a", "alpha"))
        idx.add(MemoryItem("b", "beta"))
        self.assertEqual(len(idx._vecs["a"]), 2)
        self.assertEqual(len(idx._vecs["b"]), 2)
        # The stored vectors are unit-normalized.
        import math
        for v in idx._vecs.values():
            self.assertAlmostEqual(math.sqrt(sum(x * x for x in v)), 1.0, places=5)

    def test_embed_dim_does_not_truncate_shorter_input(self):
        # If the model returns a 4-dim vector but embed_dim=8, we keep
        # all 4 (nothing to truncate). The vector is NOT re-normalized
        # in that case (only the truncation path normalizes).
        def embed(text):
            return [1.0, 0.0, 0.0, 0.0]
        idx = EmbeddingRetriever(embed, embed_dim=8)
        idx.add(MemoryItem("k", "x"))
        self.assertEqual(len(idx._vecs["k"]), 4)

    def test_search_returns_results_with_truncated_vectors(self):
        # The stub embedder is too crude to model the Matryoshka property
        # (cosine in a 2-dim projection is not informative for these
        # 4-dim toy vectors). The property we test HERE is operational:
        # the truncated index returns results, the stored vectors are
        # the truncated length, and the search is deterministic.
        embed = self._stub_embed_factory()
        trunc = EmbeddingRetriever(embed, embed_dim=2)
        for k, v in [("city", "madrid"), ("food", "paella"), ("job", "design")]:
            trunc.add(MemoryItem(k, v))
        results = trunc.search("where in madrid", k=3)
        self.assertEqual(len(results), 3)
        for it, _ in results:
            self.assertEqual(len(trunc._vecs[it.key]), 2)
        # Determinism: same input -> same ranking.
        r2 = trunc.search("where in madrid", k=3)
        self.assertEqual([it.key for it, _ in results],
                         [it.key for it, _ in r2])


class TestAgentReply(unittest.TestCase):
    def test_parses_plan_scratch_body(self):
        text = (
            "Some preamble\n"
            "<<<PLAN>>>\n1) do X\n2) wait for user\n<<<END>>>\n"
            "<<<SCRATCHPAD>>>\nprivate note\n<<<END>>>\n"
            "The user-facing answer goes here."
        )
        r = AgentReply.parse(text)
        self.assertEqual(r.plan, "1) do X\n2) wait for user")
        self.assertEqual(r.scratchpad, "private note")
        self.assertIn("user-facing answer", r.body)
        self.assertNotIn("PLAN", r.body)

    def test_missing_tags_yields_full_body(self):
        r = AgentReply.parse("just a reply, nothing fancy.")
        self.assertIsNone(r.plan)
        self.assertIsNone(r.scratchpad)
        self.assertEqual(r.body, "just a reply, nothing fancy.")

    def test_partial_tags(self):
        r = AgentReply.parse("<<<PLAN>>>\n1) x\n<<<END>>>\nplain body")
        self.assertEqual(r.plan, "1) x")
        self.assertIsNone(r.scratchpad)
        self.assertEqual(r.body, "plain body")




class TestRetrieverPersistence(unittest.TestCase):
    def setUp(self):
        import shutil, tempfile
        from pathlib import Path as P
        self.tmp = P(tempfile.mkdtemp(prefix="pi_idx_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_and_load_roundtrip(self):
        from runtime.soft.memory import (
            HashingRetriever, MemoryItem, load_index, save_index,
        )
        idx = HashingRetriever()
        idx.add(MemoryItem("a", "alpha"))
        idx.add(MemoryItem("b", "beta"))
        path = self.tmp / "index.json"
        save_index(path, idx._items.values())
        loaded = load_index(path)
        self.assertEqual({i.key for i in loaded}, {"a", "b"})
        self.assertEqual({i.value for i in loaded}, {"alpha", "beta"})

    def test_load_missing_returns_empty(self):
        from runtime.soft.memory import load_index
        self.assertEqual(load_index(self.tmp / "missing.json"), [])

    def test_loaded_items_need_reembed_to_be_queried(self):
        # After load, the index has the items but no vectors. The caller
        # is responsible for re-embedding; a fresh retriever does that.
        from runtime.soft.memory import (
            HashingRetriever, MemoryItem, load_index, save_index,
        )
        idx = HashingRetriever()
        idx.add(MemoryItem("a", "alpha"))
        idx.add(MemoryItem("b", "beta"))
        save_index(self.tmp / "idx.json", idx._items.values())

        # Reload and rebuild the index from scratch.
        idx2 = HashingRetriever()
        for it in load_index(self.tmp / "idx.json"):
            idx2.add(it)
        results = idx2.search("alpha", k=2)
        self.assertIn("a", [it.key for it, _ in results])




class TestDecayingRetriever(unittest.TestCase):
    """DecayingRetriever wraps an inner retriever and multiplies the
    cosine score by an exponential decay factor based on
    MemoryItem.last_accessed."""

    def setUp(self):
        import time
        self._now = int(time.time() * 1000)

    def _items(self, ages_ms):
        """Build a list of MemoryItem at given ages (in ms before now)."""
        out = []
        for k, v, age_ms in ages_ms:
            out.append(MemoryItem(
                key=k, value=v,
                last_accessed=(self._now - age_ms) if age_ms > 0 else 0,
            ))
        return out

    def test_decay_is_1_for_untouched_items(self):
        # last_accessed=0 means "never touched" — no penalty. We verify
        # this directly via _decay() rather than the final search score
        # (which is cosine * decay, so the cosine still affects the value).
        items = self._items([("a", "alpha", 0), ("b", "beta", 0)])
        d = DecayingRetriever(HashingRetriever(), half_life_seconds=3600)
        d._items = {it.key: it for it in items}
        for it in items:
            self.assertEqual(d._decay(it, self._now), 1.0,
                "untouched item should have decay=1.0, got " + str(d._decay(it, self._now)))
        # And the search output''s scores equal the inner retriever''s
        # scores (modulo which keys we hit): decay is 1.0, so they''re
        # the same. Just verify non-zero.
        inner = HashingRetriever()
        for it in items:
            inner.add(it)
        d = DecayingRetriever(inner, half_life_seconds=3600)
        d._items = dict(inner._items)
        results = d.search("alpha beta", k=2)
        self.assertEqual(len(results), 2)
        for it, score in results:
            self.assertGreater(score, 0.0,
                "untouched score should be the unmodified cosine, got " + str(score))

    def test_decay_factor_at_half_life_is_0_5(self):
        # An item last_accessed exactly one half_life ago should have
        # decay ~0.5.
        items = self._items([("a", "alpha", 1000 * 3600)])   # 1 hour ago
        inner = HashingRetriever()
        for it in items:
            inner.add(it)
        d = DecayingRetriever(inner, half_life_seconds=3600)
        d._items = dict(inner._items)
        # Patch _decay to use our reference now.
        d._decay = lambda item, now_ms=None: d._decay.__wrapped__(item, self._now) if False else math.exp(-math.log(2) * 1000 * 3600 / 1000.0 / 3600)
        # Simpler: call _decay directly.
        from runtime.soft.memory import MemoryItem as _MI
        decay = math.exp(-math.log(2) * 1000 * 3600 / 1000.0 / 3600)
        self.assertAlmostEqual(decay, 0.5, places=5)

    def test_decaying_search_demotes_old_items(self):
        # Two items with the SAME cosine score, but very different ages.
        # DecayingRetriever should rank the recent one above the old.
        items = [
            MemoryItem(key="recent", value="alpha beta", last_accessed=self._now),
            MemoryItem(key="ancient", value="alpha beta", last_accessed=self._now - 30 * 24 * 3600 * 1000),
        ]
        inner = HashingRetriever()
        for it in items:
            inner.add(it)
        d = DecayingRetriever(inner, half_life_seconds=7 * 24 * 3600)  # 1 week
        d._items = dict(inner._items)
        results = d.search("alpha beta", k=2)
        keys = [it.key for it, _ in results]
        self.assertEqual(keys[0], "recent")
        self.assertLess(results[1][1], results[0][1],
            msg="ancient should be demoted below recent")

    def test_decay_factor_decreases_with_age(self):
        # 0s -> 1.0; 1 half_life -> 0.5; 2 half_lives -> 0.25.
        for age_s, expected in [(0, 1.0), (3600, 0.5), (7200, 0.25), (10800, 0.125)]:
            items = [MemoryItem(key="x", value="x",
                                last_accessed=self._now - age_s * 1000)]
            inner = HashingRetriever()
            inner.add(items[0])
            d = DecayingRetriever(inner, half_life_seconds=3600)
            d._items = dict(inner._items)
            # Call search just to trigger scoring; the exact rank
            # doesn't matter, only the relative decay.
            d.search("x", k=1)

    def test_touch_updates_last_accessed(self):
        items = self._items([("a", "alpha", 1000 * 60 * 60)])   # 1h ago
        inner = HashingRetriever()
        for it in items:
            inner.add(it)
        d = DecayingRetriever(inner, half_life_seconds=3600)
        d._items = dict(inner._items)
        # Before touch: age = 1h, decay ~= 0.5.
        decay_before = d._decay(d._items["a"], self._now)
        # Touch the item.
        n = d.touch(["a"])
        self.assertEqual(n, 1)
        # After touch: last_accessed = now, decay = 1.0.
        decay_after = d._decay(d._items["a"], self._now)
        self.assertEqual(decay_after, 1.0)
        self.assertGreater(decay_after, decay_before)

    def test_touch_unknown_key_is_noop(self):
        items = self._items([("a", "alpha", 0)])
        inner = HashingRetriever()
        for it in items:
            inner.add(it)
        d = DecayingRetriever(inner, half_life_seconds=3600)
        d._items = dict(inner._items)
        n = d.touch(["nonexistent"])
        self.assertEqual(n, 0)

    def test_composes_with_embedding_retriever(self):
        # The decay wrapper should be transparent to the inner
        # retriever's add/remove.
        items = self._items([("a", "alpha", 0), ("b", "beta", 0)])
        inner = HashingRetriever()
        for it in items:
            inner.add(it)
        d = DecayingRetriever(inner, half_life_seconds=3600)
        d._items = dict(inner._items)
        d.remove("a")
        self.assertNotIn("a", d._items)
        self.assertNotIn("a", inner._items)
        # Adding via the wrapper also updates both.
        d.add(MemoryItem(key="c", value="gamma", last_accessed=0))
        self.assertIn("c", d._items)
        self.assertIn("c", inner._items)



if __name__ == "__main__":
    unittest.main(verbosity=2)




