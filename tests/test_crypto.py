"""Tests for encryption at rest (runtime/hard/crypto.py) and its wiring into
the persistence layer (Memory + index).

These exercise the real `cryptography` backend; if it is ever absent the
import of `runtime.hard.crypto` still succeeds (lazy import) but these tests
will fail loudly at encrypt time — which is the intended behavior (we never
silently fall back to plaintext).
"""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

from runtime.hard.crypto import decrypt_str, encrypt_str, is_encrypted
from runtime.soft.memory import (
    Memory, MemoryItem, load_index, save_index,
)

PASS = "correct horse battery staple"


class TestCryptoPrimitive(unittest.TestCase):
    def test_round_trip(self):
        pt = json.dumps({"summary": "Maria vive en Madrid"}, ensure_ascii=False)
        env = encrypt_str(pt, PASS)
        self.assertEqual(decrypt_str(env, PASS), pt)

    def test_envelope_does_not_leak_plaintext(self):
        env = encrypt_str("the secret is Madrid", PASS)
        self.assertNotIn("Madrid", env)
        self.assertTrue(is_encrypted(env))

    def test_wrong_passphrase_raises(self):
        env = encrypt_str("data", PASS)
        with self.assertRaises(Exception):
            decrypt_str(env, "wrong passphrase")

    def test_tamper_is_detected(self):
        env = encrypt_str("data", PASS)
        d = json.loads(env)
        d["ct"] = d["ct"][:-4] + "AAAA"
        with self.assertRaises(Exception):
            decrypt_str(json.dumps(d), PASS)

    def test_plaintext_is_not_flagged_encrypted(self):
        self.assertFalse(is_encrypted('{"summary": "x", "items": []}'))
        self.assertFalse(is_encrypted("not even json"))

    def test_empty_passphrase_rejected(self):
        with self.assertRaises(ValueError):
            encrypt_str("data", "")


class TestEncryptedPersistence(unittest.TestCase):
    def _tmp(self):
        return Path(tempfile.mkdtemp())

    def test_memory_save_is_encrypted_on_disk(self):
        d = self._tmp()
        p = d / "memory.json"
        m = Memory(summary="Maria vive en Madrid")
        m.items = [MemoryItem(key="city", value="Madrid")]
        m.save(p, passphrase=PASS)
        on_disk = p.read_text(encoding="utf-8")
        self.assertTrue(is_encrypted(on_disk))
        self.assertNotIn("Madrid", on_disk)

    def test_memory_round_trip_with_passphrase(self):
        d = self._tmp()
        p = d / "memory.json"
        m = Memory(summary="Maria vive en Madrid")
        m.items = [MemoryItem(key="city", value="Madrid")]
        m.save(p, passphrase=PASS)
        loaded = Memory.load(p, passphrase=PASS)
        self.assertEqual(loaded.summary, "Maria vive en Madrid")
        self.assertEqual(loaded.items[0].value, "Madrid")

    def test_load_encrypted_without_passphrase_raises(self):
        d = self._tmp()
        p = d / "memory.json"
        Memory(summary="x").save(p, passphrase=PASS)
        with self.assertRaises(ValueError):
            Memory.load(p)  # no passphrase

    def test_plaintext_loads_even_with_passphrase_set(self):
        # migration path: an existing plaintext file must still load
        d = self._tmp()
        p = d / "memory.json"
        Memory(summary="plain").save(p)  # no passphrase -> plaintext
        self.assertFalse(is_encrypted(p.read_text(encoding="utf-8")))
        loaded = Memory.load(p, passphrase=PASS)  # passphrase set, file plaintext
        self.assertEqual(loaded.summary, "plain")

    def test_index_round_trip_encrypted(self):
        d = self._tmp()
        p = d / "index.json"
        items = [MemoryItem(key="city", value="Madrid"),
                 MemoryItem(key="name", value="Maria")]
        save_index(p, items, passphrase=PASS)
        self.assertNotIn("Madrid", p.read_text(encoding="utf-8"))
        loaded = load_index(p, passphrase=PASS)
        self.assertEqual({i.value for i in loaded}, {"Madrid", "Maria"})


if __name__ == "__main__":
    unittest.main()
