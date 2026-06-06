"""Encryption at rest for persisted agent state. Hard layer.

The persisted `state/` (memory + index) is the agent's long-term record of a
single user — exactly the data the sensitive use cases (legal, medical) cannot
leave lying in plaintext. This module encrypts a JSON string into a
self-describing envelope and back.

Design choices, and why:

- **No home-grown crypto.** Python's stdlib has no AEAD cipher, and rolling
  one would be malpractice in a project whose whole thesis is verifiable
  safety. We use `cryptography`'s Fernet (AES-128-CBC + HMAC-SHA256,
  authenticated) — vetted, and tamper-evident, so a corrupted or modified
  blob fails closed on decrypt instead of yielding garbage.
- **scrypt KDF from stdlib.** The key is derived from a passphrase with
  `hashlib.scrypt` (memory-hard, resists brute force). The random salt and
  the scrypt parameters travel *with* the ciphertext, so decrypt is
  self-describing — you need only the passphrase.
- **Optional dependency, fail loud.** `cryptography` is imported lazily so the
  rest of the project runs without it (same pattern as tiktoken). But if you
  ASK to encrypt and the library is missing, we raise — we never silently
  fall back to writing plaintext, which would be the worst outcome.

Envelope (itself JSON, so it sits happily in the same `.json` files):
  {"pi_enc": 1, "kdf": "scrypt", "salt": "<b64>", "n":..,"r":..,"p":..,
   "ct": "<fernet token>"}
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
from typing import Any

# scrypt cost parameters. n=2**14 is the standard interactive default
# (~tens of ms, ~16 MiB per derivation); raise for more brute-force
# resistance. OpenSSL caps memory at 32 MiB unless `maxmem` is raised, so we
# pass a generous `maxmem` to leave headroom if these are bumped.
_SCRYPT_N = 1 << 14
_SCRYPT_R = 8
_SCRYPT_P = 1
_DKLEN = 32
_SCRYPT_MAXMEM = 256 * 1024 * 1024


class CryptoUnavailable(RuntimeError):
    """Encryption was requested but the `cryptography` package is missing."""


def _fernet():
    try:
        from cryptography.fernet import Fernet
    except ImportError as e:  # pragma: no cover - environment dependent
        raise CryptoUnavailable(
            "encryption requested but `cryptography` is not installed. "
            "Install it (`pip install cryptography`) or run without a "
            "passphrase to persist plaintext."
        ) from e
    return Fernet


def _derive(passphrase: str, salt: bytes,
            n: int = _SCRYPT_N, r: int = _SCRYPT_R, p: int = _SCRYPT_P) -> bytes:
    """Passphrase + salt -> a urlsafe-base64 Fernet key (scrypt, stdlib)."""
    raw = hashlib.scrypt(passphrase.encode("utf-8"), salt=salt,
                         n=n, r=r, p=p, dklen=_DKLEN, maxmem=_SCRYPT_MAXMEM)
    return base64.urlsafe_b64encode(raw)


def encrypt_str(plaintext: str, passphrase: str) -> str:
    """Encrypt `plaintext` under `passphrase`. Returns a JSON envelope string.
    A fresh random salt is generated per call."""
    if not passphrase:
        raise ValueError("passphrase must be non-empty to encrypt")
    Fernet = _fernet()
    salt = os.urandom(16)
    key = _derive(passphrase, salt)
    token = Fernet(key).encrypt(plaintext.encode("utf-8"))
    return json.dumps({
        "pi_enc": 1, "kdf": "scrypt",
        "salt": base64.b64encode(salt).decode("ascii"),
        "n": _SCRYPT_N, "r": _SCRYPT_R, "p": _SCRYPT_P,
        "ct": token.decode("ascii"),
    })


def decrypt_str(envelope: str, passphrase: str) -> str:
    """Inverse of `encrypt_str`. Raises on a wrong passphrase or a tampered
    blob (Fernet authenticates)."""
    env = _parse(envelope)
    if env is None:
        raise ValueError("not a pi encryption envelope")
    Fernet = _fernet()
    salt = base64.b64decode(env["salt"])
    key = _derive(passphrase, salt, n=env.get("n", _SCRYPT_N),
                  r=env.get("r", _SCRYPT_R), p=env.get("p", _SCRYPT_P))
    return Fernet(key).decrypt(env["ct"].encode("ascii")).decode("utf-8")


def is_encrypted(text: str) -> bool:
    """True iff `text` is a pi encryption envelope. Lets the load path stay
    backward compatible: existing plaintext state still loads."""
    return _parse(text) is not None


def _parse(text: str) -> dict[str, Any] | None:
    try:
        d = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(d, dict) and d.get("pi_enc") == 1 and "ct" in d and "salt" in d:
        return d
    return None
