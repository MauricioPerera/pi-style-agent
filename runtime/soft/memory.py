"""Long-term memory: two-level store, soft writes / hard reads.

Soft: the LLM decides WHAT to remember. The agent calls `commit(turn, llm_call)`
to ask the LLM to produce a delta (added/updated/removed) over the memory.

Hard: the resulting memory is plain text, lives in `long_term_mem` as a slot.
No schema magic on the LLM''s output. Replacement-friendly.

The two-level model:
  - `summary` (low churn, high signal): a paragraph about the user.
  - `items`  (high churn, structured): a list of {key, value} preferences.

`Memory.render(query=None, retriever=None, k=None)` produces the text for the
`long_term_mem` slot. With a retriever + query it picks the top-k items by
similarity to the query; without it, it dumps everything.

The seam: this module returns strings; the assembler turns those strings into
slot contents. Nothing here calls a tool that mutates state outside of the
return value.
"""
from __future__ import annotations
import json
import math
import time
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from runtime.hard.crypto import decrypt_str, encrypt_str, is_encrypted
from typing import Iterable, Protocol


# --- hard layer: key normalization -----------------------------------------

def normalize_key(raw: str) -> str:
    """Normalize a memory key: ASCII, lowercase, snake_case, max 64 chars.

    Lives in the hard layer: it is deterministic, has no model dependency,
    and is the same on every machine. The model can emit keys in Spanish,
    with accents, in CamelCase, with spaces — the runner normalizes before
    storing, so the index is consistent.
    """
    s = unicodedata.normalize("NFKD", raw)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        raise ValueError(f"clave de memoria invalida: {raw!r}")
    return s[:64]


# --- soft layer: memory record ---------------------------------------------

@dataclass
class MemoryItem:
    key: str
    value: str
    last_accessed: int = 0   # epoch ms; 0 = never touched. The chat loop updates
                              # this on every retrieval so the decay wrapper can
                              # demote items that haven''t come up in a while.

    def to_dict(self) -> dict:
        return {"key": self.key, "value": self.value, "last_accessed": self.last_accessed}

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryItem":
        return cls(key=d["key"], value=d["value"],
                   last_accessed=d.get("last_accessed", 0))


@dataclass
class Memory:
    summary: str = ""
    items: list[MemoryItem] = field(default_factory=list)

    def render(self, query: str | None = None,
               retriever: "Retriever | None" = None,
               k: int | None = None) -> str:
        """Format the memory as a single string for the `long_term_mem` slot.

        With `query` + `retriever` + `k`: include the summary (always) plus
        the top-k items by similarity to the query.
        Without: include everything (useful for tests and small memories).
        """
        parts: list[str] = []
        if self.summary.strip():
            parts.append(f"summary: {self.summary.strip()}")
        chosen = self._select_items(query, retriever, k)
        if chosen:
            parts.append("items:")
            for it in chosen:
                parts.append(f"  - {it.key}: {it.value}")
        return "\n".join(parts)

    def _select_items(self, query: str | None,
                      retriever: "Retriever | None",
                      k: int | None) -> list[MemoryItem]:
        if not self.items:
            return []
        if query is None or retriever is None or k is None:
            return self.items
        ranked = retriever.search(query, k=min(k, len(self.items)))
        return [it for it, _score in ranked]

    # --- mutators (called by the soft layer) ------------------------------

    def update_item(self, key: str, value: str) -> None:
        key = normalize_key(key)
        for it in self.items:
            if it.key == key:
                it.value = value
                return
        self.items.append(MemoryItem(key=key, value=value))

    def remove_item(self, key: str) -> None:
        key = normalize_key(key)
        self.items = [it for it in self.items if it.key != key]

    def set_summary(self, text: str) -> None:
        self.summary = text.strip()

    # --- persistence (hard: plain files, no magic) ------------------------

    @classmethod
    def load(cls, path: Path, passphrase: str | None = None) -> "Memory":
        if not path.exists():
            return cls()
        raw = path.read_text(encoding="utf-8")
        raw = _maybe_decrypt(raw, passphrase, path)
        data = json.loads(raw)
        m = cls(summary=data.get("summary", ""))
        m.items = [MemoryItem(**i) for i in data.get("items", [])]
        return m

    def save(self, path: Path, passphrase: str | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = json.dumps({
            "summary": self.summary,
            "items": [i.to_dict() for i in self.items],
        }, indent=2, ensure_ascii=False)
        if passphrase:
            blob = encrypt_str(blob, passphrase)
        path.write_text(blob + "\n", encoding="utf-8")


# --- retriever protocol + a tiny in-memory implementation ------------------

class Retriever(Protocol):
    def add(self, item: MemoryItem) -> None: ...
    def remove(self, key: str) -> None: ...
    def search(self, query: str, k: int = 5) -> list[tuple[MemoryItem, float]]: ...


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def _hash_embed(text: str, dim: int = 64) -> list[float]:
    """Deterministic, dependency-free pseudo-embedding: signed hash buckets.

    Good enough for a small in-memory index and for tests. In production you
    swap in a real model: same interface, different `embed()` body.
    """
    import hashlib
    v = [0.0] * dim
    for tok in _tokenize(text):
        h = hashlib.sha256(tok.encode("utf-8")).digest()
        # Use 8 bytes -> 64 bits, mapped to ±1 across `dim` buckets.
        for i in range(8):
            byte = h[i]
            for b in range(8):
                idx = (i * 8 + b) % dim
                if byte & (1 << b):
                    v[idx] += 1.0
                else:
                    v[idx] -= 1.0
    # L2 normalise so cosine = dot product.
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class HashingRetriever:
    """In-memory retriever. No model, no network. Same interface as a real
    one, so the assembler/agent do not care which is plugged in.

    Cosine similarity over hash-bucket embeddings. Deterministic across runs.
    """

    def __init__(self, dim: int = 64):
        self._dim = dim
        self._items: dict[str, MemoryItem] = {}
        self._vecs: dict[str, list[float]] = {}

    def add(self, item: MemoryItem) -> None:
        # If the key already exists, remove the old vector first so it
        # is not double-counted.
        self._items.pop(item.key, None)
        self._vecs.pop(item.key, None)
        self._items[item.key] = item
        self._vecs[item.key] = _hash_embed(item.value, self._dim)

    def remove(self, key: str) -> None:
        self._items.pop(key, None)
        self._vecs.pop(key, None)

    def search(self, query: str, k: int = 5) -> list[tuple[MemoryItem, float]]:
        qv = _hash_embed(query, self._dim)
        scored = [(it, _cosine(qv, self._vecs[it.key])) for it in self._items.values()]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]


class EmbeddingRetriever:
    """Retriever backed by an external embedding endpoint (LM Studio''s
    /v1/embeddings, OpenAI, etc.). Same interface as HashingRetriever.

    The embedding function is provided by the caller; this class just owns
    the index and the cosine math.

    Matryoshka support (embeddinggemma, NV-Embed, etc.):
      pass `embed_dim` at construction to truncate the stored vectors to
      that prefix and re-normalize. Smaller vectors = less memory + faster
      cosine. For embeddinggemma-300m the sweet spot is 256 for small
      corpora (no measurable quality loss vs 768); 128 starts to break on
      ambiguous queries. Pass `embed_dim=None` to use the full vector.
    """

    def __init__(self, embed_fn, embed_dim=None):
        self._embed = embed_fn          # (str) -> list[float]
        self._embed_dim = embed_dim      # Matryoshka truncation dim, or None
        self._items = {}
        self._vecs = {}

    def _vec(self, text_or_vec):
        """Return the (possibly truncated) vector for storage/query."""
        if isinstance(text_or_vec, str):
            v = self._embed(text_or_vec)
        else:
            v = text_or_vec
        if self._embed_dim is not None and len(v) > self._embed_dim:
            v = v[: self._embed_dim]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            v = [x / n for x in v]
        return list(v)

    def add(self, item):
        self._items.pop(item.key, None)
        self._vecs.pop(item.key, None)
        self._items[item.key] = item
        self._vecs[item.key] = self._vec(item.value)

    def remove(self, key):
        self._items.pop(key, None)
        self._vecs.pop(key, None)

    def search(self, query, k=5):
        qv = self._vec(query)
        scored = [(it, _cosine(qv, self._vecs[it.key])) for it in self._items.values()]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]



# --- soft-side: parse a memory delta emitted by an LLM ---------------------

DELTA_HEADER = "<<<MEMORY-DELTA>>>"


def parse_delta(text: str) -> dict:
    """Parse the agent''s memory update into a delta dict.

    The expected format (deliberately human-readable, easy to test):

        <<<MEMORY-DELTA>>>
        summary: <one paragraph>
        + key: value
        ~ key: value
        - key
        <<<END>>>
    """
    if DELTA_HEADER not in text:
        return {"summary": None, "ops": []}
    block = text.split(DELTA_HEADER, 1)[1]
    if "<<<END>>>" in block:
        block = block.split("<<<END>>>", 1)[0]

    summary: str | None = None
    ops: list[tuple[str, str, str | None]] = []  # (op, raw_key, value)

    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("summary:"):
            summary = line.split(":", 1)[1].strip()
        elif line.startswith("+ "):
            kv = line[2:].split(":", 1)
            if len(kv) == 2:
                ops.append(("+", kv[0].strip(), kv[1].strip()))
        elif line.startswith("~ "):
            kv = line[2:].split(":", 1)
            if len(kv) == 2:
                ops.append(("~", kv[0].strip(), kv[1].strip()))
        elif line.startswith("- "):
            ops.append(("-", line[2:].strip(), None))
    return {"summary": summary, "ops": ops}


def apply_delta(memory: Memory, delta: dict,
                retriever: Retriever | None = None) -> None:
    """Apply a parsed delta to a Memory in place. Failures are silent — the
    soft layer''s job is to be best-effort; the next turn will reconcile.

    If `retriever` is provided, items are also added/removed from the index
    after the in-memory mutation, so the index never gets out of sync.
    """
    if delta.get("summary"):
        memory.set_summary(delta["summary"])
    for op, raw_key, value in delta.get("ops", []):
        if op in ("+", "~"):
            try:
                norm = normalize_key(raw_key)
            except ValueError:
                continue
            # Detect collision with an existing key (different raw form
            # normalising to the same key): the LLM is overwriting.
            memory.update_item(norm, value or "")
            if retriever is not None:
                item = next((i for i in memory.items if i.key == norm), None)
                if item is not None:
                    retriever.add(item)
        elif op == "-":
            try:
                norm = normalize_key(raw_key)
            except ValueError:
                continue
            memory.remove_item(norm)
            if retriever is not None:
                retriever.remove(norm)


# --- retriever persistence -------------------------------------------------

def _maybe_decrypt(raw: str, passphrase: str | None, path: Path) -> str:
    """If `raw` is an encryption envelope, decrypt it (passphrase required);
    otherwise return it unchanged. Lets encrypted and plaintext state coexist
    and makes migration a no-op: a plaintext file loads fine even with a
    passphrase set, and the next save re-writes it encrypted.
    """
    if is_encrypted(raw):
        if not passphrase:
            raise ValueError(
                f"{path} is encrypted; set PI_STATE_PASSPHRASE to load it")
        return decrypt_str(raw, passphrase)
    return raw


def save_index(path: Path, items: Iterable["MemoryItem"],
               passphrase: str | None = None) -> None:
    """Persist a list of MemoryItem to JSON. Embeddings are NOT saved; the
    caller is expected to re-embed on load. The rationale: the embedder
    may have changed (e.g. switched from hash to a real model) and we
    don''t want stale vectors from a different embedding space.

    For the hashing embedder, this means a one-time re-embed cost on load.
    For an API-backed embedder, it means a few network calls. Both are
    cheap relative to the LLM calls they support.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps([i.to_dict() for i in items], indent=2, ensure_ascii=False)
    if passphrase:
        blob = encrypt_str(blob, passphrase)
    path.write_text(blob + "\n", encoding="utf-8")


def load_index(path: Path, passphrase: str | None = None) -> list["MemoryItem"]:
    if not path.exists():
        return []
    raw = _maybe_decrypt(path.read_text(encoding="utf-8"), passphrase, path)
    data = json.loads(raw)
    return [MemoryItem(**d) for d in data]



class DecayingRetriever:
    """Wrap any Retriever with an exponential time-decay on the score.

    Each MemoryItem carries `last_accessed` (epoch ms). When search()
    returns, the cosine score is multiplied by:

        decay = exp(-ln(2) * age_seconds / half_life)

    Items touched recently (or never touched = 0) get full weight;
    items not accessed in `half_life` seconds get half weight; etc.

    The chat loop is expected to call `touch(keys)` after every search
    so the items the user actually asked about rise to the top of
    future queries. Items that go unaccessed for long periods sink.

    Default half_life: 7 days. Override per-instance.
    """

    DEFAULT_HALF_LIFE_SECONDS = 7 * 24 * 3600   # 1 week

    def __init__(self, inner, half_life_seconds: int | None = None):
        self._inner = inner
        self._half_life = half_life_seconds or self.DEFAULT_HALF_LIFE_SECONDS
        # We need access to the items to read last_accessed. We grab
        # them from the inner retriever if it exposes _items; otherwise
        # we keep our own dict updated via touch().
        self._items: dict[str, MemoryItem] = {}
        # Mirror items from the inner retriever on add, so the wrapper
        # has them when scoring.
        if hasattr(inner, "_items"):
            self._items = dict(inner._items)

    def _decay(self, item: MemoryItem, now_ms: int | None = None) -> float:
        if item.last_accessed <= 0:
            return 1.0   # never touched: no penalty
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        age_s = max(0, (now_ms - item.last_accessed) / 1000.0)
        return math.exp(-math.log(2) * age_s / self._half_life)

    def add(self, item: MemoryItem) -> None:
        self._items[item.key] = item
        self._inner.add(item)

    def remove(self, key: str) -> None:
        self._items.pop(key, None)
        self._inner.remove(key)

    def search(self, query: str, k: int = 5) -> list[tuple[MemoryItem, float]]:
        raw = self._inner.search(query, k=max(k, k * 3))   # over-fetch to compensate
        now_ms = int(time.time() * 1000)
        decayed: list[tuple[MemoryItem, float]] = []
        for item, base in raw:
            local = self._items.get(item.key)
            if local is None:
                local = item   # fall back to whatever the inner returned
            decayed.append((local, base * self._decay(local, now_ms)))
        decayed.sort(key=lambda t: t[1], reverse=True)
        return decayed[:k]

    def touch(self, keys) -> int:
        """Update `last_accessed` to now for the given key(s).

        Call this from the chat loop after every search, with the top-k
        keys. Returns the number of items touched.
        """
        now_ms = int(time.time() * 1000)
        n = 0
        for k in keys:
            item = self._items.get(k)
            if item is not None:
                item.last_accessed = now_ms
                n += 1
        return n
