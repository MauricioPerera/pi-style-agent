"""Embeddings client + retriever factory.

The seam is small: the only thing the rest of the project needs is a function
`build_lmstudio_embedder()` that returns a `(str) -> list[float]` callable,
and a `build_lmstudio_retriever()` that wraps it in an EmbeddingRetriever.

Endpoint via `PI_LLM_ENDPOINT` (default http://localhost:1234/v1/embeddings).
Model via `PI_EMBED_MODEL` (default text-embedding-embeddinggemma-300m-qat,
the Q4_0-quantized variant; ~30% smaller than the Q8_0 reference with
negligible quality loss).

Matryoshka truncation dim: pass `embed_dim` to `build_lmstudio_retriever`
or set `PI_EMBED_DIM` in the env. Default is 256, which preserves retrieval
quality for embeddinggemma-300m on corpora up to a few hundred items and
cuts vector storage + cosine cost to 1/3.
"""
from __future__ import annotations
import json
import os
import urllib.error
import urllib.request
from typing import Callable

from .memory import EmbeddingRetriever


# The Q4_0-QAT variant is what we have on disk and what the agent
# should default to: ~30% smaller than the Q8_0 reference (~219 MB vs
# 313 MB) with negligible quality loss (cosine ~0.97 to the Q8_0 for
# the same text, identical top-1 ranking on small corpora). Override
# with PI_EMBED_MODEL=text-embedding-embeddinggemma-300m to use the
# larger reference model instead.
DEFAULT_EMBED_MODEL = os.environ.get("PI_EMBED_MODEL",
    "text-embedding-embeddinggemma-300m-qat")
DEFAULT_EMBED_ENDPOINT = "http://localhost:1234/v1/embeddings"
DEFAULT_TIMEOUT = int(os.environ.get("PI_LLM_TIMEOUT", "300"))

# Recommended Matryoshka dim for embeddinggemma-300m on small corpora.
# Below ~128, ambiguous queries start to mis-rank; above 256, quality
# gains are marginal for our scale.
DEFAULT_EMBED_DIM = 256


def build_lmstudio_embedder(
    endpoint: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
) -> Callable[[str], list[float]]:
    """Return a (text) -> list[float] callable backed by an OpenAI-compatible
    /v1/embeddings endpoint (LM Studio, llama.cpp server, etc.).

    The returned function caches the per-text vector in a small dict, so
    adding the same item twice does not pay for a second embedding call.
    Caching is the caller's responsibility to clear when the corpus changes
    meaningfully (for our usage, the cache is per-process and we never
    re-embed the same item).
    """
    endpoint = endpoint or os.environ.get("PI_EMBED_ENDPOINT", DEFAULT_EMBED_ENDPOINT)
    model = model or os.environ.get("PI_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    timeout = timeout or DEFAULT_TIMEOUT
    cache: dict[str, list[float]] = {}

    def embed(text: str) -> list[float]:
        if text in cache:
            return cache[text]
        body = json.dumps({"model": model, "input": text}).encode("utf-8")
        req = urllib.request.Request(endpoint, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"embeddings endpoint no responde en {endpoint}: {e}") from e
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"embeddings {e.code} {e.reason} on {endpoint}: {err_body[:300]}"
            ) from e
        vec = payload["data"][0]["embedding"]
        cache[text] = list(vec)
        return cache[text]

    return embed


def _resolve_embed_dim(embed_dim: int | None) -> int | None:
    """Resolution order: explicit arg > PI_EMBED_DIM env > DEFAULT_EMBED_DIM."""
    if embed_dim is not None:
        return embed_dim
    env = os.environ.get("PI_EMBED_DIM", "").strip()
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    return DEFAULT_EMBED_DIM


def build_lmstudio_retriever(
    endpoint: str | None = None,
    model: str | None = None,
    timeout: int | None = None,
    embed_dim: int | None = None,
) -> EmbeddingRetriever:
    """Convenience: return an EmbeddingRetriever wired to the local server.

    `embed_dim`: Matryoshka truncation dim (see EmbeddingRetriever).
    Resolved by: explicit arg > PI_EMBED_DIM env > DEFAULT_EMBED_DIM (256).
    """
    embed = build_lmstudio_embedder(endpoint=endpoint, model=model, timeout=timeout)
    return EmbeddingRetriever(embed, embed_dim=_resolve_embed_dim(embed_dim))
