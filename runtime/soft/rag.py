"""RAG: document ingestion, chunking, retrieval.

The agent''s hard/soft split: this module is soft (it talks to the LLM
indirectly via the retriever) but does not call the LLM itself. It
sits between the filesystem and the retriever.

Design:
  Document    = a single source file (or any string with metadata)
  Chunk       = a slice of a Document, with provenance
  RAGIndex    = a thin wrapper around a Retriever that holds Chunk metadata
                so the chat loop can render "{filename}: ..." citations

The retriever interface (search(query, k) -> [(item, score)]) is the
only contract RAGIndex needs from the retriever. Same as
EmbeddingRetriever, HashingRetriever, or a custom implementation.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

from runtime.hard.budget import estimate_tokens, truncate_to_tokens


@dataclass
class Document:
    """A source. May be a file, a URL, an inline string."""
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A slice of a Document. Carries the source for citation."""
    id: str                 # doc_id + ":" + offset, unique within the index
    doc_id: str
    text: str
    offset: int             # char offset in the original document
    metadata: dict = field(default_factory=dict)


def chunk_document(doc: Document, max_tokens: int = 200,
                   overlap_tokens: int = 20) -> list[Chunk]:
    """Split a document into overlapping token-bounded chunks.

    The default `overlap_tokens` keeps a small bridge between chunks
    so a sentence that spans the boundary is still retrievable from
    one of the sides. Overlap=0 gives clean disjoint chunks.
    """
    text = doc.text
    if not text.strip():
        return []
    # Tokenize at the character level using the cheap approximation;
    # boundary is approximate but good enough for the demo.
    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4
    if max_chars <= 0 or len(text) <= max_chars:
        return [Chunk(id=doc.id + ":0", doc_id=doc.id, text=text,
                      offset=0, metadata=dict(doc.metadata))]

    out: list[Chunk] = []
    pos = 0
    while pos < len(text):
        end = min(pos + max_chars, len(text))
        # Try to end on a sentence boundary (period + whitespace) for
        # prettier chunks; otherwise end on the hard limit.
        if end < len(text):
            for sep in ("\n\n", ". ", ".\n", "\n"):
                idx = text.rfind(sep, pos + max_chars // 2, end)
                if idx > 0:
                    end = idx + len(sep)
                    break
        chunk_text = text[pos:end]
        out.append(Chunk(
            id=doc.id + ":" + str(pos),
            doc_id=doc.id,
            text=chunk_text,
            offset=pos,
            metadata=dict(doc.metadata),
        ))
        if end >= len(text):
            break
        pos = max(end - overlap_chars, pos + 1)
    return out


def load_text_file(path: Path) -> Document:
    """Load a plain-text or markdown file as a Document."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return Document(id=str(path), text=text, metadata={
        "path": str(path),
        "size": len(text),
        "name": path.name,
    })


def load_directory(directory: Path, glob: str = "**/*.md") -> list[Document]:
    """Load all matching files under a directory. Default: markdown only.

    Skips binary files and unreadable paths silently; the index is
    best-effort and the user can re-run on a cleaner tree if needed.
    """
    docs: list[Document] = []
    if not directory.exists():
        return docs
    for p in sorted(directory.glob(glob)):
        if p.is_file():
            try:
                docs.append(load_text_file(p))
            except (OSError, UnicodeDecodeError):
                continue
    return docs


class RetrieverLike(Protocol):
    """Subset of the Retriever interface that RAGIndex needs."""
    def search(self, query: str, k: int = 5) -> list[tuple]: ...


class RAGIndex:
    """Index of Chunks over a Retriever. Holds a parallel dict of metadata
    so the chat loop can render "<doc>: <chunk>" citations without
    re-reading the retriever''s internals.
    """

    def __init__(self, retriever: RetrieverLike, embed_dim_cap: int | None = None):
        self._retriever = retriever
        self._chunks: dict[str, Chunk] = {}
        self._embed_dim_cap = embed_dim_cap

    def add(self, chunk: Chunk) -> None:
        """Add a chunk to the index."""
        self._chunks[chunk.id] = chunk
        # Adapt the chunk to whatever shape the retriever expects.
        # EmbeddingRetriever takes MemoryItem (key, value). We use the
        # chunk id as key and chunk text as value.
        from runtime.soft.memory import MemoryItem
        self._retriever.add(MemoryItem(key=chunk.id, value=chunk.text))

    def remove(self, chunk_id: str) -> None:
        self._chunks.pop(chunk_id, None)
        self._retriever.remove(chunk_id)

    def add_documents(self, docs: Iterable[Document],
                      max_tokens: int = 200) -> int:
        """Chunk and index each document. Returns the number of chunks added."""
        n = 0
        for doc in docs:
            for chunk in chunk_document(doc, max_tokens=max_tokens):
                self.add(chunk)
                n += 1
        return n

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """Top-k (chunk, score) pairs. Score is whatever the retriever returns."""
        results = self._retriever.search(query, k=k)
        out: list[tuple[Chunk, float]] = []
        for item, score in results:
            chunk = self._chunks.get(item.key)
            if chunk is not None:
                out.append((chunk, score))
        return out

    def render_for_prompt(self, query: str, k: int = 5,
                          max_chars_per_chunk: int = 1500) -> str:
        """Format the top-k chunks as a single string the LLM can read.

        Includes a brief provenance header per chunk so the model
        (and the audit log) can attribute answers to sources.
        """
        results = self.search(query, k=k)
        if not results:
            return ""
        parts: list[str] = []
        for chunk, score in results:
            name = chunk.metadata.get("name", chunk.doc_id)
            text = chunk.text if len(chunk.text) <= max_chars_per_chunk else (
                chunk.text[: max_chars_per_chunk] + "...")
            parts.append("[source: " + str(name) + ", score=" + ("%.3f" % score) + "]\n" + text)
        return "\n\n---\n\n".join(parts)

    def __len__(self) -> int:
        return len(self._chunks)
