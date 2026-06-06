# docs/

The RAG knowledge base. Markdown files in this directory are
auto-loaded by `demo_live.py` and any chat loop that wires
`runtime.soft.rag.load_directory` against this path.

The shipped files (Madrid, Barcelona, Valencia) are demo content
about Spanish cities. Replace them with your own.

## Format

- **One `.md` file per topic.** Filenames become the `name` shown
  in `[source: <name>, score=…]` provenance headers in the prompt.
- **Plain text or markdown.** No frontmatter required. The chunker
  splits at paragraph / sentence boundaries within a 200-token
  window (configurable via `RAGIndex.add_documents(max_tokens=…)`).
- **Languages are fine.** The default embedder
  (`embeddinggemma-300m-qat`) is multilingual, so docs in Spanish +
  English + others share the same semantic space.

## How a query becomes context

1. The user types a message.
2. The chat loop calls `RAGIndex.render_for_prompt(user_input, k=4)`.
3. The retriever returns the top-4 chunks by cosine similarity
   (× time-decay, if a `DecayingRetriever` is in use).
4. The render joins them with `\n\n---\n\n` and prepends each with
   `[source: <name>, score=0.832]`.
5. The render is dropped into the `tool_results` slot as
   `{"tool": "rag_search", "ok": true, "data": [{"query": ..., "context": ...}]}`.
6. The LLM sees this as a search-tool response and answers using the
   chunks, citing the source by name.

## Adding a new file

Drop a `.md` in this directory. The next time the demo or chat
loop starts, it gets indexed automatically.

For real production use, consider:

- **Chunking larger docs.** A 5000-word document with `max_tokens=200`
  yields ~25 chunks. Tune `max_tokens` to balance recall vs noise.
- **Re-indexing.** The index lives in process memory only. Restart
  the demo to pick up new files. (Persistence to disk is on the
  roadmap.)
- **Multi-corpus.** For separate indexes (docs, user-uploads,
  history), build multiple `RAGIndex` instances and pass them to
  the chat loop in priority order.
