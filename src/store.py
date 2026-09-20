from __future__ import annotations

from typing import Any, Callable

from .chunking import compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    In-memory only: records live in `self._store`. (The ChromaDB branch was dropped on
    purpose — no test needs it, and a half-initialised Chroma path would silently take
    over on any machine that happens to have `chromadb` installed.)
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # Copy the caller's metadata; `doc_id` always points at the *source file*, so that
        # chunks "file#0", "file#1"... can all be deleted through delete_document("file").
        metadata = dict(doc.metadata)
        metadata.setdefault("doc_id", doc.id)
        self._next_index += 1
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content),
            "index": self._next_index,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not records or top_k <= 0:
            return []
        query_vec = self._embedding_fn(query)
        scored = [(compute_similarity(query_vec, r["embedding"]), r) for r in records]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        # Drop the raw vector from results: it is noise when printed and never needed downstream.
        return [
            {"id": r["id"], "content": r["content"], "metadata": r["metadata"], "score": score}
            for score, r in scored[:top_k]
        ]

    @staticmethod
    def _matches(metadata: dict, metadata_filter: dict) -> bool:
        for key, expected in metadata_filter.items():
            value = metadata.get(key)
            if isinstance(expected, (list, tuple, set)):
                if value not in expected:
                    return False
            elif value != expected:
                return False
        return True

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return
        # Optional hook: a caching embedder can fetch every vector in one batched API call
        # here, so the per-document calls in _make_record become cache hits.
        prefetch = getattr(self._embedding_fn, "prefetch", None)
        if callable(prefetch):
            prefetch([doc.content for doc in docs])
        self._store.extend(self._make_record(doc) for doc in docs)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # Filter FIRST, then rank: filtering after top-k could leave 0 results even though
        # valid chunks exist, because wrong-audience chunks already took the k slots.
        candidates = self._store if not metadata_filter else [
            r for r in self._store if self._matches(r["metadata"], metadata_filter)
        ]
        return self._search_records(query, candidates, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        before = len(self._store)
        self._store = [r for r in self._store if r["metadata"].get("doc_id") != doc_id]
        return len(self._store) < before
