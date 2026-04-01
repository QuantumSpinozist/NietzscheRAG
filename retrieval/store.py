"""Abstract VectorStore interface and backend factory."""

from __future__ import annotations

from abc import ABC, abstractmethod


class VectorStore(ABC):
    """Backend-agnostic interface for storing and searching chunk embeddings."""

    @abstractmethod
    def store_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """Upsert *chunks* with their *embeddings*.

        Each dict in *chunks* must contain:
          - ``id``: unique string identifier (e.g. ``"bge_chunk_0"``)
          - ``content``: the chunk text
          - all metadata fields: work_title, work_slug, work_period,
            section_number, aphorism_number, aphorism_number_end,
            chunk_index, chunk_type
        """

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_period: str | None = None,
        filter_slug: str | None = None,
    ) -> list[dict]:
        """Return the *top_k* most similar chunks.

        Returns a list of dicts, each with keys:
          ``id``, ``document``, ``metadata`` (dict), ``distance`` (float).
        """

    @abstractmethod
    def get_all_documents(self) -> dict:
        """Return all stored documents.

        Returns a dict with keys ``ids``, ``documents``, ``metadatas``.
        Used to build the BM25 in-memory index.
        """

    @abstractmethod
    def delete_all(self) -> None:
        """Delete every document from the store."""


def get_vector_store() -> VectorStore:
    """Read ``VECTOR_STORE_BACKEND`` from the environment and return the matching implementation.

    Returns:
        :class:`SupabaseStore` when ``VECTOR_STORE_BACKEND=supabase``,
        :class:`ChromaStore` otherwise (default).
    """
    import os

    backend = os.getenv("VECTOR_STORE_BACKEND", "chroma")
    if backend == "supabase":
        from retrieval.supabase_store import SupabaseStore
        return SupabaseStore()
    from retrieval.chroma_store import ChromaStore
    return ChromaStore()
