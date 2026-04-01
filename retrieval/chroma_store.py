"""ChromaDB implementation of VectorStore (local development backend)."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb import Collection

import config
from retrieval.store import VectorStore


def get_chroma_collection(
    persist_dir: Path = config.CHROMA_PERSIST_DIR,
    collection_name: str = config.COLLECTION_NAME,
) -> Collection:
    """Return (or create) a persistent ChromaDB collection.

    Args:
        persist_dir: Directory where ChromaDB stores its data files.
        collection_name: Name of the collection to get or create.

    Returns:
        A ChromaDB :class:`Collection` object.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(collection_name)


def _build_where(filter_period: str | None, filter_slug: str | None) -> dict | None:
    """Convert period/slug filters to a ChromaDB ``where`` dict."""
    conditions = []
    if filter_period:
        conditions.append({"work_period": filter_period})
    if filter_slug:
        conditions.append({"work_slug": filter_slug})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


class ChromaStore(VectorStore):
    """VectorStore backed by a local ChromaDB persistent collection."""

    def __init__(
        self,
        persist_dir: Path = config.CHROMA_PERSIST_DIR,
        collection_name: str = config.COLLECTION_NAME,
    ) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._collection = get_chroma_collection(persist_dir, collection_name)

    def store_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """Upsert chunks and their embeddings into ChromaDB."""
        self._collection.upsert(
            ids=[c["id"] for c in chunks],
            embeddings=embeddings,
            documents=[c["content"] for c in chunks],
            metadatas=[{k: v for k, v in c.items() if k not in ("id", "content")} for c in chunks],
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_period: str | None = None,
        filter_slug: str | None = None,
    ) -> list[dict]:
        """Return the *top_k* most similar chunks from ChromaDB."""
        kwargs: dict = dict(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        where = _build_where(filter_period, filter_slug)
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)
        ids = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        return [
            {"id": i, "document": d, "metadata": m, "distance": dist}
            for i, d, m, dist in zip(ids, documents, metadatas, distances)
        ]

    def get_all_documents(self) -> dict:
        """Return all stored documents as a dict with ids, documents, metadatas."""
        return self._collection.get(include=["documents", "metadatas"])

    def delete_all(self) -> None:
        """Delete the entire ChromaDB collection."""
        client = chromadb.PersistentClient(path=str(self._persist_dir))
        client.delete_collection(self._collection_name)
