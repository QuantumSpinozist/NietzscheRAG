"""Dense (semantic) retrieval via ChromaDB vector search."""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb import Collection
from sentence_transformers import SentenceTransformer

import config
from ingest.embed import get_chroma_collection

from rich.console import Console

console = Console(stderr=True)


# ── Result type ───────────────────────────────────────────────────────────────


class DenseResult:
    """A single retrieval result with document text, metadata, and distance."""

    __slots__ = ("id", "document", "metadata", "distance")

    def __init__(
        self,
        id: str,
        document: str,
        metadata: dict,
        distance: float,
    ) -> None:
        self.id = id
        self.document = document
        self.metadata = metadata
        self.distance = distance

    def __repr__(self) -> str:
        slug = self.metadata.get("work_slug", "?")
        aph = self.metadata.get("aphorism_number", "?")
        return (
            f"DenseResult(id={self.id!r}, work={slug!r}, §{aph}, "
            f"dist={self.distance:.4f})"
        )


# ── Core retrieval ────────────────────────────────────────────────────────────


def dense_search(
    query: str,
    *,
    persist_dir: Path = config.CHROMA_PERSIST_DIR,
    collection_name: str = config.COLLECTION_NAME,
    model_name: str = config.EMBEDDING_MODEL,
    top_k: int = config.DENSE_TOP_K,
    where: dict | None = None,
) -> list[DenseResult]:
    """Embed *query* and return the *top_k* nearest chunks from ChromaDB.

    Args:
        query: The user's natural-language question.
        persist_dir: ChromaDB persistence directory.
        collection_name: Collection to search.
        model_name: SentenceTransformer model used to embed the query.
        top_k: Number of results to return.
        where: Optional ChromaDB metadata filter dict, e.g.
               ``{"work_period": "late"}`` or ``{"work_slug": "beyond_good_and_evil"}``.

    Returns:
        List of :class:`DenseResult` objects sorted by ascending distance
        (closest first).
    """
    collection = get_chroma_collection(persist_dir, collection_name)
    model = SentenceTransformer(model_name)

    query_embedding: list[float] = model.encode(
        [query], show_progress_bar=False, convert_to_numpy=True
    ).tolist()[0]

    kwargs: dict = dict(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    raw = collection.query(**kwargs)

    # ChromaDB returns lists-of-lists (one per query embedding)
    ids = raw["ids"][0]
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    return [
        DenseResult(id=i, document=d, metadata=m, distance=dist)
        for i, d, m, dist in zip(ids, documents, metadatas, distances)
    ]
