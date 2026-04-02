"""Dense (semantic) retrieval via the configured vector store."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

import config
from retrieval.store import get_vector_store

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
    embed_text: str | None = None,
    model_name: str = config.EMBEDDING_MODEL,
    model: SentenceTransformer | None = None,
    top_k: int = config.DENSE_TOP_K,
    filter_period: str | None = None,
    filter_slug: str | None = None,
) -> list[DenseResult]:
    """Embed *query* and return the *top_k* nearest chunks from the vector store.

    Args:
        query: The user's natural-language question.
        embed_text: Text to embed instead of *query*.  Pass a HyDE-generated
            hypothetical passage here to shift the embedding toward corpus
            language while keeping *query* available for other pipeline steps.
            When *None* (default), *query* itself is embedded.
        model_name: SentenceTransformer model used to embed the query.
        model: Optional pre-loaded :class:`SentenceTransformer` instance.  If
            provided, *model_name* is ignored and no model is loaded from disk.
        top_k: Number of results to return.
        filter_period: Optional period filter (e.g. ``"late"``).
        filter_slug: Optional work slug filter (e.g. ``"beyond_good_and_evil"``).

    Returns:
        List of :class:`DenseResult` objects sorted by ascending distance
        (closest first).
    """
    if model is None:
        model = SentenceTransformer(model_name)

    text_to_embed = embed_text if embed_text is not None else query
    query_embedding: list[float] = model.encode(
        [text_to_embed], show_progress_bar=False, convert_to_numpy=True
    ).tolist()[0]

    store = get_vector_store()
    raw = store.similarity_search(
        query_embedding,
        top_k=top_k,
        filter_period=filter_period,
        filter_slug=filter_slug,
    )

    return [
        DenseResult(id=r["id"], document=r["document"], metadata=r["metadata"], distance=r["distance"])
        for r in raw
    ]
