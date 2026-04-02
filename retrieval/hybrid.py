"""Hybrid retrieval: RRF merge of dense + sparse results, then cross-encoder reranking."""

from __future__ import annotations

from dataclasses import dataclass, field

from sentence_transformers import CrossEncoder, SentenceTransformer

import config
from retrieval.dense import DenseResult, dense_search
from retrieval.sparse import BM25Index, SparseResult


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class HybridResult:
    """A single result after RRF merge (and optionally reranking)."""

    id: str
    document: str
    metadata: dict
    rrf_score: float
    rerank_score: float | None = field(default=None)

    def __repr__(self) -> str:
        slug = self.metadata.get("work_slug", "?")
        aph = self.metadata.get("aphorism_number", "?")
        rerank = f", rerank={self.rerank_score:.4f}" if self.rerank_score is not None else ""
        return (
            f"HybridResult(id={self.id!r}, work={slug!r}, §{aph}, "
            f"rrf={self.rrf_score:.4f}{rerank})"
        )


# ── RRF merge ─────────────────────────────────────────────────────────────────


def reciprocal_rank_fusion(
    dense_results: list[DenseResult],
    sparse_results: list[SparseResult],
    k: int = config.RRF_K,
) -> list[HybridResult]:
    """Merge dense and sparse results using Reciprocal Rank Fusion.

    For each document, its RRF score is the sum of ``1 / (k + rank)`` across
    every result list it appears in (1-indexed rank).  Documents that appear
    in both lists get contributions from both, naturally surfacing results
    that are strong in at least one modality.

    Args:
        dense_results: Ranked list from :func:`dense_search` (closest first).
        sparse_results: Ranked list from :class:`BM25Index` (highest score first).
        k: RRF smoothing constant (default 60, per the original paper).

    Returns:
        List of :class:`HybridResult` sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    # id → (document, metadata) for reconstruction
    docs: dict[str, tuple[str, dict]] = {}

    for rank, r in enumerate(dense_results, start=1):
        scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (k + rank)
        docs[r.id] = (r.document, r.metadata)

    for rank, r in enumerate(sparse_results, start=1):
        scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (k + rank)
        docs[r.id] = (r.document, r.metadata)

    return [
        HybridResult(
            id=doc_id,
            document=docs[doc_id][0],
            metadata=docs[doc_id][1],
            rrf_score=score,
        )
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]


# ── Cross-encoder reranking ───────────────────────────────────────────────────


def rerank(
    query: str,
    results: list[HybridResult],
    top_n: int = config.RERANK_TOP_N,
    model_name: str = config.RERANKER_MODEL,
    model: CrossEncoder | None = None,
) -> list[HybridResult]:
    """Rerank *results* with a cross-encoder and return the top *top_n*.

    The cross-encoder scores each (query, document) pair jointly, which is
    more accurate than embedding-based similarity but more expensive.  We only
    apply it to the small merged candidate set from RRF.

    Args:
        query: The user's natural-language question.
        results: Candidate list from :func:`reciprocal_rank_fusion`.
        top_n: Number of results to return after reranking.
        model_name: HuggingFace cross-encoder model identifier.
        model: Optional pre-loaded :class:`CrossEncoder` instance.  If
            provided, *model_name* is ignored.

    Returns:
        Top *top_n* :class:`HybridResult` objects sorted by descending
        rerank score, with ``rerank_score`` populated.
    """
    if not results:
        return []

    if model is None:
        model = CrossEncoder(model_name)
    pairs = [(query, r.document) for r in results]
    scores: list[float] = model.predict(pairs).tolist()

    for result, score in zip(results, scores):
        result.rerank_score = score

    ranked = sorted(results, key=lambda r: r.rerank_score, reverse=True)  # type: ignore[arg-type]
    return ranked[:top_n]


# ── End-to-end hybrid search ──────────────────────────────────────────────────


def hybrid_search(
    query: str,
    *,
    embedding_model: str = config.EMBEDDING_MODEL,
    reranker_model: str = config.RERANKER_MODEL,
    dense_top_k: int = config.DENSE_TOP_K,
    sparse_top_k: int = config.SPARSE_TOP_K,
    rrf_k: int = config.RRF_K,
    top_n: int = config.RERANK_TOP_N,
    filter_period: str | None = None,
    filter_slug: str | None = None,
    bm25_index: BM25Index | None = None,
    sentence_transformer: SentenceTransformer | None = None,
    cross_encoder: CrossEncoder | None = None,
    use_hyde: bool = False,
    hyde_model: str = config.HYDE_MODEL,
) -> list[HybridResult]:
    """Run full hybrid retrieval pipeline for *query*.

    Pipeline:
    1. (Optional) HyDE: generate a hypothetical Nietzsche passage and use its
       embedding instead of the raw question embedding for dense search.
    2. Dense search via the configured vector store (``dense_top_k`` results).
    3. BM25 sparse search over all corpus documents (``sparse_top_k`` results).
    4. RRF merge of both lists.
    5. Cross-encoder reranking → top ``top_n`` results.

    Args:
        query: Natural-language question.
        embedding_model: SentenceTransformer model for dense search.
        reranker_model: Cross-encoder model for reranking.
        dense_top_k: Number of dense candidates.
        sparse_top_k: Number of sparse candidates.
        rrf_k: RRF smoothing constant.
        top_n: Number of final results after reranking.
        filter_period: Optional period filter for dense search (e.g. ``"late"``).
        filter_slug: Optional work slug filter for dense search.
        bm25_index: Pre-built :class:`BM25Index`.  If *None*, all documents
            are fetched from the vector store and a fresh index is built.
        sentence_transformer: Optional pre-loaded :class:`SentenceTransformer`.
        cross_encoder: Optional pre-loaded :class:`CrossEncoder`.
        use_hyde: If *True*, generate a hypothetical Nietzsche passage via
            Claude and embed it for dense search instead of the raw question.
            BM25 and the reranker still use the original *query*.
        hyde_model: Claude model used for HyDE generation (default: haiku).

    Returns:
        Top *top_n* :class:`HybridResult` objects sorted by rerank score.
    """
    # ── 0. HyDE (optional) ────────────────────────────────────────────────────
    embed_text: str | None = None
    if use_hyde:
        from retrieval.hyde import generate_hypothetical_passage
        embed_text = generate_hypothetical_passage(query, model=hyde_model)

    # ── 1. Dense search ───────────────────────────────────────────────────────
    dense_results = dense_search(
        query,
        embed_text=embed_text,
        model_name=embedding_model,
        model=sentence_transformer,
        top_k=dense_top_k,
        filter_period=filter_period,
        filter_slug=filter_slug,
    )

    # ── 2. Sparse search ─────────────────────────────────────────────────────
    if bm25_index is None:
        from retrieval.store import get_vector_store
        data = get_vector_store().get_all_documents()
        bm25_index = BM25Index(
            ids=data["ids"],
            documents=data["documents"],
            metadatas=data["metadatas"],
        )

    sparse_results = bm25_index.search(query, top_k=sparse_top_k)

    # ── 3. RRF merge ─────────────────────────────────────────────────────────
    merged = reciprocal_rank_fusion(dense_results, sparse_results, k=rrf_k)

    # ── 4. Rerank ─────────────────────────────────────────────────────────────
    return rerank(query, merged, top_n=top_n, model_name=reranker_model, model=cross_encoder)
