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
    aphorism_bonus: float = 0.0,
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
        aphorism_bonus: Additive bonus applied to chunks with
            ``chunk_type == "aphorism"`` after cross-encoder scoring.
            Helps specific aphorisms beat broad prose passages when their
            raw cross-encoder scores are close.  Set to 0.0 to disable.

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
        bonus = aphorism_bonus if result.metadata.get("chunk_type") == "aphorism" else 0.0
        result.rerank_score = score + bonus

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
    aphorism_bonus: float = config.APHORISM_RERANK_BONUS,
    use_multiquery: bool = False,
    multiquery_n: int = config.MULTIQUERY_N,
    multiquery_model: str = config.HYDE_MODEL,
) -> list[HybridResult]:
    """Run full hybrid retrieval pipeline for *query*.

    Pipeline:
    1. (Optional) HyDE: generate a hypothetical Nietzsche passage and use its
       embedding instead of the raw question embedding for dense search.
    2. (Optional) Multi-query expansion: generate *multiquery_n* paraphrases
       and run dense search for each; all dense result lists feed into RRF.
    3. Dense search via the configured vector store (``dense_top_k`` results).
    4. BM25 sparse search over all corpus documents (``sparse_top_k`` results).
    5. RRF merge of all dense lists + sparse list.
    6. Cross-encoder reranking → top ``top_n`` results.

    Args:
        query: Natural-language question.
        embedding_model: SentenceTransformer model for dense search.
        reranker_model: Cross-encoder model for reranking.
        dense_top_k: Number of dense candidates per query.
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
        aphorism_bonus: Additive score bonus applied to aphorism-type chunks
            after cross-encoder scoring (default: ``config.APHORISM_RERANK_BONUS``).
            Helps specific aphorisms beat broad prose passages.  Set to 0.0
            to disable.
        use_multiquery: If *True*, generate *multiquery_n* paraphrases of the
            query and run dense search for each, unioning all result lists
            before RRF.  Broadens the candidate pool for abstract queries.
        multiquery_n: Number of query paraphrases to generate (default:
            ``config.MULTIQUERY_N``).
        multiquery_model: Claude model for paraphrase generation.

    Returns:
        Top *top_n* :class:`HybridResult` objects sorted by rerank score.
    """
    # ── 0. HyDE (optional) ────────────────────────────────────────────────────
    embed_text: str | None = None
    if use_hyde:
        from retrieval.hyde import generate_hypothetical_passage
        embed_text = generate_hypothetical_passage(query, model=hyde_model)

    # ── 1. Dense search (original query + optional variants) ─────────────────
    dense_results_lists: list[list[DenseResult]] = []

    dense_results_lists.append(dense_search(
        query,
        embed_text=embed_text,
        model_name=embedding_model,
        model=sentence_transformer,
        top_k=dense_top_k,
        filter_period=filter_period,
        filter_slug=filter_slug,
    ))

    if use_multiquery:
        from retrieval.multiquery import generate_query_variants
        variants = generate_query_variants(query, n=multiquery_n, model=multiquery_model)
        for variant in variants:
            dense_results_lists.append(dense_search(
                variant,
                model_name=embedding_model,
                model=sentence_transformer,
                top_k=dense_top_k,
                filter_period=filter_period,
                filter_slug=filter_slug,
            ))

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

    # ── 3. RRF merge (all dense lists + sparse) ───────────────────────────────
    # Use the first dense list as the primary; merge in remaining lists
    # and the sparse results via successive RRF passes.
    merged = reciprocal_rank_fusion(dense_results_lists[0], sparse_results, k=rrf_k)
    for extra_dense in dense_results_lists[1:]:
        # Convert HybridResult list back to DenseResult-compatible list for RRF.
        # We treat the current merged list as the "dense" side and the new
        # variant results as additional dense candidates.
        extra_as_hybrid = [
            HybridResult(id=r.id, document=r.document, metadata=r.metadata, rrf_score=0.0)
            for r in extra_dense
        ]
        # Re-run RRF treating merged as dense and extra as sparse
        scores: dict[str, float] = {r.id: r.rrf_score for r in merged}
        docs: dict[str, tuple[str, dict]] = {r.id: (r.document, r.metadata) for r in merged}
        for rank, r in enumerate(extra_as_hybrid, start=1):
            scores[r.id] = scores.get(r.id, 0.0) + 1.0 / (rrf_k + rank)
            docs[r.id] = (r.document, r.metadata)
        merged = [
            HybridResult(id=doc_id, document=docs[doc_id][0], metadata=docs[doc_id][1], rrf_score=score)
            for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ]

    # ── 4. Rerank ─────────────────────────────────────────────────────────────
    return rerank(
        query, merged,
        top_n=top_n,
        model_name=reranker_model,
        model=cross_encoder,
        aphorism_bonus=aphorism_bonus,
    )
