"""POST /query — run the full RAG pipeline and return an answer with sources."""

from __future__ import annotations

from fastapi import APIRouter, Request

import config
from api.models import QueryRequest, QueryResponse, SourceResult
from generation.claude import generate_answer
from retrieval.hybrid import hybrid_search

router = APIRouter()


def run_pipeline(
    question: str,
    filter_period: str | None,
    filter_slug: str | None,
    state: object,
    use_hyde: bool = False,
) -> dict:
    """Execute hybrid retrieval + generation and return a serialisable result dict.

    Accepts pre-loaded models from ``app.state`` so the embedding model and
    cross-encoder are not reloaded on every request.

    Args:
        question: User's natural-language question.
        filter_period: Optional period filter (``"early"`` / ``"middle"`` / ``"late"``).
        filter_slug: Optional work slug filter.
        state: FastAPI ``app.state`` carrying ``sentence_transformer``,
            ``cross_encoder``, and ``bm25_index``.
        use_hyde: If *True*, generate a hypothetical Nietzsche passage before
            dense retrieval (HyDE query expansion).

    Returns:
        Dict with ``"answer"`` (str) and ``"sources"`` (list of dicts).
    """
    results = hybrid_search(
        question,
        filter_period=filter_period,
        filter_slug=filter_slug,
        bm25_index=getattr(state, "bm25_index", None),
        sentence_transformer=getattr(state, "sentence_transformer", None),
        cross_encoder=getattr(state, "cross_encoder", None),
        use_hyde=use_hyde,
    )

    answer = generate_answer(question, results)

    sources = [
        {
            "work_title": r.metadata.get("work_title", ""),
            "work_slug": r.metadata.get("work_slug", ""),
            "section_number": r.metadata.get("section_number") or None,
            "chunk_type": r.metadata.get("chunk_type", ""),
            "content": r.document,
            "similarity": r.rerank_score if r.rerank_score is not None else r.rrf_score,
        }
        for r in results
    ]

    return {"answer": answer, "sources": sources}


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest, request: Request) -> QueryResponse:
    """Ask a philosophical question and receive a grounded, cited answer.

    The pipeline runs hybrid retrieval (dense + BM25 + RRF + reranking) then
    passes the top passages to Claude for generation.
    """
    result = run_pipeline(
        body.question,
        body.filter_period,
        body.filter_slug,
        request.app.state,
        use_hyde=body.use_hyde,
    )
    return QueryResponse(**result)
