"""Sparse (BM25 keyword) retrieval over a corpus of document chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from rank_bm25 import BM25Okapi

import config


# ── Tokenisation ──────────────────────────────────────────────────────────────

_SPLIT_RE = re.compile(r"[^a-zA-Z0-9\u00C0-\u024F']+")


def _tokenise(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric characters, drop empty tokens.

    Retains accented characters so Nietzsche's coined terms (e.g. *Ressentiment*,
    *Übermensch*) survive tokenisation.
    """
    return [t for t in _SPLIT_RE.split(text.lower()) if t]


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SparseResult:
    """A single BM25 retrieval result."""

    id: str
    document: str
    metadata: dict
    score: float

    def __repr__(self) -> str:
        slug = self.metadata.get("work_slug", "?")
        aph = self.metadata.get("aphorism_number", "?")
        return (
            f"SparseResult(id={self.id!r}, work={slug!r}, §{aph}, "
            f"score={self.score:.4f})"
        )


# ── BM25 index ────────────────────────────────────────────────────────────────


class BM25Index:
    """An in-memory BM25 index over a fixed corpus of document chunks.

    Build once, query many times.  The index stores the original documents so
    results can be returned as :class:`SparseResult` objects without a
    secondary lookup.

    Args:
        ids: Stable document identifiers (e.g. ``"bge_chunk_0"``).
        documents: Raw text for each document.
        metadatas: Metadata dicts for each document (must be ChromaDB-safe).
    """

    def __init__(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        if not (len(ids) == len(documents) == len(metadatas)):
            raise ValueError(
                f"ids, documents, and metadatas must have the same length "
                f"({len(ids)}, {len(documents)}, {len(metadatas)})"
            )
        self._ids = list(ids)
        self._documents = list(documents)
        self._metadatas = list(metadatas)
        # BM25Okapi raises ZeroDivisionError on empty corpus
        if self._ids:
            tokenised = [_tokenise(doc) for doc in documents]
            self._bm25: BM25Okapi | None = BM25Okapi(tokenised)
        else:
            self._bm25 = None

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def corpus_size(self) -> int:
        """Number of documents in the index."""
        return len(self._ids)

    # ── search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = config.SPARSE_TOP_K,
    ) -> list[SparseResult]:
        """Return the *top_k* highest-scoring documents for *query*.

        Args:
            query: Natural-language question or keyword string.
            top_k: Maximum number of results to return.

        Returns:
            List of :class:`SparseResult` sorted by descending BM25 score.
            If all scores are zero (no term overlap), an empty list is returned.
        """
        if self._bm25 is None:
            return []

        query_tokens = _tokenise(query)
        if not query_tokens:
            return []

        scores: list[float] = self._bm25.get_scores(query_tokens).tolist()

        # Filter out documents with zero score (no term overlap).
        # Note: BM25 IDF can be negative in tiny corpora, so we keep any
        # non-zero score rather than requiring strictly positive values.
        ranked = sorted(
            ((s, i) for i, s in enumerate(scores) if s != 0.0),
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            SparseResult(
                id=self._ids[i],
                document=self._documents[i],
                metadata=self._metadatas[i],
                score=s,
            )
            for s, i in ranked[:top_k]
        ]


# ── Convenience function ──────────────────────────────────────────────────────


def sparse_search(
    query: str,
    ids: Sequence[str],
    documents: Sequence[str],
    metadatas: Sequence[dict],
    top_k: int = config.SPARSE_TOP_K,
) -> list[SparseResult]:
    """One-shot BM25 search: build an index from *documents* and query it.

    Suitable for ad-hoc use.  When querying the same corpus repeatedly, prefer
    instantiating :class:`BM25Index` directly to avoid rebuilding.

    Args:
        query: Natural-language question or keyword string.
        ids: Stable document identifiers.
        documents: Raw text for each document.
        metadatas: Metadata dicts for each document.
        top_k: Maximum number of results to return.

    Returns:
        List of :class:`SparseResult` sorted by descending BM25 score.
    """
    if not ids:
        return []
    index = BM25Index(ids=ids, documents=documents, metadatas=metadatas)
    return index.search(query, top_k=top_k)
