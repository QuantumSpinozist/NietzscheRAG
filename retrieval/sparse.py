"""Sparse (BM25 keyword) retrieval over a corpus of document chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from rank_bm25 import BM25Okapi

import config


# ── Synonym / term-expansion table ────────────────────────────────────────────
#
# Maps any token (after lowercasing) to a list of additional tokens that should
# be injected alongside it.  This bridges the gap between the user's query
# vocabulary and Nietzsche's actual prose / Gutenberg translation choices.
#
# Design rules:
#   - Keys are tokens AS THEY APPEAR IN QUERIES (e.g. "overman").
#   - Values are tokens AS THEY APPEAR IN THE CORPUS (e.g. "übermensch").
#   - Expansion is symmetric: corpus tokens also map to query tokens so that
#     indexing adds the query-side synonyms and searching adds corpus-side ones.
#   - All strings are pre-lowercased.

_SYNONYMS: dict[str, list[str]] = {
    # Übermensch / overman / superman — Gutenberg translations vary
    "overman":          ["übermensch", "ubermensch", "superman"],
    "übermensch":       ["overman", "superman", "ubermensch"],
    "ubermensch":       ["übermensch", "overman", "superman"],
    "superman":         ["übermensch", "overman", "ubermensch"],

    # Eternal recurrence / eternal return — GS §341 title is "The Heaviest Burden"
    "recurrence":       ["return", "burden", "heaviest"],
    "return":           ["recurrence"],
    "eternal":          ["recurrence", "return", "heaviest"],

    # Ressentiment — French/German term vs. English "resentment"
    "resentment":       ["ressentiment"],
    "ressentiment":     ["resentment", "rancour", "rancor"],

    # Amor fati — Latin phrase vs. English paraphrase
    "amor":             ["fate", "necessity", "necessary", "beautiful"],
    "fati":             ["fate", "amor", "love"],
    "fate":             ["amor", "fati", "necessity"],

    # Revaluation / transvaluation — "Umwertung aller Werte"
    "revaluation":      ["transvaluation", "umwertung", "inversion"],
    "transvaluation":   ["revaluation", "umwertung"],

    # Dionysian / Apollonian — capitalization stripped by tokeniser
    "dionysian":        ["dionysus", "dionysos"],
    "dionysus":         ["dionysian"],
    "apollonian":       ["apollo", "apolline"],
    "apollo":           ["apollonian", "apolline"],

    # Will to power — common paraphrase vs. "Wille zur Macht"
    "macht":            ["power", "will"],
    "wille":            ["will", "power"],

    # Nihilism variants
    "nihilism":         ["nihilist", "nihilistic", "nothingness"],
    "nihilist":         ["nihilism", "nihilistic"],

    # Perspectivism
    "perspectivism":    ["perspective", "perspectives", "perspectival"],
    "perspectival":     ["perspectivism", "perspective"],

    # Pathos of distance
    "pathos":           ["distance", "noble", "rank"],

    # Ascetic ideal
    "ascetic":          ["asceticism", "priest", "ideal"],
    "asceticism":       ["ascetic"],
}


def _expand(tokens: list[str]) -> list[str]:
    """Append synonym tokens for any recognised token in *tokens*."""
    extra: list[str] = []
    for t in tokens:
        for syn in _SYNONYMS.get(t, []):
            if syn not in tokens:
                extra.append(syn)
    return tokens + extra


# ── Tokenisation ──────────────────────────────────────────────────────────────

_SPLIT_RE = re.compile(r"[^a-zA-Z0-9\u00C0-\u024F']+")


def _tokenise(text: str, use_synonyms: bool = False) -> list[str]:
    """Lowercase, split on non-alphanumeric characters, drop empty tokens.

    Retains accented characters so Nietzsche's coined terms (e.g. *Ressentiment*,
    *Übermensch*) survive tokenisation.  When *use_synonyms* is True, recognised
    terms are expanded with their synonym equivalents before returning.
    """
    tokens = [t for t in _SPLIT_RE.split(text.lower()) if t]
    return _expand(tokens) if use_synonyms else tokens


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
        use_synonyms: When True, recognised Nietzsche-specific terms are
            expanded with synonyms during both indexing and querying.
    """

    def __init__(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[dict],
        use_synonyms: bool = False,
    ) -> None:
        if not (len(ids) == len(documents) == len(metadatas)):
            raise ValueError(
                f"ids, documents, and metadatas must have the same length "
                f"({len(ids)}, {len(documents)}, {len(metadatas)})"
            )
        self._ids = list(ids)
        self._documents = list(documents)
        self._metadatas = list(metadatas)
        self._use_synonyms = use_synonyms
        # BM25Okapi raises ZeroDivisionError on empty corpus
        if self._ids:
            tokenised = [_tokenise(doc, use_synonyms=use_synonyms) for doc in documents]
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
        filter_period: str | None = None,
        filter_slug: str | None = None,
    ) -> list[SparseResult]:
        """Return the *top_k* highest-scoring documents for *query*.

        Args:
            query: Natural-language question or keyword string.
            top_k: Maximum number of results to return.
            filter_period: If set, only return chunks whose ``work_period``
                metadata matches (e.g. ``"late"``).
            filter_slug: If set, only return chunks whose ``work_slug``
                metadata matches (e.g. ``"beyond_good_and_evil"``).

        Returns:
            List of :class:`SparseResult` sorted by descending BM25 score.
            If all scores are zero (no term overlap), an empty list is returned.
        """
        if self._bm25 is None:
            return []

        query_tokens = _tokenise(query, use_synonyms=self._use_synonyms)
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

        results: list[SparseResult] = []
        for s, i in ranked:
            meta = self._metadatas[i]
            if filter_period and meta.get("work_period") != filter_period:
                continue
            if filter_slug and meta.get("work_slug") != filter_slug:
                continue
            results.append(
                SparseResult(
                    id=self._ids[i],
                    document=self._documents[i],
                    metadata=meta,
                    score=s,
                )
            )
            if len(results) >= top_k:
                break
        return results


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
