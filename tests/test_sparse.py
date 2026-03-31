"""Tests for retrieval/sparse.py — pure Python, no external dependencies."""

from __future__ import annotations

import pytest

from retrieval.sparse import BM25Index, SparseResult, _tokenise, sparse_search


# ── Fixtures ──────────────────────────────────────────────────────────────────

_CORPUS_IDS = [
    "bge_chunk_0",
    "bge_chunk_1",
    "bge_chunk_2",
    "bge_chunk_3",
    "bge_chunk_4",
]

_CORPUS_DOCS = [
    "Supposing that Truth is a woman--what then?",
    "The will to power is the fundamental drive of life.",
    "Master morality and slave morality are opposed.",
    "The eternal recurrence: what if you had to live this life again?",
    "Dionysus versus the Crucified--that is my formula.",
]

_CORPUS_META = [
    {"work_slug": "beyond_good_and_evil", "work_period": "late",
     "section_number": 1, "aphorism_number": 1, "aphorism_number_end": 1,
     "chunk_index": 0, "chunk_type": "aphorism", "work_title": "BGE"},
    {"work_slug": "beyond_good_and_evil", "work_period": "late",
     "section_number": 1, "aphorism_number": 2, "aphorism_number_end": 2,
     "chunk_index": 1, "chunk_type": "aphorism", "work_title": "BGE"},
    {"work_slug": "beyond_good_and_evil", "work_period": "late",
     "section_number": 2, "aphorism_number": 3, "aphorism_number_end": 3,
     "chunk_index": 2, "chunk_type": "aphorism", "work_title": "BGE"},
    {"work_slug": "beyond_good_and_evil", "work_period": "late",
     "section_number": 3, "aphorism_number": 4, "aphorism_number_end": 4,
     "chunk_index": 3, "chunk_type": "aphorism", "work_title": "BGE"},
    {"work_slug": "beyond_good_and_evil", "work_period": "late",
     "section_number": 4, "aphorism_number": 5, "aphorism_number_end": 5,
     "chunk_index": 4, "chunk_type": "aphorism", "work_title": "BGE"},
]


def _index() -> BM25Index:
    return BM25Index(ids=_CORPUS_IDS, documents=_CORPUS_DOCS, metadatas=_CORPUS_META)


# ── _tokenise ─────────────────────────────────────────────────────────────────


class TestTokenise:
    def test_lowercases(self) -> None:
        assert _tokenise("Truth") == ["truth"]

    def test_splits_on_punctuation(self) -> None:
        assert _tokenise("a-b--c") == ["a", "b", "c"]

    def test_drops_empty_tokens(self) -> None:
        tokens = _tokenise("  hello   world  ")
        assert "" not in tokens
        assert tokens == ["hello", "world"]

    def test_preserves_accented_chars(self) -> None:
        tokens = _tokenise("Ressentiment Übermensch")
        assert "ressentiment" in tokens
        assert "übermensch" in tokens

    def test_empty_string(self) -> None:
        assert _tokenise("") == []

    def test_punctuation_only(self) -> None:
        assert _tokenise("---!!!") == []


# ── SparseResult ──────────────────────────────────────────────────────────────


class TestSparseResult:
    def test_attributes_stored(self) -> None:
        r = SparseResult(
            id="bge_chunk_0",
            document="Some text.",
            metadata=_CORPUS_META[0],
            score=3.14,
        )
        assert r.id == "bge_chunk_0"
        assert r.score == 3.14
        assert r.metadata["work_slug"] == "beyond_good_and_evil"

    def test_frozen(self) -> None:
        r = SparseResult(id="x", document="d", metadata={}, score=1.0)
        with pytest.raises(Exception):
            r.score = 9.9  # type: ignore[misc]

    def test_repr_contains_key_fields(self) -> None:
        r = SparseResult(id="bge_chunk_2", document="d", metadata=_CORPUS_META[2], score=2.5)
        rep = repr(r)
        assert "beyond_good_and_evil" in rep
        assert "2.5" in rep


# ── BM25Index construction ────────────────────────────────────────────────────


class TestBM25IndexConstruction:
    def test_corpus_size(self) -> None:
        assert _index().corpus_size == len(_CORPUS_IDS)

    def test_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            BM25Index(ids=["a", "b"], documents=["doc"], metadatas=[{}])

    def test_empty_corpus_allowed(self) -> None:
        idx = BM25Index(ids=[], documents=[], metadatas=[])
        assert idx.corpus_size == 0


# ── BM25Index.search ──────────────────────────────────────────────────────────


class TestBM25IndexSearch:
    def test_returns_list_of_sparse_results(self) -> None:
        results = _index().search("will to power")
        assert isinstance(results, list)
        assert all(isinstance(r, SparseResult) for r in results)

    def test_top_k_limits_results(self) -> None:
        results = _index().search("life morality power truth", top_k=2)
        assert len(results) <= 2

    def test_relevant_doc_ranks_first(self) -> None:
        """Query for 'will power' should surface chunk_1 above others."""
        results = _index().search("will power")
        assert len(results) > 0
        assert results[0].id == "bge_chunk_1"

    def test_scores_descending(self) -> None:
        results = _index().search("morality power life truth")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_scores_nonzero(self) -> None:
        """Every returned result had a non-zero BM25 score (term overlap exists)."""
        results = _index().search("will to power")
        assert all(r.score != 0 for r in results)

    def test_no_results_for_absent_term(self) -> None:
        """A query with no term overlap should return an empty list."""
        results = _index().search("xyzzy foobar quux")
        assert results == []

    def test_empty_query_returns_empty(self) -> None:
        results = _index().search("")
        assert results == []

    def test_empty_corpus_returns_empty(self) -> None:
        idx = BM25Index(ids=[], documents=[], metadatas=[])
        assert idx.search("anything") == []

    def test_result_ids_are_from_corpus(self) -> None:
        results = _index().search("morality slave master")
        for r in results:
            assert r.id in _CORPUS_IDS

    def test_result_documents_match_corpus(self) -> None:
        results = _index().search("eternal recurrence life")
        for r in results:
            idx = _CORPUS_IDS.index(r.id)
            assert r.document == _CORPUS_DOCS[idx]

    def test_result_metadata_matches_corpus(self) -> None:
        results = _index().search("dionysus crucified formula")
        for r in results:
            idx = _CORPUS_IDS.index(r.id)
            assert r.metadata == _CORPUS_META[idx]

    def test_query_hits_multiple_docs(self) -> None:
        """A broad query should return more than one result."""
        results = _index().search("life power truth morality recurrence")
        assert len(results) > 1

    def test_case_insensitive(self) -> None:
        lower = _index().search("will to power")
        upper = _index().search("WILL TO POWER")
        assert [r.id for r in lower] == [r.id for r in upper]

    def test_nietzsche_coined_term(self) -> None:
        """Accented terms like 'Ressentiment' survive tokenisation."""
        corpus_ids = ["a"]
        corpus_docs = ["The concept of Ressentiment is central to slave morality."]
        corpus_meta = [{"work_slug": "gm"}]
        idx = BM25Index(ids=corpus_ids, documents=corpus_docs, metadatas=corpus_meta)
        results = idx.search("Ressentiment")
        assert len(results) == 1
        assert results[0].id == "a"

    def test_top_k_larger_than_corpus(self) -> None:
        """Requesting more results than corpus size should not raise."""
        results = _index().search("morality truth", top_k=100)
        assert len(results) <= _index().corpus_size


# ── sparse_search convenience function ───────────────────────────────────────


class TestSparseSearch:
    def test_delegates_to_bm25_index(self) -> None:
        results = sparse_search(
            "will to power",
            ids=_CORPUS_IDS,
            documents=_CORPUS_DOCS,
            metadatas=_CORPUS_META,
            top_k=3,
        )
        assert isinstance(results, list)
        assert all(isinstance(r, SparseResult) for r in results)
        assert len(results) <= 3

    def test_top_result_is_relevant(self) -> None:
        results = sparse_search(
            "eternal recurrence life again",
            ids=_CORPUS_IDS,
            documents=_CORPUS_DOCS,
            metadatas=_CORPUS_META,
        )
        assert results[0].id == "bge_chunk_3"

    def test_empty_input_returns_empty(self) -> None:
        results = sparse_search("anything", ids=[], documents=[], metadatas=[])
        assert results == []
