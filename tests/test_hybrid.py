"""Tests for retrieval/hybrid.py — all model/DB calls are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from retrieval.dense import DenseResult
from retrieval.hybrid import HybridResult, reciprocal_rank_fusion, rerank
from retrieval.sparse import SparseResult


# ── Helpers ───────────────────────────────────────────────────────────────────


def _meta(slug: str = "beyond_good_and_evil", aph: int = 1) -> dict:
    return {
        "work_title": "Beyond Good and Evil",
        "work_slug": slug,
        "work_period": "late",
        "section_number": 1,
        "aphorism_number": aph,
        "aphorism_number_end": aph,
        "chunk_index": aph - 1,
        "chunk_type": "aphorism",
    }


def _dense(id: str, doc: str = "doc", aph: int = 1) -> DenseResult:
    return DenseResult(id=id, document=doc, metadata=_meta(aph=aph), distance=0.5)


def _sparse(id: str, doc: str = "doc", aph: int = 1, score: float = 1.0) -> SparseResult:
    return SparseResult(id=id, document=doc, metadata=_meta(aph=aph), score=score)


# ── HybridResult ──────────────────────────────────────────────────────────────


class TestHybridResult:
    def test_attributes(self) -> None:
        r = HybridResult(id="x", document="d", metadata=_meta(), rrf_score=0.5)
        assert r.id == "x"
        assert r.rrf_score == 0.5
        assert r.rerank_score is None

    def test_rerank_score_settable(self) -> None:
        r = HybridResult(id="x", document="d", metadata=_meta(), rrf_score=0.5)
        r.rerank_score = 3.14
        assert r.rerank_score == 3.14

    def test_repr_no_rerank(self) -> None:
        r = HybridResult(id="bge_chunk_0", document="d", metadata=_meta(), rrf_score=0.1)
        assert "bge_chunk_0" in repr(r)
        assert "rrf=" in repr(r)
        assert "rerank=" not in repr(r)

    def test_repr_with_rerank(self) -> None:
        r = HybridResult(id="x", document="d", metadata=_meta(), rrf_score=0.1, rerank_score=2.5)
        assert "rerank=" in repr(r)


# ── reciprocal_rank_fusion ────────────────────────────────────────────────────


class TestRRF:
    def test_rrf_scores_sum_correctly(self) -> None:
        """RRF score = sum of 1/(k+rank) across lists (1-indexed), k=60."""
        k = 60
        dense = [_dense("a"), _dense("b")]
        sparse = [_sparse("b"), _sparse("c")]
        results = reciprocal_rank_fusion(dense, sparse, k=k)

        by_id = {r.id: r.rrf_score for r in results}
        # "a" appears only in dense at rank 1
        assert abs(by_id["a"] - 1 / (k + 1)) < 1e-9
        # "c" appears only in sparse at rank 2
        assert abs(by_id["c"] - 1 / (k + 2)) < 1e-9
        # "b" appears in dense rank 2 AND sparse rank 1 → sum of both
        expected_b = 1 / (k + 2) + 1 / (k + 1)
        assert abs(by_id["b"] - expected_b) < 1e-9

    def test_rrf_deduplicates_results(self) -> None:
        """A chunk appearing in both lists must appear exactly once."""
        dense = [_dense("shared"), _dense("dense_only")]
        sparse = [_sparse("shared"), _sparse("sparse_only")]
        results = reciprocal_rank_fusion(dense, sparse)
        ids = [r.id for r in results]
        assert ids.count("shared") == 1
        assert len(ids) == 3  # shared + dense_only + sparse_only

    def test_rrf_ordering_is_descending(self) -> None:
        """Merged results must be sorted by RRF score highest first."""
        # Make "b" appear in both lists so it gets a higher combined score
        dense = [_dense("b"), _dense("a")]
        sparse = [_sparse("b"), _sparse("c")]
        results = reciprocal_rank_fusion(dense, sparse)
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_top_scorer_appears_in_both_lists(self) -> None:
        """The top result should be the doc that benefits from both lists."""
        dense = [_dense("a"), _dense("shared")]
        sparse = [_sparse("shared"), _sparse("b")]
        results = reciprocal_rank_fusion(dense, sparse)
        assert results[0].id == "shared"

    def test_empty_dense(self) -> None:
        sparse = [_sparse("x"), _sparse("y")]
        results = reciprocal_rank_fusion([], sparse)
        assert [r.id for r in results] == ["x", "y"]

    def test_empty_sparse(self) -> None:
        dense = [_dense("a"), _dense("b")]
        results = reciprocal_rank_fusion(dense, [])
        assert [r.id for r in results] == ["a", "b"]

    def test_both_empty(self) -> None:
        assert reciprocal_rank_fusion([], []) == []

    def test_returns_hybrid_results(self) -> None:
        results = reciprocal_rank_fusion([_dense("a")], [_sparse("b")])
        assert all(isinstance(r, HybridResult) for r in results)

    def test_document_and_metadata_preserved(self) -> None:
        dense = [_dense("a", doc="dense doc", aph=5)]
        results = reciprocal_rank_fusion(dense, [])
        assert results[0].document == "dense doc"
        assert results[0].metadata["aphorism_number"] == 5

    def test_custom_k(self) -> None:
        """Verify manual calculation with a non-default k."""
        k = 10
        dense = [_dense("x")]
        results = reciprocal_rank_fusion(dense, [], k=k)
        assert abs(results[0].rrf_score - 1 / (k + 1)) < 1e-9

    def test_many_results_all_unique(self) -> None:
        dense = [_dense(f"d{i}") for i in range(10)]
        sparse = [_sparse(f"s{i}") for i in range(10)]
        results = reciprocal_rank_fusion(dense, sparse)
        ids = [r.id for r in results]
        assert len(ids) == len(set(ids)) == 20


# ── rerank ────────────────────────────────────────────────────────────────────


class TestRerank:
    def _hybrid_results(self, n: int) -> list[HybridResult]:
        return [
            HybridResult(id=f"chunk_{i}", document=f"doc {i}",
                         metadata=_meta(aph=i + 1), rrf_score=1.0 / (i + 1))
            for i in range(n)
        ]

    def _mock_cross_encoder(self, scores: list[float]) -> MagicMock:
        import numpy as np
        model = MagicMock()
        model.predict = MagicMock(return_value=np.array(scores))
        return model

    def test_reranker_reduces_to_top_n(self) -> None:
        """After reranking 10 results, output length == RERANK_TOP_N (5)."""
        results = self._hybrid_results(10)
        mock_model = self._mock_cross_encoder([float(i) for i in range(10)])

        with patch("retrieval.hybrid.CrossEncoder", return_value=mock_model):
            ranked = rerank("query", results, top_n=5)

        assert len(ranked) == 5

    def test_reranker_returns_hybrid_results(self) -> None:
        results = self._hybrid_results(4)
        mock_model = self._mock_cross_encoder([1.0, 3.0, 2.0, 0.5])

        with patch("retrieval.hybrid.CrossEncoder", return_value=mock_model):
            ranked = rerank("query", results, top_n=4)

        assert all(isinstance(r, HybridResult) for r in ranked)

    def test_reranker_orders_by_score_descending(self) -> None:
        results = self._hybrid_results(4)
        # scores: chunk_2 best, then chunk_0, chunk_3, chunk_1
        mock_model = self._mock_cross_encoder([2.0, 0.1, 3.5, 0.5])

        with patch("retrieval.hybrid.CrossEncoder", return_value=mock_model):
            ranked = rerank("query", results, top_n=4)

        assert ranked[0].id == "chunk_2"
        assert ranked[1].id == "chunk_0"

    def test_reranker_populates_rerank_score(self) -> None:
        results = self._hybrid_results(3)
        mock_model = self._mock_cross_encoder([1.1, 2.2, 3.3])

        with patch("retrieval.hybrid.CrossEncoder", return_value=mock_model):
            ranked = rerank("query", results, top_n=3)

        assert all(r.rerank_score is not None for r in ranked)

    def test_reranker_passes_correct_pairs_to_model(self) -> None:
        results = self._hybrid_results(2)
        mock_model = self._mock_cross_encoder([1.0, 2.0])

        with patch("retrieval.hybrid.CrossEncoder", return_value=mock_model):
            rerank("my query", results, top_n=2)

        pairs = mock_model.predict.call_args[0][0]
        assert pairs == [("my query", "doc 0"), ("my query", "doc 1")]

    def test_reranker_model_name_forwarded(self) -> None:
        results = self._hybrid_results(2)
        mock_model = self._mock_cross_encoder([1.0, 2.0])

        with patch("retrieval.hybrid.CrossEncoder", return_value=mock_model) as mock_cls:
            rerank("q", results, top_n=1, model_name="custom/reranker")

        mock_cls.assert_called_once_with("custom/reranker")

    def test_empty_results_returns_empty(self) -> None:
        with patch("retrieval.hybrid.CrossEncoder"):
            assert rerank("query", [], top_n=5) == []

    def test_top_n_larger_than_input_returns_all(self) -> None:
        results = self._hybrid_results(3)
        mock_model = self._mock_cross_encoder([1.0, 2.0, 3.0])

        with patch("retrieval.hybrid.CrossEncoder", return_value=mock_model):
            ranked = rerank("q", results, top_n=100)

        assert len(ranked) == 3
