"""Tests for retrieval/dense.py — all model and ChromaDB calls are mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from retrieval.dense import DenseResult, dense_search


# ── Mock helpers ──────────────────────────────────────────────────────────────


def _mock_model(embedding_dim: int = 768) -> MagicMock:
    """Fake SentenceTransformer that returns a deterministic embedding."""
    model = MagicMock()
    model.encode = MagicMock(
        side_effect=lambda texts, **kw: MagicMock(
            tolist=lambda: [[0.1] * embedding_dim for _ in texts]
        )
    )
    return model


def _mock_collection(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
    distances: list[float],
) -> MagicMock:
    """Fake ChromaDB collection whose query() returns the given data."""
    col = MagicMock()
    col.query.return_value = {
        "ids": [ids],
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }
    return col


def _sample_metadata(work_slug: str = "beyond_good_and_evil", aph: int = 1) -> dict:
    return {
        "work_title": "Beyond Good and Evil",
        "work_slug": work_slug,
        "work_period": "late",
        "section_number": 1,
        "aphorism_number": aph,
        "aphorism_number_end": aph,
        "chunk_index": aph - 1,
        "chunk_type": "aphorism",
    }


# ── DenseResult ───────────────────────────────────────────────────────────────


class TestDenseResult:
    def test_attributes_stored(self) -> None:
        r = DenseResult(
            id="bge_chunk_0",
            document="Truth is a woman.",
            metadata=_sample_metadata(),
            distance=0.12,
        )
        assert r.id == "bge_chunk_0"
        assert r.document == "Truth is a woman."
        assert r.distance == 0.12
        assert r.metadata["work_slug"] == "beyond_good_and_evil"

    def test_repr_contains_key_fields(self) -> None:
        r = DenseResult("id", "doc", _sample_metadata(aph=7), 0.5)
        rep = repr(r)
        assert "beyond_good_and_evil" in rep
        assert "7" in rep


# ── dense_search ──────────────────────────────────────────────────────────────


class TestDenseSearch:
    def _run(
        self,
        tmp_path: Path,
        *,
        query: str = "What is the will to power?",
        top_k: int = 3,
        where: dict | None = None,
        ids: list[str] | None = None,
        documents: list[str] | None = None,
        metadatas: list[dict] | None = None,
        distances: list[float] | None = None,
    ) -> tuple[list[DenseResult], MagicMock, MagicMock]:
        """Run dense_search with fully mocked model and collection."""
        n = top_k
        ids = ids or [f"bge_chunk_{i}" for i in range(n)]
        documents = documents or [f"doc {i}" for i in range(n)]
        metadatas = metadatas or [_sample_metadata(aph=i + 1) for i in range(n)]
        distances = distances or [0.1 * (i + 1) for i in range(n)]

        mock_col = _mock_collection(ids, documents, metadatas, distances)
        mock_mdl = _mock_model()

        with patch("retrieval.dense.get_chroma_collection", return_value=mock_col) as mock_col_fn, \
             patch("retrieval.dense.SentenceTransformer", return_value=mock_mdl):
            results = dense_search(
                query,
                persist_dir=tmp_path,
                top_k=top_k,
                where=where,
            )

        return results, mock_col, mock_mdl

    # ── return type and length ────────────────────────────────────────────────

    def test_returns_list_of_dense_results(self, tmp_path: Path) -> None:
        results, _, _ = self._run(tmp_path, top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, DenseResult) for r in results)

    def test_returns_top_k_results(self, tmp_path: Path) -> None:
        results, _, _ = self._run(tmp_path, top_k=5)
        assert len(results) == 5

    # ── embedding ────────────────────────────────────────────────────────────

    def test_model_encode_called_with_query(self, tmp_path: Path) -> None:
        query = "What is the eternal recurrence?"
        _, _, mock_mdl = self._run(tmp_path, query=query, top_k=2)
        texts_arg = mock_mdl.encode.call_args[0][0]
        assert texts_arg == [query]

    def test_model_encode_called_once(self, tmp_path: Path) -> None:
        _, _, mock_mdl = self._run(tmp_path, top_k=3)
        assert mock_mdl.encode.call_count == 1

    def test_embedding_dim_768_passed_to_query(self, tmp_path: Path) -> None:
        """The query embedding forwarded to ChromaDB must be 768-dimensional."""
        _, mock_col, _ = self._run(tmp_path, top_k=2)
        query_embs = mock_col.query.call_args.kwargs.get(
            "query_embeddings"
        ) or mock_col.query.call_args[1]["query_embeddings"]
        assert len(query_embs[0]) == 768

    # ── ChromaDB interaction ──────────────────────────────────────────────────

    def test_collection_query_called_once(self, tmp_path: Path) -> None:
        _, mock_col, _ = self._run(tmp_path, top_k=3)
        assert mock_col.query.call_count == 1

    def test_n_results_matches_top_k(self, tmp_path: Path) -> None:
        _, mock_col, _ = self._run(tmp_path, top_k=7)
        call_kwargs = mock_col.query.call_args.kwargs
        assert call_kwargs.get("n_results") == 7

    def test_include_has_required_fields(self, tmp_path: Path) -> None:
        _, mock_col, _ = self._run(tmp_path, top_k=2)
        include = mock_col.query.call_args.kwargs.get("include", [])
        assert "documents" in include
        assert "metadatas" in include
        assert "distances" in include

    # ── where filter ─────────────────────────────────────────────────────────

    def test_where_filter_forwarded_to_collection(self, tmp_path: Path) -> None:
        filt = {"work_period": "late"}
        _, mock_col, _ = self._run(tmp_path, top_k=2, where=filt)
        call_kwargs = mock_col.query.call_args.kwargs
        assert call_kwargs.get("where") == filt

    def test_no_where_key_when_filter_is_none(self, tmp_path: Path) -> None:
        _, mock_col, _ = self._run(tmp_path, top_k=2, where=None)
        call_kwargs = mock_col.query.call_args.kwargs
        assert "where" not in call_kwargs

    # ── result content ────────────────────────────────────────────────────────

    def test_result_documents_match_collection_output(self, tmp_path: Path) -> None:
        docs = ["doc A", "doc B", "doc C"]
        results, _, _ = self._run(tmp_path, top_k=3, documents=docs)
        assert [r.document for r in results] == docs

    def test_result_distances_match_collection_output(self, tmp_path: Path) -> None:
        dists = [0.05, 0.15, 0.25]
        results, _, _ = self._run(tmp_path, top_k=3, distances=dists)
        assert [r.distance for r in results] == dists

    def test_result_ids_match_collection_output(self, tmp_path: Path) -> None:
        ids = ["bge_chunk_0", "bge_chunk_5", "bge_chunk_10"]
        results, _, _ = self._run(tmp_path, top_k=3, ids=ids)
        assert [r.id for r in results] == ids

    def test_result_metadata_preserved(self, tmp_path: Path) -> None:
        metas = [_sample_metadata(aph=i + 1) for i in range(2)]
        results, _, _ = self._run(tmp_path, top_k=2, metadatas=metas)
        for r, m in zip(results, metas):
            assert r.metadata == m

    def test_results_ordered_by_ascending_distance(self, tmp_path: Path) -> None:
        """ChromaDB returns results sorted by distance; we preserve that order."""
        dists = [0.1, 0.2, 0.4]
        results, _, _ = self._run(tmp_path, top_k=3, distances=dists)
        assert [r.distance for r in results] == dists

    # ── model name and collection name forwarded ──────────────────────────────

    def test_model_name_passed_to_sentence_transformer(self, tmp_path: Path) -> None:
        mock_col = _mock_collection(
            ["id"], ["doc"], [_sample_metadata()], [0.1]
        )
        with patch("retrieval.dense.get_chroma_collection", return_value=mock_col), \
             patch("retrieval.dense.SentenceTransformer", return_value=_mock_model()) as mock_cls:
            dense_search("q", persist_dir=tmp_path, model_name="custom/model", top_k=1)
        mock_cls.assert_called_once_with("custom/model")

    def test_collection_name_forwarded(self, tmp_path: Path) -> None:
        mock_col = _mock_collection(
            ["id"], ["doc"], [_sample_metadata()], [0.1]
        )
        with patch("retrieval.dense.get_chroma_collection", return_value=mock_col) as mock_fn, \
             patch("retrieval.dense.SentenceTransformer", return_value=_mock_model()):
            dense_search("q", persist_dir=tmp_path, collection_name="my_col", top_k=1)
        passed = mock_fn.call_args.args[1] if len(mock_fn.call_args.args) > 1 \
            else mock_fn.call_args.kwargs.get("collection_name")
        assert passed == "my_col"
