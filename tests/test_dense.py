"""Tests for retrieval/dense.py — all model and vector store calls are mocked."""

from __future__ import annotations

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
        *,
        query: str = "What is the will to power?",
        top_k: int = 3,
        filter_period: str | None = None,
        filter_slug: str | None = None,
        ids: list[str] | None = None,
        documents: list[str] | None = None,
        metadatas: list[dict] | None = None,
        distances: list[float] | None = None,
    ) -> tuple[list[DenseResult], MagicMock, MagicMock]:
        """Run dense_search with fully mocked model and store."""
        n = top_k
        ids = ids or [f"bge_chunk_{i}" for i in range(n)]
        documents = documents or [f"doc {i}" for i in range(n)]
        metadatas = metadatas or [_sample_metadata(aph=i + 1) for i in range(n)]
        distances = distances or [0.1 * (i + 1) for i in range(n)]

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            {"id": i, "document": d, "metadata": m, "distance": dist}
            for i, d, m, dist in zip(ids, documents, metadatas, distances)
        ]
        mock_mdl = _mock_model()

        with patch("retrieval.dense.get_vector_store", return_value=mock_store), \
             patch("retrieval.dense.SentenceTransformer", return_value=mock_mdl):
            results = dense_search(
                query,
                top_k=top_k,
                filter_period=filter_period,
                filter_slug=filter_slug,
            )

        return results, mock_store, mock_mdl

    # ── return type and length ────────────────────────────────────────────────

    def test_returns_list_of_dense_results(self) -> None:
        results, _, _ = self._run(top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(r, DenseResult) for r in results)

    def test_returns_top_k_results(self) -> None:
        results, _, _ = self._run(top_k=5)
        assert len(results) == 5

    # ── embedding ────────────────────────────────────────────────────────────

    def test_model_encode_called_with_query(self) -> None:
        query = "What is the eternal recurrence?"
        _, _, mock_mdl = self._run(query=query, top_k=2)
        texts_arg = mock_mdl.encode.call_args[0][0]
        assert texts_arg == [query]

    def test_model_encode_called_once(self) -> None:
        _, _, mock_mdl = self._run(top_k=3)
        assert mock_mdl.encode.call_count == 1

    def test_embedding_dim_768_passed_to_search(self) -> None:
        """The query embedding forwarded to similarity_search must be 768-dimensional."""
        _, mock_store, _ = self._run(top_k=2)
        call_args = mock_store.similarity_search.call_args
        query_emb = call_args[0][0] if call_args[0] else call_args[1]["query_embedding"]
        assert len(query_emb) == 768

    # ── store interaction ─────────────────────────────────────────────────────

    def test_store_similarity_search_called_once(self) -> None:
        _, mock_store, _ = self._run(top_k=3)
        assert mock_store.similarity_search.call_count == 1

    def test_n_results_matches_top_k(self) -> None:
        _, mock_store, _ = self._run(top_k=7)
        call_kwargs = mock_store.similarity_search.call_args[1]
        assert call_kwargs.get("top_k") == 7

    # ── filter forwarding ─────────────────────────────────────────────────────

    def test_filter_period_forwarded(self) -> None:
        _, mock_store, _ = self._run(top_k=2, filter_period="late")
        call_kwargs = mock_store.similarity_search.call_args[1]
        assert call_kwargs.get("filter_period") == "late"

    def test_filter_slug_forwarded(self) -> None:
        _, mock_store, _ = self._run(top_k=2, filter_slug="beyond_good_and_evil")
        call_kwargs = mock_store.similarity_search.call_args[1]
        assert call_kwargs.get("filter_slug") == "beyond_good_and_evil"

    def test_no_filter_when_none(self) -> None:
        _, mock_store, _ = self._run(top_k=2, filter_period=None, filter_slug=None)
        call_kwargs = mock_store.similarity_search.call_args[1]
        assert call_kwargs.get("filter_period") is None
        assert call_kwargs.get("filter_slug") is None

    # ── result content ────────────────────────────────────────────────────────

    def test_result_documents_match_store_output(self) -> None:
        docs = ["doc A", "doc B", "doc C"]
        results, _, _ = self._run(top_k=3, documents=docs)
        assert [r.document for r in results] == docs

    def test_result_distances_match_store_output(self) -> None:
        dists = [0.05, 0.15, 0.25]
        results, _, _ = self._run(top_k=3, distances=dists)
        assert [r.distance for r in results] == dists

    def test_result_ids_match_store_output(self) -> None:
        ids = ["bge_chunk_0", "bge_chunk_5", "bge_chunk_10"]
        results, _, _ = self._run(top_k=3, ids=ids)
        assert [r.id for r in results] == ids

    def test_result_metadata_preserved(self) -> None:
        metas = [_sample_metadata(aph=i + 1) for i in range(2)]
        results, _, _ = self._run(top_k=2, metadatas=metas)
        for r, m in zip(results, metas):
            assert r.metadata == m

    def test_results_ordered_by_ascending_distance(self) -> None:
        """Store returns results sorted by distance; we preserve that order."""
        dists = [0.1, 0.2, 0.4]
        results, _, _ = self._run(top_k=3, distances=dists)
        assert [r.distance for r in results] == dists

    # ── model name forwarded ──────────────────────────────────────────────────

    def test_model_name_passed_to_sentence_transformer(self) -> None:
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            {"id": "id", "document": "doc", "metadata": _sample_metadata(), "distance": 0.1}
        ]
        with patch("retrieval.dense.get_vector_store", return_value=mock_store), \
             patch("retrieval.dense.SentenceTransformer", return_value=_mock_model()) as mock_cls:
            dense_search("q", model_name="custom/model", top_k=1)
        mock_cls.assert_called_once_with("custom/model")
