"""Integration tests for retrieval/chroma_store.py — uses a real (tmp) ChromaDB."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from retrieval.chroma_store import ChromaStore, get_chroma_collection


class TestGetChromaCollection:
    def test_creates_collection(self, tmp_path: Path) -> None:
        with patch("retrieval.chroma_store.chromadb.PersistentClient") as mock_client_cls:
            mock_col = MagicMock()
            mock_client_cls.return_value.get_or_create_collection.return_value = mock_col
            result = get_chroma_collection(tmp_path, "test")
        mock_client_cls.assert_called_once_with(path=str(tmp_path))
        mock_client_cls.return_value.get_or_create_collection.assert_called_once_with("test")
        assert result is mock_col

    def test_creates_persist_dir_if_missing(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "chroma"
        with patch("retrieval.chroma_store.chromadb.PersistentClient"):
            get_chroma_collection(deep, "test")
        assert deep.exists()


class TestChromaStore:
    def test_store_and_retrieve(self, tmp_path: Path) -> None:
        """Real ChromaDB round-trip: store one chunk, retrieve it."""
        store = ChromaStore(persist_dir=tmp_path, collection_name="test_col")
        chunks = [
            {
                "id": "bge_chunk_0",
                "content": "Truth is a woman.",
                "work_title": "Beyond Good and Evil",
                "work_slug": "beyond_good_and_evil",
                "work_period": "late",
                "section_number": 1,
                "aphorism_number": 1,
                "aphorism_number_end": 1,
                "chunk_index": 0,
                "chunk_type": "aphorism",
            }
        ]
        embedding = [[0.1] * 768]
        store.store_chunks(chunks, embedding)

        results = store.similarity_search([0.1] * 768, top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "bge_chunk_0"
        assert results[0]["document"] == "Truth is a woman."
        assert "metadata" in results[0]
        assert "distance" in results[0]

    def test_get_all_documents(self, tmp_path: Path) -> None:
        store = ChromaStore(persist_dir=tmp_path, collection_name="test_col2")
        chunks = [
            {
                "id": f"chunk_{i}",
                "content": f"doc {i}",
                "work_title": "BGE", "work_slug": "bge", "work_period": "late",
                "section_number": i, "aphorism_number": i, "aphorism_number_end": i,
                "chunk_index": i, "chunk_type": "aphorism",
            }
            for i in range(3)
        ]
        store.store_chunks(chunks, [[0.1] * 768] * 3)
        corpus = store.get_all_documents()
        assert set(corpus.keys()) >= {"ids", "documents", "metadatas"}
        assert len(corpus["ids"]) == 3

    def test_delete_all(self, tmp_path: Path) -> None:
        store = ChromaStore(persist_dir=tmp_path, collection_name="test_del")
        chunks = [
            {
                "id": "chunk_0",
                "content": "test",
                "work_title": "BGE", "work_slug": "bge", "work_period": "late",
                "section_number": 1, "aphorism_number": 1, "aphorism_number_end": 1,
                "chunk_index": 0, "chunk_type": "aphorism",
            }
        ]
        store.store_chunks(chunks, [[0.1] * 768])
        store.delete_all()
        # After delete, a new store on the same path should have an empty (or new) collection
        store2 = ChromaStore(persist_dir=tmp_path, collection_name="test_del")
        corpus = store2.get_all_documents()
        assert len(corpus["ids"]) == 0
