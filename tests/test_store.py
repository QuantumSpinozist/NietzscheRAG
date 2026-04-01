"""Unit tests for retrieval/store.py — abstract interface + factory."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from retrieval.store import VectorStore, get_vector_store


class TestVectorStoreAbstract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            VectorStore()  # type: ignore[abstract]


class TestGetVectorStore:
    def test_returns_chroma_by_default(self) -> None:
        with patch.dict(os.environ, {"VECTOR_STORE_BACKEND": "chroma"}):
            with patch("retrieval.chroma_store.ChromaStore.__init__", return_value=None):
                from retrieval.chroma_store import ChromaStore
                store = get_vector_store()
                assert isinstance(store, ChromaStore)

    def test_returns_supabase_when_configured(self) -> None:
        with patch.dict(os.environ, {"VECTOR_STORE_BACKEND": "supabase"}):
            with patch("retrieval.supabase_store.SupabaseStore.__init__", return_value=None):
                from retrieval.supabase_store import SupabaseStore
                store = get_vector_store()
                assert isinstance(store, SupabaseStore)

    def test_unknown_backend_returns_chroma(self) -> None:
        with patch.dict(os.environ, {"VECTOR_STORE_BACKEND": "unknown"}):
            with patch("retrieval.chroma_store.ChromaStore.__init__", return_value=None):
                from retrieval.chroma_store import ChromaStore
                store = get_vector_store()
                assert isinstance(store, ChromaStore)
