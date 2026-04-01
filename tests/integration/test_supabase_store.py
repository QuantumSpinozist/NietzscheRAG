"""Integration tests for retrieval/supabase_store.py.

Skipped automatically when SUPABASE_URL is not set.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("SUPABASE_URL"),
    reason="Supabase credentials not configured",
)


def _sample_chunk(idx: int = 0) -> dict:
    return {
        "id": f"test_chunk_{idx}",
        "content": f"Test passage {idx}.",
        "work_title": "Beyond Good and Evil",
        "work_slug": "beyond_good_and_evil",
        "work_period": "late",
        "section_number": idx + 1,
        "aphorism_number": idx + 1,
        "aphorism_number_end": idx + 1,
        "chunk_index": idx,
        "chunk_type": "aphorism",
    }


class TestSupabaseStore:
    def test_store_and_retrieve_chunk(self) -> None:
        from retrieval.supabase_store import SupabaseStore

        store = SupabaseStore()
        store.delete_all()

        chunk = _sample_chunk(0)
        store.store_chunks([chunk], [[0.1] * 768])

        results = store.similarity_search([0.1] * 768, top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "test_chunk_0"
        assert results[0]["document"] == "Test passage 0."

    def test_filter_by_period(self) -> None:
        from retrieval.supabase_store import SupabaseStore

        store = SupabaseStore()
        store.delete_all()

        chunks = [_sample_chunk(i) for i in range(3)]
        store.store_chunks(chunks, [[0.1] * 768] * 3)

        results = store.similarity_search([0.1] * 768, top_k=10, filter_period="late")
        assert all(r["metadata"]["work_period"] == "late" for r in results)

    def test_delete_all_clears_table(self) -> None:
        from retrieval.supabase_store import SupabaseStore

        store = SupabaseStore()
        store.store_chunks([_sample_chunk(0)], [[0.1] * 768])
        store.delete_all()
        corpus = store.get_all_documents()
        assert len(corpus["ids"]) == 0
