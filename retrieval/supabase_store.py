"""Supabase + pgvector implementation of VectorStore (production backend).

Prerequisites
-------------
Run the following SQL in your Supabase project once:

    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE chunks (
        id                 BIGSERIAL PRIMARY KEY,
        chunk_id           TEXT NOT NULL UNIQUE,   -- e.g. "bge_chunk_0"
        work_title         TEXT NOT NULL,
        work_slug          TEXT NOT NULL,
        work_period        TEXT NOT NULL,
        chunk_type         TEXT NOT NULL,
        section_number     INT,
        aphorism_number    INT,
        aphorism_number_end INT,
        chunk_index        INT NOT NULL,
        content            TEXT NOT NULL,
        embedding          vector(768)
    );

    CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);

    CREATE OR REPLACE FUNCTION match_chunks(
        query_embedding     vector(768),
        match_count         int DEFAULT 10,
        filter_period       text DEFAULT NULL,
        filter_slug         text DEFAULT NULL
    )
    RETURNS TABLE (
        chunk_id            text,
        content             text,
        work_title          text,
        work_slug           text,
        work_period         text,
        section_number      int,
        aphorism_number     int,
        aphorism_number_end int,
        chunk_index         int,
        chunk_type          text,
        similarity          float
    )
    LANGUAGE sql STABLE AS $$
        SELECT
            chunk_id, content, work_title, work_slug, work_period,
            section_number, aphorism_number, aphorism_number_end,
            chunk_index, chunk_type,
            1 - (embedding <=> query_embedding) AS similarity
        FROM chunks
        WHERE
            (filter_period IS NULL OR work_period = filter_period)
            AND (filter_slug IS NULL OR work_slug = filter_slug)
        ORDER BY embedding <=> query_embedding
        LIMIT match_count;
    $$;
"""

from __future__ import annotations

import config
from retrieval.store import VectorStore

_METADATA_COLS = (
    "work_title", "work_slug", "work_period", "chunk_type",
    "section_number", "aphorism_number", "aphorism_number_end",
    "chunk_index",
)
_ALL_COLS = "chunk_id,content," + ",".join(_METADATA_COLS)


class SupabaseStore(VectorStore):
    """VectorStore backed by Supabase + pgvector (production)."""

    def __init__(self) -> None:
        from supabase import create_client

        if not config.SUPABASE_URL or not config.SUPABASE_KEY:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env "
                "when VECTOR_STORE_BACKEND=supabase"
            )
        self._client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

    def store_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        """Upsert chunks and their embeddings into Supabase."""
        rows = [
            {
                "chunk_id": chunk["id"],
                "content": chunk["content"],
                **{k: chunk.get(k) for k in _METADATA_COLS},
                "embedding": embedding,
            }
            for chunk, embedding in zip(chunks, embeddings)
        ]
        self._client.table("chunks").upsert(rows, on_conflict="chunk_id").execute()

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_period: str | None = None,
        filter_slug: str | None = None,
    ) -> list[dict]:
        """Return the *top_k* most similar chunks via pgvector RPC."""
        result = self._client.rpc(
            "match_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": top_k,
                "filter_period": filter_period,
                "filter_slug": filter_slug,
            },
        ).execute()

        return [
            {
                "id": row["chunk_id"],
                "document": row["content"],
                "metadata": {k: row.get(k) for k in _METADATA_COLS},
                "distance": 1.0 - float(row["similarity"]),
            }
            for row in result.data
        ]

    def get_all_documents(self) -> dict:
        """Return all stored documents as a dict with ids, documents, metadatas.

        Paginates in batches of 1000 to work around the Supabase client's
        default row-count limit.
        """
        rows: list[dict] = []
        page_size = 1000
        offset = 0
        while True:
            batch = (
                self._client.table("chunks")
                .select(_ALL_COLS)
                .range(offset, offset + page_size - 1)
                .execute()
                .data
            )
            rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
        return {
            "ids": [r["chunk_id"] for r in rows],
            "documents": [r["content"] for r in rows],
            "metadatas": [{k: r.get(k) for k in _METADATA_COLS} for r in rows],
        }

    def delete_all(self) -> None:
        """Delete all rows from the chunks table."""
        self._client.table("chunks").delete().neq("id", 0).execute()
