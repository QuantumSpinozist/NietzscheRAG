"""POST /ingest — trigger a full corpus ingest run (protected endpoint)."""

from __future__ import annotations

import threading

from fastapi import APIRouter, BackgroundTasks, Depends

import config
from api.dependencies import verify_ingest_token
from ingest.embed import WORK_END_BEFORE, WORK_REGISTRY, embed_chunks

router = APIRouter()


def run_ingest() -> int:
    """Chunk and embed every work whose raw ``.txt`` file is present on disk.

    Returns:
        Total number of chunks written to the vector store.
    """
    from ingest.chunk import chunk_work

    total = 0
    for slug, (title, period, chunk_style) in WORK_REGISTRY.items():
        raw_path = config.RAW_DIR / f"{slug}.txt"
        if not raw_path.exists():
            continue
        text = raw_path.read_text(encoding="utf-8")
        end_before = WORK_END_BEFORE.get(slug)
        chunks = chunk_work(text, title, slug, period, chunk_style, end_before=end_before)
        total += embed_chunks(chunks)

    return total


@router.post("/ingest")
async def ingest_endpoint(
    background_tasks: BackgroundTasks,
    _: None = Depends(verify_ingest_token),
) -> dict:
    """Re-ingest all available works into the vector store (runs in background).

    Returns immediately with ``{"status": "started"}`` while ingestion proceeds.
    Requires the ``X-Ingest-Token`` header matching ``INGEST_TOKEN`` env var.
    """
    background_tasks.add_task(run_ingest)
    return {"status": "started"}
