"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Header, HTTPException

import config


async def verify_ingest_token(x_ingest_token: str | None = Header(None)) -> None:
    """Raise 401 if the ``X-Ingest-Token`` header is missing or wrong.

    Raises:
        HTTPException(401): Token is absent or does not match ``config.INGEST_TOKEN``.
        HTTPException(500): ``INGEST_TOKEN`` is not configured on the server.
    """
    if not config.INGEST_TOKEN:
        raise HTTPException(status_code=500, detail="INGEST_TOKEN not configured on server.")
    if x_ingest_token != config.INGEST_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Ingest-Token header.")
