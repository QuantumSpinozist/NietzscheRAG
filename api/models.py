"""Pydantic request/response schemas for the Nietzsche RAG API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Body for ``POST /query``."""

    question: str = Field(..., description="Natural-language question about Nietzsche.")
    filter_period: Literal["early", "middle", "late"] | None = Field(
        None, description="Restrict retrieval to a philosophical period."
    )
    filter_slug: str | None = Field(
        None, description="Restrict retrieval to a single work slug."
    )


class SourceResult(BaseModel):
    """A single retrieved passage included in the response."""

    work_title: str
    work_slug: str
    section_number: int | None
    chunk_type: str
    content: str
    similarity: float


class QueryResponse(BaseModel):
    """Body returned by ``POST /query``."""

    answer: str
    sources: list[SourceResult]
