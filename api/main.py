"""FastAPI application — mounts all routers and manages startup/shutdown.

Run locally with::

    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import CrossEncoder, SentenceTransformer

import config
from api.routes import ingest, query
from retrieval.sparse import BM25Index
from retrieval.store import get_vector_store


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load heavy resources once at startup and store them on ``app.state``.

    Loaded resources:
    - ``sentence_transformer``: embedding model for dense retrieval.
    - ``cross_encoder``: reranking model.
    - ``bm25_index``: BM25 index built over all corpus documents.
    """
    app.state.sentence_transformer = SentenceTransformer(config.EMBEDDING_MODEL)
    app.state.cross_encoder = CrossEncoder(config.RERANKER_MODEL)

    data = get_vector_store().get_all_documents()
    app.state.bm25_index = BM25Index(
        ids=data["ids"],
        documents=data["documents"],
        metadatas=data["metadatas"],
    )

    yield
    # No cleanup required.


app = FastAPI(
    title="Nietzsche RAG API",
    description="Retrieval-Augmented Generation over the complete Nietzsche corpus.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router)
app.include_router(ingest.router)
