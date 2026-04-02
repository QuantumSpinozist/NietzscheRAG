"""Central configuration — paths, model names, and retrieval hyperparameters."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── API keys ──────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# ── Models ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATION_MODEL: str = "claude-sonnet-4-5"
HYDE_MODEL: str = "claude-haiku-4-5-20251001"  # cheap/fast model for hypothetical generation

# ── Storage ───────────────────────────────────────────────────────────────────

CHROMA_PERSIST_DIR: Path = Path("./data/chroma")
COLLECTION_NAME: str = "nietzsche"
RAW_DIR: Path = Path("./data/raw")
VECTOR_STORE_BACKEND: str = os.getenv("VECTOR_STORE_BACKEND", "chroma")
SUPABASE_URL: str | None = os.getenv("SUPABASE_URL")
SUPABASE_KEY: str | None = os.getenv("SUPABASE_SERVICE_KEY")

# ── API ───────────────────────────────────────────────────────────────────────

INGEST_TOKEN: str = os.getenv("INGEST_TOKEN", "")
ALLOWED_ORIGINS: list[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ── Retrieval hyperparameters ─────────────────────────────────────────────────

DENSE_TOP_K: int = 20
SPARSE_TOP_K: int = 20
RERANK_TOP_N: int = 10
RRF_K: int = 60
# Additive bonus applied to aphorism chunks after cross-encoder scoring.
# Cross-encoder logits are roughly in [-10, +10]; 0.5 is a meaningful nudge
# that prefers specific aphorisms over broad prose when scores are close.
APHORISM_RERANK_BONUS: float = 1.5
MULTIQUERY_N: int = 2  # number of paraphrase variants for multi-query expansion

# ── Ingestion hyperparameters ─────────────────────────────────────────────────

EMBED_BATCH_SIZE: int = 64
MIN_APHORISM_TOKENS: int = 50
TARGET_PROSE_TOKENS: int = 300
OVERLAP_TOKENS: int = 100
