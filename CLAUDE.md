# Nietzsche RAG — Claude Code Instructions

## Project Overview

A Retrieval-Augmented Generation (RAG) system over the complete Nietzsche corpus.
Users ask philosophical questions; the system retrieves relevant passages and generates
grounded, cited answers using Claude.

---

## Project Structure

```
nietzsche-rag/
├── data/
│   └── raw/                  # Plain .txt files, one per work (sourced from Gutenberg)
├── ingest/
│   ├── fetch.py              # Download texts from Project Gutenberg
│   ├── chunk.py              # Aphorism-aware and paragraph-aware chunking
│   └── embed.py              # Embed chunks and write via store interface
├── retrieval/
│   ├── store.py              # Abstract VectorStore base class + get_vector_store() factory
│   ├── chroma_store.py       # ChromaDB implementation (local dev)
│   ├── supabase_store.py     # Supabase + pgvector implementation (production)
│   ├── sparse.py             # BM25 keyword search via rank_bm25
│   ├── dense.py              # Dense search with optional embed_text override (for HyDE)
│   ├── hyde.py               # HyDE: generate hypothetical Nietzsche passage via Claude
│   ├── multiquery.py         # Multi-query expansion: generate paraphrase variants via Claude
│   └── hybrid.py             # RRF merge + cross-encoder reranking + aphorism bonus
├── generation/
│   └── claude.py             # Build prompt and call Claude API
├── api/
│   ├── main.py               # FastAPI app — mounts all routers
│   ├── routes/
│   │   ├── query.py          # POST /query
│   │   └── ingest.py         # POST /ingest  (protected)
│   ├── models.py             # Pydantic request/response schemas
│   └── dependencies.py       # Shared FastAPI dependencies (auth header check, etc.)
├── frontend/                 # Next.js app (separate repo or subdirectory)
│   ├── app/
│   │   ├── page.tsx          # Main chat interface
│   │   └── api/
│   │       └── query/
│   │           └── route.ts  # Next.js API route — proxies to FastAPI
│   ├── components/
│   │   ├── ChatInput.tsx
│   │   ├── MessageList.tsx
│   │   ├── SourceCard.tsx    # Displays cited passage + work/section metadata
│   │   └── FilterBar.tsx     # Period + work dropdowns
│   └── package.json
├── tests/
│   ├── conftest.py
│   ├── test_chunk.py
│   ├── test_fetch.py
│   ├── test_store.py
│   ├── test_sparse.py
│   ├── test_hybrid.py
│   ├── test_generation.py
│   ├── test_api.py           # FastAPI route tests (TestClient, mocked pipeline)
│   └── integration/
│       ├── test_chroma_store.py
│       └── test_supabase_store.py
├── app.py                    # CLI entry point (typer + rich) — kept for local use
├── config.py                 # Central config (paths, model names, hyperparams)
├── requirements.txt
└── CLAUDE.md                 # This file
```

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Data source | Project Gutenberg | Plain text, public domain |
| Chunking | Custom Python | Aphorism-aware (see below) |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` | Local, free, 768-dim |
| Vector store (local) | `chromadb` | Used when `VECTOR_STORE_BACKEND=chroma` |
| Vector store (prod) | Supabase + `pgvector` | Used when `VECTOR_STORE_BACKEND=supabase` |
| Store interface | `retrieval/store.py` | Abstract base — rest of app never imports a backend directly |
| Keyword search | `rank_bm25` | Hybrid retrieval complement |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Top-10 → Top-3 refinement |
| Generation | Claude (via Anthropic SDK) | See generation instructions |
| API backend | FastAPI | Deployed on Fly.io |
| Frontend | Next.js (App Router) | Deployed on Vercel (free hobby plan) |
| CLI | `typer` + `rich` | Kept for local ingestion runs |

---

## Corpus

Define all works in `config.py` as a single list of dicts. This is the single source of truth
used by `fetch.py`, `chunk.py`, and metadata tagging:

```python
WORKS = [
    # Late period
    {"title": "Beyond Good and Evil",         "slug": "beyond_good_and_evil",       "period": "late",   "type": "aphoristic"},
    {"title": "On the Genealogy of Morality", "slug": "genealogy_of_morality",      "period": "late",   "type": "essay"},
    {"title": "Twilight of the Idols",        "slug": "twilight_of_the_idols",      "period": "late",   "type": "aphoristic"},
    {"title": "The Antichrist",               "slug": "the_antichrist",             "period": "late",   "type": "essay"},
    {"title": "Ecce Homo",                    "slug": "ecce_homo",                  "period": "late",   "type": "essay"},
    {"title": "Nietzsche contra Wagner",      "slug": "nietzsche_contra_wagner",    "period": "late",   "type": "essay"},
    # Middle period
    {"title": "The Gay Science",              "slug": "the_gay_science",            "period": "middle", "type": "aphoristic"},
    {"title": "Dawn",                         "slug": "dawn",                       "period": "middle", "type": "aphoristic"},
    {"title": "Human, All Too Human",         "slug": "human_all_too_human",        "period": "middle", "type": "aphoristic"},
    {"title": "Thus Spoke Zarathustra",       "slug": "thus_spoke_zarathustra",     "period": "middle", "type": "essay"},
    # Early period
    {"title": "The Birth of Tragedy",         "slug": "birth_of_tragedy",           "period": "early",  "type": "essay"},
    {"title": "Untimely Meditations",         "slug": "untimely_meditations",       "period": "early",  "type": "essay"},
]
```

The `type` field drives chunking strategy — `aphoristic` uses aphorism-boundary chunking,
`essay` uses paragraph chunking. Store raw texts as `data/raw/{slug}.txt`.

---

## Chunking Strategy

This is the most important part of the pipeline. Nietzsche's works fall into two structural types:

### Aphoristic works (BGE, GS, Dawn, HH, TI, AC)
- Each numbered aphorism/section is its own natural chunk
- Parse by detecting section headers: lines matching `^\d+\.` or `^\d+$`
- Do NOT split aphorisms further — they are self-contained arguments
- Short aphorisms (<50 tokens) may be merged with adjacent ones

### Essay/prose works (GM, BT, Untimely Meditations)
- Chunk by paragraph
- Target chunk size: ~300 tokens
- Overlap: ~50 tokens between chunks to preserve context across boundaries

### Metadata to store per chunk
Every chunk must carry this metadata regardless of which backend is active:
```python
{
    "work_title": str,          # e.g. "Beyond Good and Evil"
    "work_slug": str,           # e.g. "beyond_good_and_evil"
    "work_period": str,         # "early" | "middle" | "late"
    "section_number": int,      # aphorism or chapter number (if available)
    "chunk_index": int,         # position within work
    "chunk_type": str,          # "aphorism" | "paragraph"
}
```

---

## Vector Store — Repository Pattern

The app supports two vector store backends switchable via `.env`. All code outside
`retrieval/` must use only the abstract interface — never import `ChromaStore` or
`SupabaseStore` directly.

### Abstract interface (`retrieval/store.py`)

```python
from abc import ABC, abstractmethod

class VectorStore(ABC):

    @abstractmethod
    def store_chunks(self, chunks: list[dict], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_period: str | None = None,
        filter_slug: str | None = None,
    ) -> list[dict]: ...

    @abstractmethod
    def delete_all(self) -> None: ...


def get_vector_store() -> VectorStore:
    """Read VECTOR_STORE_BACKEND from env and return the correct implementation."""
    import os
    backend = os.getenv("VECTOR_STORE_BACKEND", "chroma")
    if backend == "supabase":
        from retrieval.supabase_store import SupabaseStore
        return SupabaseStore()
    from retrieval.chroma_store import ChromaStore
    return ChromaStore()
```

### Switching backends

```bash
# Local development (default)
VECTOR_STORE_BACKEND=chroma

# Production
VECTOR_STORE_BACKEND=supabase
```

Supabase credentials are only required when `VECTOR_STORE_BACKEND=supabase`.
ChromaDB requires no credentials — data persists to `CHROMA_PERSIST_DIR`.

### Supabase setup

Run this once in the Supabase SQL editor to create the schema:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE chunks (
    id             BIGSERIAL PRIMARY KEY,
    work_title     TEXT NOT NULL,
    work_slug      TEXT NOT NULL,
    work_period    TEXT NOT NULL,
    chunk_type     TEXT NOT NULL,
    section_number INT,
    chunk_index    INT NOT NULL,
    content        TEXT NOT NULL,
    embedding      vector(768)
);

-- HNSW index for fast approximate nearest-neighbour search
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);

-- Similarity search function called via Supabase RPC
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding vector(768),
    match_count     int DEFAULT 10,
    filter_period   text DEFAULT NULL,
    filter_slug     text DEFAULT NULL
)
RETURNS TABLE (
    id             bigint,
    content        text,
    work_title     text,
    work_slug      text,
    work_period    text,
    section_number int,
    chunk_type     text,
    similarity     float
)
LANGUAGE sql STABLE AS $$
    SELECT
        id, content, work_title, work_slug,
        work_period, section_number, chunk_type,
        1 - (embedding <=> query_embedding) AS similarity
    FROM chunks
    WHERE
        (filter_period IS NULL OR work_period = filter_period)
        AND (filter_slug IS NULL OR work_slug = filter_slug)
    ORDER BY embedding <=> query_embedding
    LIMIT match_count;
$$;
```

---

## Retrieval

Use hybrid retrieval — do not rely on dense search alone. Nietzsche uses coined terms
(*Ressentiment*, *Übermensch*, *Umwertung*) that keyword search captures better.

### Pipeline
0. (Optional) **HyDE**: generate a hypothetical Nietzsche-style passage via Claude Haiku,
   embed that instead of the raw question for the dense search step.
0. (Optional) **Multi-query expansion**: generate 2 paraphrase variants of the question
   via Claude Haiku and run dense search for each; all lists feed into RRF.
1. Call `get_vector_store().similarity_search()` → top 20 dense results
2. Run BM25 search → top 20 results from in-memory index
3. Merge with Reciprocal Rank Fusion (RRF): `score = 1 / (k + rank)` where k=60
4. Re-rank merged candidates with cross-encoder → return top 7
   - After cross-encoder scoring, apply `APHORISM_RERANK_BONUS = 1.5` to chunks
     with `chunk_type == "aphorism"` to prefer specific aphorisms over broad prose.

### Configurable parameters (in `config.py`)
```python
DENSE_TOP_K = 20
SPARSE_TOP_K = 20
RERANK_TOP_N = 7
RRF_K = 60
APHORISM_RERANK_BONUS = 1.5   # additive bonus for aphorism chunks after cross-encoder scoring
MULTIQUERY_N = 2               # number of paraphrase variants for multi-query expansion
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Backend selection
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "chroma")

# ChromaDB (local)
CHROMA_PERSIST_DIR = "./data/chroma"

# Supabase (production)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # use service key, not anon key
```

---

## Generation

### System prompt (use exactly this)
```
You are a philosophical research assistant specialising in Friedrich Nietzsche.
Answer the user's question using ONLY the provided source passages.
For every claim you make, cite the source work and section number in brackets, e.g. [BGE §36].
If the passages do not contain enough information to answer, say so explicitly.
Distinguish between what Nietzsche literally says and scholarly interpretation.
Do not editorialize or inject views not supported by the passages.
```

### Prompt structure
```
Source passages:
---
[PASSAGE 1]
Source: {work_title}, §{section_number}
---
[PASSAGE 2]
Source: {work_title}, §{section_number}
---

Question: {user_question}
```

### Model
Use `claude-sonnet-4-5` via the Anthropic Python SDK. Max tokens: 1024.

---

## Dependencies

```
# requirements.txt
anthropic
chromadb
supabase
sentence-transformers
rank_bm25
torch                        # required by sentence-transformers
fastapi
uvicorn[standard]
typer
rich
requests                     # for Gutenberg fetching
python-dotenv
pytest
pytest-mock
httpx                        # required by FastAPI TestClient
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Environment

```bash
# .env
ANTHROPIC_API_KEY=your_key_here

# Vector store backend: "chroma" (default, local) or "supabase" (production)
VECTOR_STORE_BACKEND=chroma

# Required only when VECTOR_STORE_BACKEND=supabase
SUPABASE_URL=https://yourproject.supabase.co
SUPABASE_SERVICE_KEY=your_service_key_here

# API
INGEST_TOKEN=a_long_random_secret_string    # protects the /ingest endpoint
ALLOWED_ORIGINS=http://localhost:3000       # comma-separated; set to Vercel URL in prod
```

Commit a `.env.example` with placeholder values. Never commit `.env` itself.
Load with `python-dotenv` in `config.py`. Never hardcode any value.

---

## CLI Interface

The app should support these commands:

```bash
# Ingest the full corpus
python app.py ingest

# Query interactively
python app.py query "What does Nietzsche mean by the will to power?"

# Query with period filter
python app.py query "What is the eternal recurrence?" --period late

# Query a specific work
python app.py query "How does Nietzsche view Socrates?" --work twilight_of_the_idols
```

---

## Development Order

Build in this sequence — each step is independently testable:

**Core pipeline (completed):**
1. `ingest/fetch.py` — download and save raw texts ✓
2. `ingest/chunk.py` — chunking logic ✓
3. `retrieval/store.py` — abstract interface + factory function ✓
4. `retrieval/chroma_store.py` — ChromaDB implementation ✓
5. `retrieval/supabase_store.py` — Supabase implementation ✓
6. `ingest/embed.py` — embed chunks, write via `get_vector_store()` ✓
7. `retrieval/sparse.py` — BM25 index ✓
8. `retrieval/hybrid.py` — RRF merge + reranker ✓
9. `generation/claude.py` — prompt builder + Claude call ✓
10. `app.py` — CLI ✓

**API backend (completed):**
11. `api/models.py` — Pydantic schemas for request/response ✓
12. `api/dependencies.py` — shared dependencies (ingest auth header) ✓
13. `api/routes/query.py` — `POST /query` route ✓
14. `api/routes/ingest.py` — `POST /ingest` route (protected, background task) ✓
15. `api/main.py` — mount routers, CORS, lifespan, `/health` endpoint ✓
16. `tests/test_api.py` — FastAPI TestClient tests ✓
17. Deploy to Fly.io ✓ — live at https://nietzsche-rag.fly.dev

**Frontend (completed):**
18. Scaffold Next.js 14 app in `frontend/` ✓
19. `components/SourceCard.tsx` — cited passage display ✓
20. `components/FilterBar.tsx` — period + work filter dropdowns ✓
21. `components/MessageList.tsx` + `ChatInput.tsx` — chat UI ✓
22. `app/page.tsx` — wire everything together ✓
23. `app/api/query/route.ts` — proxy route to FastAPI backend ✓
24. Deploy to Vercel ✓ — live at https://nietzscherag.vercel.app

**Retrieval improvements (completed):**
25. `retrieval/hyde.py` — HyDE query expansion via Claude Haiku ✓
26. `retrieval/multiquery.py` — multi-query paraphrase expansion ✓
27. `retrieval/dense.py` — `embed_text` override parameter for HyDE ✓
28. `hybrid_search()` — aphorism reranker bonus, HyDE, multi-query flags ✓
29. `eval/eval_set.py` — expanded to 15 questions with two-GT items ✓
30. `eval/run_eval.py` — HR@5, HR@10, MRR@5, MRR@10 metrics ✓
31. Tuned hyperparameters: DENSE/SPARSE_TOP_K=20, RERANK_TOP_N=7, APHORISM_BONUS=1.5 ✓

Do not move to step N+1 until step N has a passing smoke test.

---

## Code Style

- Type hints on all function signatures
- Docstrings on all public functions
- No global state — pass config explicitly
- Log progress to stderr with `rich` (not print statements)
- Raise specific exceptions, never bare `except:`

---

## API Backend (FastAPI)

The FastAPI layer wraps the existing pipeline and exposes it over HTTP. It must not
contain any retrieval or generation logic itself — it only calls into the existing modules.

### Endpoints

#### `POST /query`
```
Request:
{
  "question": str,
  "filter_period": "early" | "middle" | "late" | null,
  "filter_slug": str | null
}

Response:
{
  "answer": str,
  "sources": [
    {
      "work_title": str,
      "work_slug": str,
      "section_number": int | null,
      "chunk_type": str,
      "content": str,
      "similarity": float
    }
  ]
}
```

#### `POST /ingest`
Protected by a static `X-Ingest-Token` header (set via env var `INGEST_TOKEN`).
Triggers a full corpus ingest run. Returns `{"status": "ok", "chunks_written": int}`.
This endpoint exists so ingestion can be triggered remotely without SSH access.

### Pydantic models (`api/models.py`)
Define `QueryRequest`, `SourceResult`, and `QueryResponse` — all fields typed and documented.
Never use raw dicts in route handlers.

### CORS
Allow all origins in development. In production, restrict to your Vercel frontend domain via
`ALLOWED_ORIGINS` env var:
```python
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
```

### Lifespan
Load the embedding model and BM25 index once at startup using FastAPI's `lifespan` context
manager — do not reload them per request. Store on `app.state`.

### Running locally
```bash
uvicorn api.main:app --reload --port 8000
```

### New dependencies to add to `requirements.txt`
```
fastapi
uvicorn[standard]
```

---

## Frontend (Next.js)

A focused, well-designed chat interface. The goal is something that looks portfolio-quality,
not a generic chatbot. Lean into the Nietzsche theme in the design — dark, typographic, serious.

### Stack
- Next.js 14+ with App Router
- TypeScript throughout — no `any` types
- Tailwind CSS for styling
- No UI component library — write components from scratch for portfolio value

### Key components

#### `SourceCard.tsx`
Displays a retrieved passage with its citation. This is the most important UI component —
it's what makes this app different from a generic chatbot:
```
┌─────────────────────────────────┐
│ Beyond Good and Evil  §36       │
│ Late period · Aphorism          │
├─────────────────────────────────┤
│ "Supposing truth is a woman..." │
│                                 │
│ Similarity: 0.87                │
└─────────────────────────────────┘
```

#### `FilterBar.tsx`
Two dropdowns: period (`early` / `middle` / `late` / `all`) and work (slug list from
`WORKS` config, or `all`). Filters are sent with every query.

#### `MessageList.tsx`
Renders the conversation. Each assistant message has an expandable "Sources" section
below it showing `SourceCard` components.

#### `ChatInput.tsx`
Simple textarea + submit button. Support `Shift+Enter` for newlines, `Enter` to submit.

### API proxy route (`app/api/query/route.ts`)
Proxy requests from the frontend to the FastAPI backend. This hides the backend URL
from the browser and lets you set the `FASTAPI_URL` as a server-side env var:
```typescript
const res = await fetch(`${process.env.FASTAPI_URL}/query`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});
```

### Environment variables (frontend)
```
FASTAPI_URL=https://your-fly-app.fly.dev   # server-side only, no NEXT_PUBLIC_ prefix
```

---

## Deployment

### Backend — Fly.io

Fly.io free tier supports one always-on VM. The embedding model (~420MB) needs to be
bundled into the Docker image.

**`Dockerfile` (in project root):**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-mpnet-base-v2')"

COPY . .
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**`fly.toml` (generated by `fly launch`, then edit):**
```toml
[http_service]
  internal_port = 8080
  force_https = true

[[vm]]
  memory = "1gb"     # needed for the embedding model
  cpu_kind = "shared"
  cpus = 1
```

**Deploy commands:**
```bash
fly launch          # first time — generates fly.toml
fly secrets set ANTHROPIC_API_KEY=... SUPABASE_URL=... SUPABASE_SERVICE_KEY=... INGEST_TOKEN=... ALLOWED_ORIGINS=https://your-vercel-app.vercel.app
fly deploy          # subsequent deploys
```

**Trigger ingestion after deploy:**
```bash
curl -X POST https://your-fly-app.fly.dev/ingest \
  -H "X-Ingest-Token: your_token_here"
```

### Frontend — Vercel

```bash
cd frontend
vercel                    # first deploy, follow prompts
vercel env add FASTAPI_URL   # set to your Fly.io URL
vercel --prod             # subsequent deploys
```

Vercel auto-deploys on every push to `main` once connected to GitHub.

---

## Testing

### Philosophy

Every module gets a test file written **at the same time** as the implementation — not after.
When Claude Code writes `ingest/chunk.py`, it must also write `tests/test_chunk.py` in the same step.
Tests are not optional polish; they are how we verify the pipeline before wiring modules together.

Do not use the live embedding model or the Claude API in unit tests.
Do not instantiate `ChromaStore` or `SupabaseStore` directly in unit tests — mock
`get_vector_store()` to return a `MagicMock`. This keeps all unit tests backend-agnostic.
Use fixtures and mocks for anything with I/O or external dependencies.

---

### Test Structure

```
nietzsche-rag/
├── tests/
│   ├── conftest.py               # Shared fixtures
│   ├── test_chunk.py             # Chunking logic
│   ├── test_fetch.py             # Fetching / text cleaning
│   ├── test_store.py             # Abstract interface + factory (mocked backends)
│   ├── test_sparse.py            # BM25 search
│   ├── test_hybrid.py            # RRF merge + reranking logic
│   ├── test_generation.py        # Prompt building (no live Claude calls)
│   ├── test_api.py               # FastAPI routes (TestClient, mocked pipeline)
│   └── integration/
│       ├── test_chroma_store.py  # Integration: real ChromaDB (runs locally)
│       └── test_supabase_store.py # Integration: real Supabase (skipped without creds)
```

Add `pytest` and `pytest-mock` to `requirements.txt` (already included above).

Run unit tests only:
```bash
pytest tests/ -v --ignore=tests/integration
```

Run all including integration:
```bash
pytest tests/ -v
```

---

### What to Test Per Module

#### `tests/test_api.py`
Use FastAPI's `TestClient`. Mock the entire pipeline at the `hybrid.py` level so no
embedding model or vector store is touched:

```python
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_query_returns_200(mocker):
    mocker.patch("api.routes.query.run_pipeline", return_value={
        "answer": "Nietzsche argues...",
        "sources": [sample_chunk]
    })
    res = client.post("/query", json={"question": "What is the will to power?"})
    assert res.status_code == 200
    assert "answer" in res.json()

def test_query_missing_question_returns_422():
    res = client.post("/query", json={})
    assert res.status_code == 422

def test_ingest_requires_token():
    res = client.post("/ingest")
    assert res.status_code == 401

def test_ingest_accepts_valid_token(mocker):
    mocker.patch("api.routes.ingest.run_ingest", return_value=42)
    res = client.post("/ingest", headers={"X-Ingest-Token": "test-token"})
    assert res.status_code == 200
    assert res.json()["chunks_written"] == 42
```

#### `tests/test_store.py`
```python
def test_factory_returns_chroma_by_default(mocker):
    mocker.patch.dict(os.environ, {"VECTOR_STORE_BACKEND": "chroma"})
    store = get_vector_store()
    assert isinstance(store, ChromaStore)

def test_factory_returns_supabase_when_configured(mocker):
    mocker.patch.dict(os.environ, {"VECTOR_STORE_BACKEND": "supabase"})
    store = get_vector_store()
    assert isinstance(store, SupabaseStore)

def test_store_interface_enforced():
    # VectorStore cannot be instantiated directly (abstract)
    with pytest.raises(TypeError):
        VectorStore()
```

#### `tests/integration/test_supabase_store.py`
```python
import pytest, os

pytestmark = pytest.mark.skipif(
    not os.getenv("SUPABASE_URL"),
    reason="Supabase credentials not configured"
)

def test_store_and_retrieve_chunk():
    store = SupabaseStore()
    store.delete_all()
    store.store_chunks([sample_chunk], [mock_embedding])
    results = store.similarity_search(mock_embedding, top_k=1)
    assert results[0]["work_slug"] == "beyond_good_and_evil"
```

#### `tests/test_chunk.py`
This is the most important test file. Cover:

```python
# Aphorism detection
def test_aphorism_boundaries_detected():
    # Given a sample BGE-style text with numbered sections,
    # assert each aphorism becomes exactly one chunk

def test_short_aphorisms_merged():
    # Aphorisms under 50 tokens should be merged with the next one

def test_paragraph_chunking_respects_token_limit():
    # Essay-style text chunks should not exceed 300 tokens

def test_chunk_overlap_is_present():
    # Adjacent paragraph chunks should share ~50 tokens at boundaries

def test_metadata_fields_present():
    # Every chunk dict must have: work_title, work_slug, work_period,
    # section_number, chunk_index, chunk_type

def test_no_empty_chunks():
    # No chunk should have empty or whitespace-only content

def test_gutenberg_header_stripped():
    # Project Gutenberg boilerplate (before "*** START OF") must not
    # appear in any chunk
```

#### `tests/test_fetch.py`
```python
def test_gutenberg_boilerplate_removed():
    # Raw text with Gutenberg header/footer is cleaned correctly

def test_output_file_written(tmp_path):
    # Verify file is written to correct path with correct slug name
```

#### `tests/test_hybrid.py`
```python
def test_rrf_scores_sum_correctly():
    # Given known dense and sparse rankings, assert RRF output scores
    # match manual calculation: 1/(k+rank)

def test_rrf_deduplicates_results():
    # If the same chunk appears in both dense and sparse results,
    # it should appear only once in merged output

def test_rrf_ordering_is_descending():
    # Merged results must be sorted by RRF score, highest first

def test_reranker_reduces_to_top_n():
    # After reranking 10 results, output length == RERANK_TOP_N
```

#### `tests/test_generation.py`
```python
def test_prompt_contains_all_passages():
    # Given 3 chunks, all 3 appear in the built prompt

def test_prompt_contains_source_citations():
    # Each passage block in the prompt includes work_title and section_number

def test_prompt_contains_question():
    # The user question appears verbatim at the end of the prompt

def test_no_api_call_in_prompt_builder():
    # build_prompt() is a pure function — it must not call the API
    # (mock anthropic.Anthropic and assert it was never called)
```

---

### Fixtures (`conftest.py`)

Claude Code should populate `conftest.py` with reusable fixtures:

```python
import pytest

@pytest.fixture
def sample_aphoristic_text():
    """Minimal BGE-style text with 3 numbered aphorisms for chunking tests."""
    return """
1.
Supposing that Truth is a woman—what then?

2.
The will to truth which will still tempt us to many a venture.

3.
And if you gaze long into an abyss, the abyss also gazes into you.
"""

@pytest.fixture
def sample_chunks():
    """Three minimal chunk dicts matching the required metadata schema."""
    return [
        {
            "content": "Supposing that Truth is a woman—what then?",
            "work_title": "Beyond Good and Evil",
            "work_slug": "beyond_good_and_evil",
            "work_period": "late",
            "section_number": 1,
            "chunk_index": 0,
            "chunk_type": "aphorism",
        },
        # ... two more
    ]

@pytest.fixture
def mock_embeddings():
    """Returns deterministic fake 768-dim embeddings for two chunks."""
    import numpy as np
    return np.random.default_rng(42).random((2, 768)).tolist()
```

---

### Mocking External Dependencies

Never hit the real embedding model or Claude API in unit tests.
Never instantiate a real vector store — mock `get_vector_store()` instead.

**Mocking the vector store (backend-agnostic):**
```python
def test_embed_stores_chunks(mocker):
    mock_store = mocker.MagicMock()
    mocker.patch("ingest.embed.get_vector_store", return_value=mock_store)
    embed_chunks([sample_chunk])
    mock_store.store_chunks.assert_called_once()

def test_dense_search_calls_store(mocker):
    mock_store = mocker.MagicMock()
    mock_store.similarity_search.return_value = [sample_chunk]
    mocker.patch("retrieval.hybrid.get_vector_store", return_value=mock_store)
    results = dense_search(query_embedding=[0.1] * 768)
    assert len(results) == 1
```

**Mocking the embedding model:**
```python
def test_embed_calls_model(mocker):
    mocker.patch(
        "ingest.embed.SentenceTransformer.encode",
        return_value=[[0.1] * 768]
    )
    embed_chunks([sample_chunk])
```

**Mocking the Claude API:**
```python
def test_generation_returns_string(mocker):
    mocker.patch(
        "generation.claude.anthropic.Anthropic.messages.create",
        return_value=mocker.MagicMock(content=[mocker.MagicMock(text="Mocked answer.")])
    )
    result = generate_answer("What is the will to power?", sample_chunks)
    assert isinstance(result, str)
    assert len(result) > 0
```

---

### Claude Code Test Workflow

When Claude Code implements any module, the instruction should be:

> "Implement `X.py` and write `tests/test_X.py` at the same time.
> Tests must cover the cases listed in CLAUDE.md.
> Run `pytest tests/test_X.py -v` and confirm all tests pass before finishing."

If a test fails, Claude Code must fix the **implementation**, not weaken the test.
Deleting or skipping a failing test is not acceptable unless the test itself is provably wrong.