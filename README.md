# Nietzsche RAG

A Retrieval-Augmented Generation system over the complete Nietzsche corpus. Ask philosophical questions; get grounded, cited answers drawn from the primary texts.

**Live:** [nietzsche-rag.fly.dev](https://nietzsche-rag.fly.dev) (API) · frontend on Vercel

---

## What it does

- Retrieves the most relevant passages from 12 Nietzsche works using **hybrid search** (dense vectors + BM25 keyword search, merged with Reciprocal Rank Fusion, re-ranked with a cross-encoder)
- Generates answers with **Claude** that cite every claim back to a specific work and section number, e.g. `[BGE §36]`
- Exposes the pipeline as a **FastAPI** backend deployed on Fly.io
- Chat interface built with **Next.js 14** showing the source passages inline

---

## Architecture

```
Query
  │
  ├─ [Optional] HyDE: generate hypothetical Nietzsche passage via Claude Haiku,
  │   embed that instead of the raw question for better semantic alignment
  │
  ├─ [Optional] Multi-query: generate 2 paraphrase variants via Claude Haiku,
  │   run dense search for each and union all candidate lists
  │
  ├─ Dense retrieval ×(1+variants) → top 20 each (all-mpnet-base-v2 → Supabase pgvector)
  ├─ Sparse retrieval (BM25 via rank_bm25, in-memory) → top 20
  │
  ▼
RRF merge (Reciprocal Rank Fusion, k=60)
  │
  ▼
Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) → top 7
  + aphorism bonus (+1.5): specific aphorisms ranked above broad prose when scores are close
  │
  ▼
Claude (claude-sonnet-4-5) with citation-enforcing system prompt
  │
  ▼
Response: answer + source passages with metadata
```

---

## Corpus

12 works across Nietzsche's three periods, sourced from Project Gutenberg:

| Period | Works |
|--------|-------|
| Early | The Birth of Tragedy, Untimely Meditations |
| Middle | The Gay Science, Dawn, Human All Too Human, Thus Spoke Zarathustra |
| Late | Beyond Good and Evil, On the Genealogy of Morality, Twilight of the Idols, The Antichrist, Ecce Homo, Nietzsche contra Wagner |

Aphoristic works (BGE, GS, Dawn, etc.) are chunked by numbered section. Essay/prose works are chunked by paragraph (~300 tokens, 100-token overlap). Total: **~2,800 chunks** in production.

---

## Project structure

```
nietzsche-rag/
├── ingest/
│   ├── fetch.py              # Download texts from Project Gutenberg
│   ├── chunk.py              # Aphorism-aware and paragraph-aware chunking
│   └── embed.py              # Embed chunks and write to vector store
├── retrieval/
│   ├── store.py              # Abstract VectorStore interface + factory
│   ├── chroma_store.py       # ChromaDB implementation (local dev)
│   ├── supabase_store.py     # Supabase + pgvector implementation (production)
│   ├── dense.py              # Dense search with optional embed_text override
│   ├── sparse.py             # BM25 keyword search
│   ├── hyde.py               # HyDE: hypothetical passage generation via Claude
│   ├── multiquery.py         # Multi-query: paraphrase variants via Claude
│   └── hybrid.py             # RRF merge + cross-encoder reranking + aphorism bonus
├── generation/
│   └── claude.py             # Prompt builder + Claude API call
├── api/
│   ├── main.py               # FastAPI app: routers, CORS, lifespan model loading
│   ├── routes/
│   │   ├── query.py          # POST /query
│   │   └── ingest.py         # POST /ingest (token-protected, runs in background)
│   ├── models.py             # Pydantic request/response schemas
│   └── dependencies.py       # Auth header check
├── frontend/                 # Next.js 14 chat UI
│   ├── app/
│   │   ├── page.tsx          # Main chat page
│   │   └── api/query/route.ts # Proxy route → FastAPI
│   ├── components/
│   │   ├── ChatInput.tsx
│   │   ├── FilterBar.tsx     # Period + work dropdowns
│   │   ├── MessageList.tsx
│   │   └── SourceCard.tsx    # Cited passage card
│   └── types.ts
├── tests/
│   ├── test_chunk.py
│   ├── test_fetch.py
│   ├── test_store.py
│   ├── test_sparse.py
│   ├── test_hybrid.py
│   ├── test_generation.py
│   ├── test_api.py
│   └── integration/
│       ├── test_chroma_store.py
│       └── test_supabase_store.py
├── app.py                    # CLI (typer + rich)
├── config.py                 # Central config: paths, models, hyperparameters
├── Dockerfile
├── fly.toml
└── .env.example
```

---

## Local setup

### Prerequisites

- Python 3.11+
- Node.js 18+ (for the frontend)
- A free [Anthropic API key](https://console.anthropic.com/)

### Backend

```bash
# Clone and create a virtual environment
git clone https://github.com/QuantumSpinozist/NietzscheRAG
cd NietzscheRAG
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY at minimum

# Ingest the corpus (downloads texts + embeds into local ChromaDB)
python app.py ingest --all

# Start the API
uvicorn api.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend
npm install

# Point at the local backend
echo "FASTAPI_URL=http://localhost:8000" > .env.local

npm run dev
```

Open `http://localhost:3000`.

---

## CLI usage

```bash
# Ask a question
python app.py query "What does Nietzsche mean by the will to power?"

# Filter by period
python app.py query "What is eternal recurrence?" --period late

# Filter by specific work
python app.py query "How does Nietzsche view Socrates?" --work twilight_of_the_idols

# Ingest a single work
python app.py ingest --work beyond_good_and_evil

# Ingest everything
python app.py ingest --all
```

---

## API

### `POST /query`

```json
{
  "question": "What is the will to power?",
  "filter_period": "late",
  "filter_slug": null,
  "use_hyde": false
}
```

`use_hyde: true` generates a hypothetical Nietzsche-style passage via Claude Haiku before dense retrieval — improves recall on abstract philosophical queries at the cost of one extra fast LLM call.

Response:

```json
{
  "answer": "Nietzsche describes the will to power as... [BGE §36]",
  "sources": [
    {
      "work_title": "Beyond Good and Evil",
      "work_slug": "beyond_good_and_evil",
      "section_number": 36,
      "chunk_type": "aphorism",
      "content": "Supposing that nothing else is...",
      "similarity": 0.87
    }
  ]
}
```

### `POST /ingest`

Protected by `X-Ingest-Token` header. Triggers a full corpus re-ingest as a background task.

```bash
curl -X POST https://nietzsche-rag.fly.dev/ingest \
  -H "X-Ingest-Token: your_token_here"
```

Returns `{"status": "started"}` immediately; embedding runs in the background.

### `GET /health`

Liveness probe. Returns `{"status": "ok"}`.

---

## Vector store backends

The app uses an abstract `VectorStore` interface — the backend is selected via `VECTOR_STORE_BACKEND` in `.env`:

| Backend | Use case | Config |
|---------|----------|--------|
| `chroma` (default) | Local development | No credentials needed; data persists to `./data/chroma` |
| `supabase` | Production | Requires `SUPABASE_URL` + `SUPABASE_SERVICE_KEY` |

Supabase requires a `chunks` table with a `vector(768)` column and an HNSW index. The full SQL schema is in [CLAUDE.md](CLAUDE.md).

---

## Tests

```bash
# Unit tests only (no embedding model, no external services)
pytest tests/ -v --ignore=tests/integration

# All tests including integration (requires Supabase credentials or local ChromaDB)
pytest tests/ -v
```

### Retrieval eval

A 10-question eval set with ground-truth passage annotations lives in `eval/`:

```bash
# Baseline hybrid retrieval
python eval/run_eval.py

# With BM25 synonym expansion
python eval/run_eval.py --synonyms

# With HyDE query expansion
python eval/run_eval.py --hyde

# With multi-query paraphrase expansion
python eval/run_eval.py --multiquery

# With aphorism reranker bonus
python eval/run_eval.py --aphorism-bonus 1.5
```

Current results (Supabase backend, 15-question eval set):

| Config | HR@5 | MRR@5 | HR@10 | MRR@10 |
|--------|------|-------|-------|--------|
| Old baseline (k=10, bonus=0) | 60% | 0.533 | 80% | 0.560 |
| + aphorism bonus (1.5), k=10 | 70% | 0.558 | 80% | 0.573 |
| **Current (k=20, bonus=1.5)** | **80%** | **0.628** | **93.3%** | **0.637** |

The 15-question eval covers: death of God, eternal recurrence, master/slave morality, will to power, amor fati, consciousness, revaluation of values, free will, origin of knowledge, nobility and pathos of distance, great suffering, perspectivism, and more. One persistent miss (BGE §211/212 "philosopher of the future") requires the target aphorism to enter the dense candidate pool — a known limitation of the current embedding model on that specific query.

---

## Deployment

### Backend (Fly.io)

```bash
fly launch          # first time
fly secrets set \
  ANTHROPIC_API_KEY=... \
  VECTOR_STORE_BACKEND=supabase \
  SUPABASE_URL=... \
  SUPABASE_SERVICE_KEY=... \
  INGEST_TOKEN=... \
  ALLOWED_ORIGINS=https://your-app.vercel.app
fly deploy
```

The Docker image pre-downloads the embedding model at build time (~420 MB). The VM needs at least 1 GB of memory. A 3-minute health-check grace period is configured in `fly.toml` to allow for model loading on cold start.

Continuous deployment via GitHub Actions is configured in `.github/workflows/fly-deploy.yml`. Add your `FLY_API_TOKEN` as a repository secret:

```bash
fly tokens create deploy -x 999999h
# → add as FLY_API_TOKEN in GitHub repo Settings → Secrets
```

### Frontend (Vercel)

```bash
cd frontend
npx vercel
npx vercel env add FASTAPI_URL   # → https://nietzsche-rag.fly.dev
npx vercel --prod
```

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Embeddings | `sentence-transformers/all-mpnet-base-v2` (768-dim, local) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` + aphorism score bonus |
| Query expansion | HyDE via Claude Haiku (optional, per-request) |
| Generation | Claude (claude-sonnet-4-5) via Anthropic SDK |
| Vector store (local) | ChromaDB |
| Vector store (production) | Supabase + pgvector |
| Keyword search | rank_bm25 |
| API backend | FastAPI + uvicorn, deployed on Fly.io |
| Frontend | Next.js 14, TypeScript, Tailwind CSS, deployed on Vercel |
| CLI | typer + rich |
