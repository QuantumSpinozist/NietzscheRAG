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
│   └── raw/              # Plain .txt files, one per work (sourced from Gutenberg)
├── ingest/
│   ├── fetch.py          # Download texts from Project Gutenberg  ✓
│   ├── chunk.py          # Aphorism-aware and paragraph-aware chunking
│   └── embed.py          # Embed chunks and store in ChromaDB
├── retrieval/
│   ├── dense.py          # Semantic vector search via ChromaDB
│   ├── sparse.py         # BM25 keyword search via rank_bm25
│   └── hybrid.py         # Merge dense + sparse results, then rerank
├── generation/
│   └── claude.py         # Build prompt and call Claude API
├── tests/
│   ├── conftest.py       # Shared fixtures (sample chunks, mock embeddings, etc.)
│   ├── test_fetch.py     # Fetching / download tests  ✓
│   ├── test_chunk.py     # Chunking logic
│   ├── test_dense.py     # Vector search (mocked ChromaDB)
│   ├── test_sparse.py    # BM25 search
│   ├── test_hybrid.py    # RRF merge + reranking logic
│   └── test_generation.py  # Prompt building (no live Claude calls)
├── app.py                # CLI entry point (typer + rich)
├── config.py             # Central config (paths, model names, hyperparams)
├── requirements.txt
└── CLAUDE.md             # This file
```

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Data source | Project Gutenberg | Plain text, public domain |
| Chunking | Custom Python | Aphorism-aware (see below) |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` | Local, free, 768-dim |
| Vector store | `chromadb` | Local persistent storage |
| Keyword search | `rank_bm25` | Hybrid retrieval complement |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Top-10 → Top-3 refinement |
| Generation | Claude (via Anthropic SDK) | See generation instructions |
| CLI | `typer` + `rich` | Pretty terminal interface |

---

## Corpus

Target works to ingest (in rough priority order):

**Late period (highest priority):**
- Beyond Good and Evil (BGE)
- On the Genealogy of Morality (GM)
- Twilight of the Idols (TI)
- The Antichrist (AC)
- Ecce Homo (EH)
- Nietzsche contra Wagner

**Middle period:**
- The Gay Science (GS)
- Dawn / Daybreak
- Human, All Too Human

**Early period:**
- The Birth of Tragedy (BT)
- Untimely Meditations

Store each work as a `.txt` file in `data/raw/` named by slug, e.g. `beyond_good_and_evil.txt`.

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
Every chunk must carry this metadata in ChromaDB:
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

## Retrieval

Use hybrid retrieval — do not rely on dense search alone. Nietzsche uses coined terms
(*Ressentiment*, *Übermensch*, *Umwertung*) that keyword search captures better.

### Pipeline
1. Run dense search → top 10 results from ChromaDB
2. Run BM25 search → top 10 results from in-memory index
3. Merge with Reciprocal Rank Fusion (RRF): `score = 1 / (k + rank)` where k=60
4. Re-rank merged top 10 with cross-encoder → return top 3–5

### Configurable parameters (in `config.py`)
```python
DENSE_TOP_K = 10
SPARSE_TOP_K = 10
RERANK_TOP_N = 5
RRF_K = 60
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_PERSIST_DIR = "./data/chroma"
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
sentence-transformers
rank_bm25
torch                        # required by sentence-transformers
typer
rich
requests                     # for Gutenberg fetching
python-dotenv
pytest
pytest-mock
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
```

Load with `python-dotenv` in `config.py`. Never hardcode the key.

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

1. `ingest/fetch.py` — download and save raw texts
2. `ingest/chunk.py` — chunking logic (test on one work first)
3. `ingest/embed.py` — embed chunks, persist to ChromaDB
4. `retrieval/dense.py` — basic vector search working end-to-end
5. `retrieval/sparse.py` — BM25 index over same chunks
6. `retrieval/hybrid.py` — RRF merge + reranker
7. `generation/claude.py` — prompt builder + Claude call
8. `app.py` — wire everything into CLI

Do not move to step N+1 until step N has a passing smoke test.

---

## Code Style

- Type hints on all function signatures
- Docstrings on all public functions
- No global state — pass config explicitly
- Log progress to stderr with `rich` (not print statements)
- Raise specific exceptions, never bare `except:`

---

## Testing

### Philosophy

Every module gets a test file written **at the same time** as the implementation — not after.
When Claude Code writes `ingest/chunk.py`, it must also write `tests/test_chunk.py` in the same step.
Tests are not optional polish; they are how we verify the pipeline before wiring modules together.

Do not use the live ChromaDB, live embedding model, or the Claude API in unit tests.
Use fixtures and mocks for anything with I/O or external dependencies.

---

### Test Structure

```
nietzsche-rag/
├── tests/
│   ├── conftest.py           # Shared fixtures (sample chunks, mock embeddings, etc.)
│   ├── test_chunk.py         # Chunking logic
│   ├── test_fetch.py         # Fetching / text cleaning
│   ├── test_dense.py         # Vector search (mocked ChromaDB)
│   ├── test_sparse.py        # BM25 search
│   ├── test_hybrid.py        # RRF merge + reranking logic
│   └── test_generation.py    # Prompt building (no live Claude calls)
```

Add `pytest` and `pytest-mock` to `requirements.txt`.

Run all tests with:
```bash
pytest tests/ -v
```

---

### What to Test Per Module

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

Never hit the real embedding model, ChromaDB, or Claude API in unit tests.

**Mocking the embedding model:**
```python
def test_embed_calls_model(mocker):
    mock_encode = mocker.patch(
        "ingest.embed.SentenceTransformer.encode",
        return_value=[[0.1] * 768]
    )
    embed_chunks([sample_chunk])
    mock_encode.assert_called_once()
```

**Mocking ChromaDB:**
```python
def test_chunks_added_to_collection(mocker):
    mock_collection = mocker.MagicMock()
    mocker.patch("ingest.embed.get_chroma_collection", return_value=mock_collection)
    embed_chunks([sample_chunk])
    mock_collection.add.assert_called_once()
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