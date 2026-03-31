"""Smoke test: chunk BGE, embed it, and verify ChromaDB has the expected entries.

Run from the project root:
    python scripts/smoke_embed_bge.py

The script uses a temporary directory for ChromaDB so it never pollutes
the real data/chroma store. It exits with a non-zero code if any assertion fails.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from rich.console import Console

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingest.chunk import chunk_work
from ingest.embed import embed_chunks, get_chroma_collection

RAW = Path("data/raw/beyond_good_and_evil.txt")
WORK_TITLE = "Beyond Good and Evil"
WORK_SLUG = "beyond_good_and_evil"
WORK_PERIOD = "late"
CHUNK_STYLE = "aphorism"
COLLECTION_NAME = "bge_smoke"
MIN_EXPECTED_CHUNKS = 200  # BGE has 296 numbered aphorisms

console = Console()


def main() -> None:
    # ── Step 1: verify raw file exists ────────────────────────────────────────
    if not RAW.exists():
        console.print(f"[red]FAIL[/red] Raw file not found: {RAW}")
        console.print("  Run: python -m ingest.fetch   (or python app.py ingest)")
        sys.exit(1)

    console.print(f"[cyan]Raw file:[/cyan] {RAW}  ({RAW.stat().st_size // 1024} KB)")

    # ── Step 2: chunk ──────────────────────────────────────────────────────────
    console.print("\n[cyan]Chunking …[/cyan]")
    chunks = chunk_work(
        RAW.read_text(encoding="utf-8"),
        WORK_TITLE, WORK_SLUG, WORK_PERIOD, CHUNK_STYLE,
    )
    console.print(f"  chunks produced : {len(chunks)}")
    assert len(chunks) >= MIN_EXPECTED_CHUNKS, (
        f"Expected ≥{MIN_EXPECTED_CHUNKS} chunks, got {len(chunks)}"
    )
    console.print(f"  [green]✓[/green] ≥{MIN_EXPECTED_CHUNKS} chunks")

    # ── Step 3: embed into a temp ChromaDB ────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="nietzsche_smoke_") as tmpdir:
        persist_dir = Path(tmpdir)
        console.print(f"\n[cyan]Embedding into temp ChromaDB[/cyan] {persist_dir}")

        n = embed_chunks(
            chunks,
            persist_dir=persist_dir,
            collection_name=COLLECTION_NAME,
        )
        assert n == len(chunks), f"embed_chunks returned {n}, expected {len(chunks)}"
        console.print(f"  [green]✓[/green] embed_chunks returned {n}")

        # ── Step 4: verify collection count ───────────────────────────────────
        console.print("\n[cyan]Verifying ChromaDB collection …[/cyan]")
        col = get_chroma_collection(persist_dir, COLLECTION_NAME)
        db_count = col.count()
        console.print(f"  collection count: {db_count}")
        assert db_count == len(chunks), (
            f"ChromaDB has {db_count} entries, expected {len(chunks)}"
        )
        console.print(f"  [green]✓[/green] collection count matches chunk count")

        # ── Step 5: spot-check a few entries ──────────────────────────────────
        console.print("\n[cyan]Spot-checking 3 entries …[/cyan]")
        sample = col.get(
            ids=[f"{WORK_SLUG}_chunk_{i}" for i in (0, len(chunks) // 2, len(chunks) - 1)],
            include=["documents", "metadatas", "embeddings"],
        )
        assert len(sample["ids"]) == 3, "Expected 3 sample entries"
        for doc, meta, emb in zip(
            sample["documents"], sample["metadatas"], sample["embeddings"]
        ):
            assert doc.strip(), "Document text is empty"
            assert meta["work_slug"] == WORK_SLUG
            assert meta["work_period"] == WORK_PERIOD
            assert isinstance(meta["section_number"], int)
            assert len(emb) == 768, f"Embedding dim should be 768, got {len(emb)}"
        console.print("  [green]✓[/green] documents, metadata, and 768-dim embeddings present")

        # ── Step 6: idempotency check — re-embed should not grow the collection
        console.print("\n[cyan]Idempotency check (re-embed) …[/cyan]")
        embed_chunks(chunks, persist_dir=persist_dir, collection_name=COLLECTION_NAME)
        assert col.count() == db_count, (
            f"Collection grew after re-embed: {col.count()} != {db_count}"
        )
        console.print(f"  [green]✓[/green] upsert is idempotent — count still {db_count}")

    console.rule()
    console.print(
        f"[bold green]All smoke checks passed.[/bold green] "
        f"{n} chunks embedded for {WORK_TITLE!r}."
    )


if __name__ == "__main__":
    main()
