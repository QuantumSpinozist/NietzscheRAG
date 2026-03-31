"""Smoke test: run BM25 retrieval over BGE corpus loaded from ChromaDB.

Requires that BGE has already been embedded:
    python -m ingest.embed --work beyond_good_and_evil

Run from the project root:
    python scripts/smoke_sparse_bge.py

Exits with a non-zero code if any assertion fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config
from retrieval.sparse import BM25Index, SparseResult

console = Console()

COLLECTION_NAME = config.COLLECTION_NAME
PERSIST_DIR = config.CHROMA_PERSIST_DIR
TOP_K = 5

# Queries: keyword-heavy terms that appear in the BGE translation
QUERIES = [
    ("Will to power", "will power"),
    ("Master and slave morality", "master slave morality"),
    ("Eternal recurrence", "eternal recurrence"),
    ("Schopenhauer", "schopenhauer"),
    ("Truth and truthfulness", "truth truthfulness philosophers"),
]


def _load_corpus() -> tuple[list[str], list[str], list[dict]]:
    """Fetch all documents from ChromaDB collection."""
    if not PERSIST_DIR.exists():
        console.print(f"[red]FAIL[/red] ChromaDB directory not found: {PERSIST_DIR}")
        console.print("  Run: python -m ingest.embed --work beyond_good_and_evil")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    try:
        col = client.get_collection(COLLECTION_NAME)
    except Exception:
        console.print(f"[red]FAIL[/red] Collection {COLLECTION_NAME!r} not found.")
        sys.exit(1)

    count = col.count()
    if count == 0:
        console.print(f"[red]FAIL[/red] Collection {COLLECTION_NAME!r} is empty.")
        sys.exit(1)

    # Fetch all documents in one call
    data = col.get(include=["documents", "metadatas"])
    return data["ids"], data["documents"], data["metadatas"]


def _print_results(label: str, results: list[SparseResult]) -> None:
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Rank", width=4)
    table.add_column("ID", width=24)
    table.add_column("§", width=8)
    table.add_column("Score", width=7)
    table.add_column("Preview")

    for rank, r in enumerate(results, 1):
        aph = r.metadata.get("aphorism_number", "?")
        aph_end = r.metadata.get("aphorism_number_end", aph)
        aph_str = f"{aph}" if aph == aph_end else f"{aph}–{aph_end}"
        preview = " ".join(r.document.split()[:20])
        if len(r.document.split()) > 20:
            preview += " …"
        table.add_row(str(rank), r.id, aph_str, f"{r.score:.3f}", preview)

    console.print(f"\n[bold yellow]❯ {label}[/bold yellow]")
    console.print(table)

    if results:
        top = results[0]
        aph = top.metadata.get("aphorism_number", "?")
        ch = top.metadata.get("section_number", "?")
        title = top.metadata.get("work_title", "?")
        console.print(
            Panel(
                top.document,
                title=f"[green]Top hit[/green] — {title}, ch.{ch} §{aph}  (score {top.score:.3f})",
                border_style="green",
                padding=(0, 1),
            )
        )
    else:
        console.print("  [dim](no results — term not in corpus)[/dim]")


def _assert_result_quality(label: str, results: list[SparseResult]) -> None:
    # All results must come from BGE (only indexed work)
    for r in results:
        assert r.metadata["work_slug"] == "beyond_good_and_evil", (
            f"[{label}] Unexpected work: {r.metadata['work_slug']}"
        )
        assert r.document.strip(), f"[{label}] Empty document in result {r.id}"

    # Scores must be in descending order
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), (
        f"[{label}] Results not sorted by score: {scores}"
    )


def main() -> None:
    ids, documents, metadatas = _load_corpus()
    console.print(
        f"[cyan]Corpus loaded:[/cyan] {len(ids)} documents from "
        f"{COLLECTION_NAME!r}   [cyan]top_k[/cyan] = {TOP_K}\n"
    )

    index = BM25Index(ids=ids, documents=documents, metadatas=metadatas)
    console.print(f"[cyan]BM25 index built.[/cyan] Corpus size: {index.corpus_size}\n")

    failures: list[str] = []

    for label, query in QUERIES:
        try:
            results = index.search(query, top_k=TOP_K)
            _assert_result_quality(label, results)
            _print_results(label, results)
        except AssertionError as exc:
            console.print(f"[red]FAIL[/red] {exc}")
            failures.append(str(exc))

    # ── keyword specificity check ─────────────────────────────────────────────
    console.print("\n[cyan]Keyword specificity check …[/cyan]")
    try:
        # "Schopenhauer" appears in a handful of aphorisms; top result must contain the word
        results = index.search("Schopenhauer", top_k=3)
        assert results, "Expected at least one result for 'Schopenhauer'"
        top_doc = results[0].document.lower()
        assert "schopenhauer" in top_doc, (
            f"Top result for 'Schopenhauer' doesn't contain the word:\n{results[0].document[:200]}"
        )
        console.print(f"  [green]✓[/green] 'Schopenhauer' → top hit (§{results[0].metadata.get('aphorism_number')}) contains the term")
    except AssertionError as exc:
        console.print(f"[red]FAIL[/red] {exc}")
        failures.append(str(exc))

    # ── absent term returns empty ─────────────────────────────────────────────
    console.print("\n[cyan]Absent-term check …[/cyan]")
    # Use a hex string that cannot tokenise to any real word
    absent_results = index.search("zz7f3b91ae204c6d")
    assert absent_results == [], f"Expected no results for nonsense query, got {absent_results}"
    console.print("  [green]✓[/green] nonsense query returns no results")

    # ── summary ───────────────────────────────────────────────────────────────
    console.rule()
    if failures:
        console.print(f"[bold red]{len(failures)} assertion(s) failed.[/bold red]")
        for f in failures:
            console.print(f"  • {f}")
        sys.exit(1)
    else:
        console.print(
            f"[bold green]All BM25 smoke checks passed.[/bold green] "
            f"({len(QUERIES)} queries + 2 extra checks)"
        )


if __name__ == "__main__":
    main()
