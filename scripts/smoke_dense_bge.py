"""Smoke test: run dense retrieval over the embedded BGE collection.

Requires that BGE has already been embedded:
    python -m ingest.embed --work beyond_good_and_evil

Run from the project root:
    python scripts/smoke_dense_bge.py

Exits with a non-zero code if any assertion fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import config
from retrieval.dense import dense_search, DenseResult

console = Console()

# Queries chosen to test topically relevant retrieval
QUERIES = [
    "What does Nietzsche mean by the will to power?",
    "What is the master-slave morality distinction?",
    "How does Nietzsche view truth and knowledge?",
    "What is Nietzsche's critique of Christianity?",
    "What does Nietzsche say about the philosopher of the future?",
]

TOP_K = 5
COLLECTION_NAME = config.COLLECTION_NAME
PERSIST_DIR = config.CHROMA_PERSIST_DIR


def _check_collection_populated() -> int:
    """Return collection count; fail fast if it is empty."""
    import chromadb
    if not PERSIST_DIR.exists():
        console.print(f"[red]FAIL[/red] ChromaDB directory not found: {PERSIST_DIR}")
        console.print("  Run: python -m ingest.embed --work beyond_good_and_evil")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    try:
        col = client.get_collection(COLLECTION_NAME)
    except Exception:
        console.print(f"[red]FAIL[/red] Collection {COLLECTION_NAME!r} not found in {PERSIST_DIR}")
        console.print("  Run: python -m ingest.embed --work beyond_good_and_evil")
        sys.exit(1)

    count = col.count()
    if count == 0:
        console.print(f"[red]FAIL[/red] Collection {COLLECTION_NAME!r} is empty")
        sys.exit(1)
    return count


def _print_results(query: str, results: list[DenseResult]) -> None:
    """Print a rich table of results followed by the full text of the top hit."""
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Rank", width=4)
    table.add_column("ID", width=22)
    table.add_column("§", width=8)
    table.add_column("Dist", width=6)
    table.add_column("Preview", no_wrap=False)

    for rank, r in enumerate(results, 1):
        aph = r.metadata.get("aphorism_number", "?")
        aph_end = r.metadata.get("aphorism_number_end", aph)
        aph_str = f"{aph}" if aph == aph_end else f"{aph}–{aph_end}"
        # Show the first 25 words as a preview
        preview = " ".join(r.document.split()[:25])
        if len(r.document.split()) > 25:
            preview += " …"
        table.add_row(str(rank), r.id, aph_str, f"{r.distance:.3f}", preview)

    console.print(f"\n[bold yellow]❯ {query}[/bold yellow]")
    console.print(table)

    # Print the full text of the top result
    top = results[0]
    aph = top.metadata.get("aphorism_number", "?")
    ch = top.metadata.get("section_number", "?")
    title = top.metadata.get("work_title", "?")
    console.print(
        Panel(
            top.document,
            title=f"[green]Top hit[/green] — {title}, ch.{ch} §{aph}  (dist {top.distance:.3f})",
            border_style="green",
            padding=(0, 1),
        )
    )


def _assert_result_quality(query: str, results: list[DenseResult]) -> None:
    assert len(results) == TOP_K, (
        f"Expected {TOP_K} results for {query!r}, got {len(results)}"
    )
    for r in results:
        assert r.metadata["work_slug"] == "beyond_good_and_evil", (
            f"Unexpected work: {r.metadata['work_slug']}"
        )
        assert r.document.strip(), f"Empty document in result {r.id}"
        assert 0.0 <= r.distance <= 2.0, (
            f"Suspicious distance {r.distance} for result {r.id}"
        )
    dists = [r.distance for r in results]
    assert dists == sorted(dists), f"Results not sorted by distance: {dists}"
    # ChromaDB defaults to L2 distance; values below 1.5 indicate a
    # meaningful semantic match (all-mpnet-base-v2 embeddings are unit-normalised,
    # so L2 ~ 2*(1 - cos_sim), giving ~0–2 range)
    assert results[0].distance < 1.5, (
        f"Top result distance {results[0].distance:.3f} looks too high for {query!r}"
    )


def main() -> None:
    count = _check_collection_populated()
    console.print(
        f"[cyan]Collection[/cyan] {COLLECTION_NAME!r} — {count} entries   "
        f"[cyan]top_k[/cyan] = {TOP_K}\n"
    )

    failures: list[str] = []

    for query in QUERIES:
        try:
            results = dense_search(
                query,
                persist_dir=PERSIST_DIR,
                collection_name=COLLECTION_NAME,
                top_k=TOP_K,
            )
            _assert_result_quality(query, results)
            _print_results(query, results)
        except AssertionError as exc:
            console.print(f"[red]FAIL[/red] {exc}")
            failures.append(str(exc))

    # ── metadata filter check ─────────────────────────────────────────────────
    console.print("\n[cyan]Testing metadata filter (work_period = late) …[/cyan]")
    try:
        filtered = dense_search(
            "What is noble?",
            persist_dir=PERSIST_DIR,
            collection_name=COLLECTION_NAME,
            top_k=3,
            where={"work_period": "late"},
        )
        assert all(r.metadata["work_period"] == "late" for r in filtered), \
            "Filter did not restrict to 'late' period"
        console.print(f"  [green]✓[/green] filter returned {len(filtered)} results, all 'late'")
    except AssertionError as exc:
        console.print(f"[red]FAIL[/red] {exc}")
        failures.append(str(exc))

    # ── summary ───────────────────────────────────────────────────────────────
    console.rule()
    if failures:
        console.print(f"[bold red]{len(failures)} assertion(s) failed.[/bold red]")
        for f in failures:
            console.print(f"  • {f}")
        sys.exit(1)
    else:
        console.print(
            f"[bold green]All dense retrieval smoke checks passed.[/bold green] "
            f"({len(QUERIES)} queries + 1 filter check)"
        )


if __name__ == "__main__":
    main()
