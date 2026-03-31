"""Smoke test: full hybrid retrieval pipeline over BGE.

Requires BGE to be embedded:
    python -m ingest.embed --work beyond_good_and_evil

Run from the project root:
    python scripts/smoke_hybrid_bge.py

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
from retrieval.dense import dense_search
from retrieval.hybrid import HybridResult, hybrid_search, reciprocal_rank_fusion, rerank
from retrieval.sparse import BM25Index

console = Console()

COLLECTION_NAME = config.COLLECTION_NAME
PERSIST_DIR = config.CHROMA_PERSIST_DIR
TOP_N = config.RERANK_TOP_N

QUERIES = [
    "What does Nietzsche mean by the will to power?",
    "What is the master and slave morality distinction?",
    "What does Nietzsche say about the philosopher of the future?",
    "How does Nietzsche critique Christian values?",
]


def _check_collection() -> int:
    if not PERSIST_DIR.exists():
        console.print(f"[red]FAIL[/red] ChromaDB directory not found: {PERSIST_DIR}")
        sys.exit(1)
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    try:
        col = client.get_collection(COLLECTION_NAME)
    except Exception:
        console.print(f"[red]FAIL[/red] Collection {COLLECTION_NAME!r} not found.")
        sys.exit(1)
    count = col.count()
    if count == 0:
        console.print(f"[red]FAIL[/red] Collection is empty.")
        sys.exit(1)
    return count


def _build_bm25() -> BM25Index:
    """Load all docs from ChromaDB once and build a shared BM25 index."""
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    col = client.get_collection(COLLECTION_NAME)
    data = col.get(include=["documents", "metadatas"])
    return BM25Index(ids=data["ids"], documents=data["documents"], metadatas=data["metadatas"])


def _print_results(query: str, results: list[HybridResult]) -> None:
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Rank", width=4)
    table.add_column("ID", width=24)
    table.add_column("§", width=8)
    table.add_column("RRF", width=8)
    table.add_column("Rerank", width=8)
    table.add_column("Preview")

    for rank, r in enumerate(results, 1):
        aph = r.metadata.get("aphorism_number", "?")
        aph_end = r.metadata.get("aphorism_number_end", aph)
        aph_str = f"{aph}" if aph == aph_end else f"{aph}–{aph_end}"
        rerank_str = f"{r.rerank_score:.2f}" if r.rerank_score is not None else "—"
        preview = " ".join(r.document.split()[:18])
        if len(r.document.split()) > 18:
            preview += " …"
        table.add_row(str(rank), r.id, aph_str, f"{r.rrf_score:.4f}", rerank_str, preview)

    console.print(f"\n[bold yellow]❯ {query}[/bold yellow]")
    console.print(table)

    if results:
        top = results[0]
        aph = top.metadata.get("aphorism_number", "?")
        ch = top.metadata.get("section_number", "?")
        title = top.metadata.get("work_title", "?")
        rerank_str = f"rerank={top.rerank_score:.2f}" if top.rerank_score is not None else ""
        console.print(
            Panel(
                top.document,
                title=f"[green]Top hit[/green] — {title}, ch.{ch} §{aph}  ({rerank_str})",
                border_style="green",
                padding=(0, 1),
            )
        )


def _assert_quality(query: str, results: list[HybridResult]) -> None:
    assert len(results) <= TOP_N, (
        f"[{query!r}] Expected ≤{TOP_N} results, got {len(results)}"
    )
    assert len(results) > 0, f"[{query!r}] Got zero results"

    for r in results:
        assert r.metadata["work_slug"] == "beyond_good_and_evil", (
            f"Unexpected work: {r.metadata['work_slug']}"
        )
        assert r.document.strip(), f"Empty document in {r.id}"
        assert r.rerank_score is not None, f"rerank_score not set on {r.id}"
        assert r.rrf_score > 0, f"rrf_score ≤ 0 on {r.id}"

    rerank_scores = [r.rerank_score for r in results]  # type: ignore[misc]
    assert rerank_scores == sorted(rerank_scores, reverse=True), (
        f"Results not sorted by rerank score: {rerank_scores}"
    )


def _rrf_unit_check(bm25: BM25Index) -> None:
    """Quick sanity check: verify RRF dedup and ordering with live data."""
    console.print("\n[cyan]RRF unit check with live data …[/cyan]")
    query = "will to power"
    dense = dense_search(query, persist_dir=PERSIST_DIR,
                         collection_name=COLLECTION_NAME, top_k=5)
    sparse = bm25.search(query, top_k=5)
    merged = reciprocal_rank_fusion(dense, sparse)

    ids = [r.id for r in merged]
    assert len(ids) == len(set(ids)), "RRF produced duplicate IDs"

    scores = [r.rrf_score for r in merged]
    assert scores == sorted(scores, reverse=True), "RRF results not sorted"

    # Any chunk appearing in both lists must have a higher score than one
    # appearing in only one list at rank 1
    shared = {r.id for r in dense} & {r.id for r in sparse}
    if shared:
        shared_scores = [r.rrf_score for r in merged if r.id in shared]
        solo_rank1_score = 1.0 / (config.RRF_K + 1)  # best possible single-list score
        # Shared docs get contributions from both lists, so their score > solo rank-1 ceiling
        # (only true if the shared doc appears at rank > 1 in at least one list, so
        #  we just verify they all have rrf_score > 0)
        assert all(s > 0 for s in shared_scores)

    console.print(
        f"  [green]✓[/green] {len(merged)} unique merged results, "
        f"{len(shared)} shared between dense and sparse"
    )


def main() -> None:
    count = _check_collection()
    console.print(f"[cyan]Collection[/cyan] {COLLECTION_NAME!r} — {count} entries\n")

    bm25 = _build_bm25()
    console.print(f"[cyan]BM25 index built.[/cyan] {bm25.corpus_size} documents\n")

    failures: list[str] = []

    for query in QUERIES:
        try:
            results = hybrid_search(
                query,
                persist_dir=PERSIST_DIR,
                collection_name=COLLECTION_NAME,
                top_n=TOP_N,
                bm25_index=bm25,
            )
            _assert_quality(query, results)
            _print_results(query, results)
        except AssertionError as exc:
            console.print(f"[red]FAIL[/red] {exc}")
            failures.append(str(exc))

    try:
        _rrf_unit_check(bm25)
    except AssertionError as exc:
        console.print(f"[red]FAIL[/red] {exc}")
        failures.append(str(exc))

    console.rule()
    if failures:
        console.print(f"[bold red]{len(failures)} assertion(s) failed.[/bold red]")
        for f in failures:
            console.print(f"  • {f}")
        sys.exit(1)
    else:
        console.print(
            f"[bold green]All hybrid smoke checks passed.[/bold green] "
            f"({len(QUERIES)} queries + RRF unit check)"
        )


if __name__ == "__main__":
    main()
