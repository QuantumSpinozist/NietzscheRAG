"""Retrieval evaluation: Hit Rate and MRR at k=5 and k=10.

Run from project root:
    python eval/run_eval.py                  # baseline (no synonym expansion)
    python eval/run_eval.py --synonyms       # with synonym expansion

Requires Supabase credentials in .env (VECTOR_STORE_BACKEND=supabase).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from eval.eval_set import EVAL_SET, EvalItem
from retrieval.hybrid import HybridResult, hybrid_search
from retrieval.sparse import BM25Index
from retrieval.store import get_vector_store

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
    err = Console(stderr=True)
except ImportError:
    import types
    console = types.SimpleNamespace(print=print, rule=lambda *a, **k: print("---"))  # type: ignore


K_VALS = [5, 10]  # cutoffs to evaluate at
_MAX_K = max(K_VALS)


# ── helpers ───────────────────────────────────────────────────────────────────


def _hits(results: list[HybridResult], item: EvalItem, k: int) -> list[int]:
    """Return 1-indexed ranks at which a ground-truth chunk appears (up to k)."""
    found = []
    for rank, r in enumerate(results[:k], start=1):
        slug = r.metadata.get("work_slug")
        aph = r.metadata.get("aphorism_number")
        for gt_slug, gt_aph in item.ground_truth:
            if slug == gt_slug and aph == gt_aph:
                found.append(rank)
    return found


def _reciprocal_rank(hit_ranks: list[int]) -> float:
    return 1.0 / min(hit_ranks) if hit_ranks else 0.0


# ── main ──────────────────────────────────────────────────────────────────────


def run(use_synonyms: bool = False) -> None:
    label = "WITH synonym expansion" if use_synonyms else "baseline (no synonyms)"
    console.print(f"\n[bold cyan]Retrieval eval — {label}[/bold cyan]\n")

    # Load corpus once for BM25
    t0 = time.time()
    console.print("[cyan]Loading corpus for BM25 index …[/cyan]")
    store = get_vector_store()
    data = store.get_all_documents()
    bm25 = BM25Index(
        ids=data["ids"],
        documents=data["documents"],
        metadatas=data["metadatas"],
        use_synonyms=use_synonyms,
    )
    console.print(
        f"[green]✓[/green] BM25 index built: {bm25.corpus_size} docs "
        f"({time.time() - t0:.1f}s)\n"
    )

    # Load embedding/reranker models once
    from sentence_transformers import CrossEncoder, SentenceTransformer
    console.print("[cyan]Loading embedding + reranker models …[/cyan]")
    t1 = time.time()
    st = SentenceTransformer(config.EMBEDDING_MODEL)
    ce = CrossEncoder(config.RERANKER_MODEL)
    console.print(f"[green]✓[/green] Models loaded ({time.time() - t1:.1f}s)\n")

    # Eval loop — retrieve _MAX_K results so we can compute all cutoffs
    results_table = Table(show_header=True, header_style="bold", expand=True)
    results_table.add_column("#", width=3)
    results_table.add_column("Question", width=46)
    results_table.add_column("Hit@5", width=6)
    results_table.add_column("Hit@10", width=7)
    results_table.add_column("Rank", width=5)
    results_table.add_column("RR", width=6)
    results_table.add_column("Top retrieved")

    # Accumulators keyed by cutoff k
    hit_counts: dict[int, int] = {k: 0 for k in K_VALS}
    rr_totals: dict[int, float] = {k: 0.0 for k in K_VALS}
    detail_rows: list[dict] = []

    for i, item in enumerate(EVAL_SET, start=1):
        t_q = time.time()
        retrieved = hybrid_search(
            item.question,
            top_n=_MAX_K,
            bm25_index=bm25,
            sentence_transformer=st,
            cross_encoder=ce,
        )
        elapsed = time.time() - t_q

        per_k: dict[int, list[int]] = {k: _hits(retrieved, item, k) for k in K_VALS}
        for k in K_VALS:
            hit_counts[k] += int(bool(per_k[k]))
            rr_totals[k] += _reciprocal_rank(per_k[k])

        # Use @5 for the per-row rank/RR display
        hit_ranks_5 = per_k[5]
        hit_ranks_10 = per_k[10]
        rr = _reciprocal_rank(hit_ranks_5)

        # Build preview of top result
        top_slug = retrieved[0].metadata.get("work_slug", "?") if retrieved else "—"
        top_aph = retrieved[0].metadata.get("aphorism_number", "?") if retrieved else "—"
        top_preview = f"{top_slug} §{top_aph}"

        hit5_str = "[green]✓[/green]" if hit_ranks_5 else "[red]✗[/red]"
        hit10_str = "[green]✓[/green]" if hit_ranks_10 else "[red]✗[/red]"
        # Show first hit rank regardless of cutoff (most informative)
        all_ranks = per_k[_MAX_K]
        rank_str = str(min(all_ranks)) if all_ranks else "—"
        rr_str = f"{rr:.3f}"

        results_table.add_row(
            str(i),
            item.question[:44],
            hit5_str,
            hit10_str,
            rank_str,
            rr_str,
            top_preview,
        )

        detail_rows.append({
            "q": item.question,
            "hit5": bool(hit_ranks_5),
            "hit10": bool(hit_ranks_10),
            "rr": rr,
            "retrieved": [(r.metadata.get("work_slug"), r.metadata.get("aphorism_number")) for r in retrieved],
            "ground_truth": item.ground_truth,
            "elapsed": elapsed,
        })

    n = len(EVAL_SET)
    summary_lines = []
    for k in K_VALS:
        hr = hit_counts[k] / n
        mrr = rr_totals[k] / n
        summary_lines.append(
            f"[bold]HR@{k}:[/bold]  {hr:.1%}  ({hit_counts[k]}/{n})   "
            f"[bold]MRR@{k}:[/bold] {mrr:.3f}"
        )

    console.print(results_table)
    console.print()
    console.print(
        Panel(
            "\n".join(summary_lines),
            title=f"[bold green]Results — {label}[/bold green]",
            border_style="green",
            padding=(0, 2),
        )
    )

    # Print misses (@10) for debugging
    misses = [d for d in detail_rows if not d["hit10"]]
    if misses:
        console.print("\n[bold yellow]Misses @10:[/bold yellow]")
        for d in misses:
            console.print(f"  Q: {d['q']}")
            console.print(f"  GT:        {d['ground_truth']}")
            console.print(f"  Retrieved: {d['retrieved']}")
            console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--synonyms", action="store_true", help="Enable BM25 synonym expansion")
    args = parser.parse_args()

    import os
    os.environ.setdefault("VECTOR_STORE_BACKEND", "supabase")
    run(use_synonyms=args.synonyms)
