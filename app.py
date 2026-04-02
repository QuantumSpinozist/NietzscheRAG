"""Nietzsche RAG — CLI entry point.

Commands
--------
    python app.py ingest [--work SLUG] [--all]
    python app.py query "What does Nietzsche mean by the will to power?"
    python app.py query "What is eternal recurrence?" --period late
    python app.py query "How does Nietzsche view Socrates?" --work twilight_of_the_idols
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

import config
from ingest.embed import WORK_REGISTRY, WORK_END_BEFORE, WORK_START_AFTER, embed_chunks
from ingest.fetch import GUTENBERG_SOURCES, save_work

app = typer.Typer(
    name="nietzsche-rag",
    help="Retrieval-Augmented Generation over the Nietzsche corpus.",
    add_completion=False,
)

console = Console()
err = Console(stderr=True)


# ── ingest ────────────────────────────────────────────────────────────────────


@app.command()
def ingest(
    work: Optional[str] = typer.Option(
        None,
        "--work",
        help="Slug of a single work to ingest (e.g. beyond_good_and_evil).",
        metavar="SLUG",
    ),
    all_works: bool = typer.Option(
        False,
        "--all",
        help="Ingest all works that have a raw file downloaded.",
    ),
) -> None:
    """Download (if needed) and embed Nietzsche works into ChromaDB.

    With no flags, ingests every work whose raw .txt file already exists.
    Use --work SLUG to ingest a single work, or --all to attempt every known work.
    """
    from ingest.chunk import chunk_work

    if work and work not in WORK_REGISTRY:
        err.print(f"[red]Unknown work slug:[/red] {work!r}")
        err.print(f"Valid slugs: {', '.join(WORK_REGISTRY)}")
        raise typer.Exit(1)

    if work:
        slugs = [work]
    elif all_works:
        slugs = list(WORK_REGISTRY)
    else:
        # Default: every slug that has a raw file on disk
        slugs = [s for s in WORK_REGISTRY if (config.RAW_DIR / f"{s}.txt").exists()]
        if not slugs:
            err.print(
                "[yellow]No raw files found.[/yellow] "
                "Run with --all to download and ingest everything."
            )
            raise typer.Exit(0)

    failures: list[str] = []

    for slug in slugs:
        title, period, chunk_style = WORK_REGISTRY[slug]
        raw_path = config.RAW_DIR / f"{slug}.txt"

        # Download if missing
        if not raw_path.exists():
            if slug not in GUTENBERG_SOURCES:
                err.print(
                    f"[yellow]No Gutenberg URL for {slug!r} — skipping download.[/yellow]"
                )
            else:
                try:
                    save_work(slug, config.RAW_DIR)
                except Exception as exc:
                    err.print(f"[red]Download failed for {slug!r}:[/red] {exc}")
                    failures.append(slug)
                    continue

        if not raw_path.exists():
            err.print(f"[yellow]Raw file still missing for {slug!r}, skipping embed.[/yellow]")
            failures.append(slug)
            continue

        console.rule(f"[bold cyan]{title}[/bold cyan]")
        text = raw_path.read_text(encoding="utf-8")
        end_before = WORK_END_BEFORE.get(slug)
        start_after = WORK_START_AFTER.get(slug)
        chunks = chunk_work(
            text, title, slug, period, chunk_style,
            end_before=end_before,
            start_after=start_after,
        )
        console.print(f"  Chunked into [cyan]{len(chunks)}[/cyan] pieces")

        try:
            embed_chunks(chunks)
        except Exception as exc:
            err.print(f"[red]Embed failed for {slug!r}:[/red] {exc}")
            failures.append(slug)

    console.rule()
    if failures:
        console.print(f"[bold red]{len(failures)} work(s) failed:[/bold red] {', '.join(failures)}")
        raise typer.Exit(1)
    else:
        console.print(f"[bold green]Ingestion complete.[/bold green] {len(slugs)} work(s) processed.")


# ── query ─────────────────────────────────────────────────────────────────────


@app.command()
def query(
    question: str = typer.Argument(..., help="Natural-language question to ask Nietzsche."),
    period: Optional[str] = typer.Option(
        None,
        "--period",
        help="Restrict retrieval to a period: early | middle | late",
    ),
    work: Optional[str] = typer.Option(
        None,
        "--work",
        help="Restrict retrieval to a single work slug (e.g. twilight_of_the_idols).",
        metavar="SLUG",
    ),
    top_n: int = typer.Option(
        config.RERANK_TOP_N,
        "--top-n",
        help="Number of passages to retrieve and pass to the model.",
    ),
) -> None:
    """Ask a philosophical question and get a grounded, cited answer."""
    from generation.claude import generate_answer
    from retrieval.hybrid import hybrid_search
    from retrieval.sparse import BM25Index
    from retrieval.store import get_vector_store

    # ── validate ──────────────────────────────────────────────────────────────
    if period and period not in ("early", "middle", "late"):
        err.print(f"[red]Invalid period:[/red] {period!r}. Must be early, middle, or late.")
        raise typer.Exit(1)

    if work and work not in WORK_REGISTRY:
        err.print(f"[red]Unknown work slug:[/red] {work!r}")
        err.print(f"Valid slugs: {', '.join(WORK_REGISTRY)}")
        raise typer.Exit(1)

    if config.VECTOR_STORE_BACKEND == "chroma" and not config.CHROMA_PERSIST_DIR.exists():
        err.print(
            "[red]ChromaDB not found.[/red] Run [bold]python app.py ingest[/bold] first."
        )
        raise typer.Exit(1)

    if not (config.ANTHROPIC_API_KEY and config.ANTHROPIC_API_KEY != "your_key_here"):
        err.print("[red]ANTHROPIC_API_KEY not set.[/red] Add it to .env")
        raise typer.Exit(1)

    # ── load corpus for BM25 ─────────────────────────────────────────────────
    try:
        store = get_vector_store()
        corpus = store.get_all_documents()
    except Exception as exc:
        err.print(f"[red]Could not open vector store:[/red] {exc}")
        raise typer.Exit(1)

    # Apply period/work filter to BM25 corpus if needed
    if period or work:
        filtered = [
            (id_, doc, meta)
            for id_, doc, meta in zip(
                corpus["ids"], corpus["documents"], corpus["metadatas"]
            )
            if _matches_filter(meta, period, work)
        ]
        if filtered:
            f_ids, f_docs, f_metas = zip(*filtered)
            bm25 = BM25Index(ids=list(f_ids), documents=list(f_docs), metadatas=list(f_metas))
        else:
            bm25 = BM25Index(ids=[], documents=[], metadatas=[])
    else:
        bm25 = BM25Index(
            ids=corpus["ids"],
            documents=corpus["documents"],
            metadatas=corpus["metadatas"],
        )

    # ── retrieve ──────────────────────────────────────────────────────────────
    console.print(f"\n[bold yellow]❯[/bold yellow] {question}\n")
    with console.status("[cyan]Retrieving passages …[/cyan]"):
        results = hybrid_search(
            question,
            top_n=top_n,
            bm25_index=bm25,
            filter_period=period,
            filter_slug=work,
        )

    if not results:
        console.print("[yellow]No passages found for this query / filter combination.[/yellow]")
        raise typer.Exit(0)

    # ── show sources table ────────────────────────────────────────────────────
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("#", width=3)
    table.add_column("Work", width=28)
    table.add_column("§", width=6)
    table.add_column("Rerank", width=7)
    table.add_column("Preview")

    for i, r in enumerate(results, 1):
        aph = r.metadata.get("aphorism_number", -1)
        aph_end = r.metadata.get("aphorism_number_end", aph)
        if aph != -1:
            aph_str = f"§{aph}" if aph == aph_end else f"§{aph}–{aph_end}"
        else:
            ch = r.metadata.get("section_number", "?")
            aph_str = f"ch.{ch}"
        rerank_str = f"{r.rerank_score:.2f}" if r.rerank_score is not None else "—"
        preview = " ".join(r.document.split()[:14])
        if len(r.document.split()) > 14:
            preview += " …"
        table.add_row(str(i), r.metadata.get("work_title", "?"), aph_str, rerank_str, preview)

    console.print(table)
    console.print()

    # ── generate ──────────────────────────────────────────────────────────────
    with console.status("[cyan]Generating answer …[/cyan]"):
        answer = generate_answer(question, results)

    console.print(
        Panel(
            answer,
            title="[bold green]Answer[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )


# ── filter helper ─────────────────────────────────────────────────────────────


def _matches_filter(meta: dict, period: str | None, work_slug: str | None) -> bool:
    if period and meta.get("work_period") != period:
        return False
    if work_slug and meta.get("work_slug") != work_slug:
        return False
    return True


# ── entrypoint ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    app()
