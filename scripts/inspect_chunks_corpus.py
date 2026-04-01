"""Inspect chunk boundaries across the full corpus.

For every work in WORK_REGISTRY, samples the first chunk, the last chunk,
and several boundary pairs (last token of chunk N / first token of chunk N+1)
to verify that:
  - Gutenberg boilerplate is absent
  - Content is genuine Nietzsche text
  - Aphorism numbers (where applicable) are sensible
  - For TI, The Antichrist body is absent

Usage::

    python scripts/inspect_chunks_corpus.py [--work SLUG]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

# allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from ingest.chunk import chunk_work
from ingest.embed import WORK_END_BEFORE, WORK_REGISTRY

console = Console()


def _show_chunk(label: str, c, *, max_chars: int = 300) -> None:
    snippet = c.content[:max_chars].replace("\n", " ")
    if len(c.content) > max_chars:
        snippet += " …"
    aph = (
        f"§{c.aphorism_number}"
        if c.aphorism_number is not None
        else "prose"
    )
    console.print(
        f"  [dim]{label}[/dim] idx={c.chunk_index} {aph} "
        f"ch={c.section_number} [{len(c.content.split())} tok]\n"
        f"  [italic]{snippet}[/italic]\n"
    )


def inspect_work(slug: str) -> int:
    """Inspect chunks for *slug*.  Returns number of problems found."""
    title, period, style = WORK_REGISTRY[slug]
    raw = config.RAW_DIR / f"{slug}.txt"
    if not raw.exists():
        console.print(f"[yellow]  SKIP — raw file not found: {raw}[/yellow]")
        return 0

    end_before = WORK_END_BEFORE.get(slug)
    chunks = chunk_work(
        raw.read_text(encoding="utf-8"),
        title, slug, period, style,
        end_before=end_before,
    )

    problems = 0
    console.rule(f"[bold cyan]{title}[/bold cyan]  ({len(chunks)} chunks, {style})")

    # ── Summary table ─────────────────────────────────────────────────────────
    table = Table(show_header=True, header_style="bold", expand=False)
    table.add_column("stat", width=22)
    table.add_column("value")

    aph_nums = sorted(
        c.aphorism_number
        for c in chunks
        if c.aphorism_number is not None
    )
    table.add_row("total chunks", str(len(chunks)))
    table.add_row("chunk type", style)
    table.add_row(
        "aphorism range",
        f"§{min(aph_nums)}–§{max(aph_nums)}" if aph_nums else "n/a",
    )
    avg_tok = sum(len(c.content.split()) for c in chunks) / len(chunks)
    table.add_row("avg tokens/chunk", f"{avg_tok:.0f}")
    table.add_row(
        "empty chunks",
        str(sum(1 for c in chunks if not c.content.strip())),
    )
    boilerplate_hits = sum(
        1 for c in chunks if "Project Gutenberg" in c.content or "START OF" in c.content
    )
    table.add_row("boilerplate hits", str(boilerplate_hits))
    if slug == "twilight_of_the_idols":
        ac_hits = sum(
            1 for c in chunks if "This book belongs to the very few" in c.content
        )
        table.add_row("Antichrist leakage", str(ac_hits))
        if ac_hits:
            problems += ac_hits

    console.print(table)

    if boilerplate_hits:
        problems += boilerplate_hits

    # ── Sample first / last chunk ─────────────────────────────────────────────
    _show_chunk("FIRST", chunks[0])
    _show_chunk("LAST ", chunks[-1])

    # ── Sample 3 boundary pairs ───────────────────────────────────────────────
    step = max(1, len(chunks) // 4)
    for i in range(step, len(chunks) - 1, step):
        if i + 1 >= len(chunks):
            break
        a, b = chunks[i], chunks[i + 1]
        tail = " ".join(a.content.split()[-10:])
        head = " ".join(b.content.split()[:10])
        aph_a = f"§{a.aphorism_number}" if a.aphorism_number else "prose"
        aph_b = f"§{b.aphorism_number}" if b.aphorism_number else "prose"
        console.print(
            f"  [dim]boundary {i}→{i+1}[/dim]  {aph_a} | {aph_b}\n"
            f"  …[yellow]{tail}[/yellow] | [green]{head}[/green]…\n"
        )

    # ── Flag suspiciously large aphorism numbers ──────────────────────────────
    large = [c.aphorism_number for c in chunks if c.aphorism_number and c.aphorism_number > 999]
    if large:
        console.print(f"  [red]WARNING: aphorism numbers > 999: {large[:5]}[/red]")
        problems += len(large)

    return problems


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Inspect chunk quality across the corpus.")
    p.add_argument(
        "--work",
        default=None,
        choices=list(WORK_REGISTRY),
        metavar="SLUG",
        help="Inspect a single work (default: all)",
    )
    args = p.parse_args(argv)

    slugs = [args.work] if args.work else list(WORK_REGISTRY)
    total_problems = 0

    for slug in slugs:
        total_problems += inspect_work(slug)

    console.rule()
    if total_problems:
        console.print(f"[bold red]{total_problems} problem(s) detected.[/bold red]")
        sys.exit(1)
    else:
        console.print(
            f"[bold green]All {len(slugs)} work(s) look clean.[/bold green]"
        )


if __name__ == "__main__":
    main()
