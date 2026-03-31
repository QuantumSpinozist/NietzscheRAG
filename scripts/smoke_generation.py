"""Smoke test for generation/claude.py.

Demonstrates:
  1. Prompt building with hardcoded passages (no API needed)
  2. Live end-to-end query through the full pipeline if ANTHROPIC_API_KEY is set

Run from the project root:
    python scripts/smoke_generation.py
    python scripts/smoke_generation.py --live   # force live API call

Exits with non-zero code if any assertion fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax

import config
from generation.claude import SYSTEM_PROMPT, build_prompt, generate_answer
from retrieval.hybrid import HybridResult

console = Console()

# ── Hardcoded example data ────────────────────────────────────────────────────

_EXAMPLE_RESULTS = [
    HybridResult(
        id="beyond_good_and_evil_chunk_18",
        document=(
            "19. Philosophers are accustomed to speak of the will as though it were "
            "the best-known thing in the world; indeed, Schopenhauer has given us to "
            "understand that the will alone is really known to us, absolutely and "
            "completely known, without deduction or addition. But it again and again "
            "seems to me that in this case Schopenhauer also only did what "
            "philosophers are in the habit of doing--he seems to have adopted a "
            "POPULAR PREJUDICE and exaggerated it. Willing seems to me to be above "
            "all something COMPLICATED, something that is a unity only in name."
        ),
        metadata={
            "work_title": "Beyond Good and Evil",
            "work_slug": "beyond_good_and_evil",
            "work_period": "late",
            "section_number": 1,
            "aphorism_number": 19,
            "aphorism_number_end": 19,
            "chunk_index": 18,
            "chunk_type": "aphorism",
        },
        rrf_score=0.031,
        rerank_score=2.1,
    ),
    HybridResult(
        id="beyond_good_and_evil_chunk_258",
        document=(
            "259. To refrain mutually from injury, from violence, from exploitation, "
            "and put one's will on a par with that of others: this may result in a "
            "certain rough sense in good conduct among individuals when the necessary "
            "conditions are given (namely, the actual similarity of the individuals "
            "in amount of force and degree of worth, and their co-relation within one "
            "organisation). As soon, however, as one wished to take this principle "
            "more generally, and if possible even as the FUNDAMENTAL PRINCIPLE OF "
            "SOCIETY, it would immediately disclose what it really is--namely, a Will "
            "to the DENIAL of life, a principle of dissolution and decay."
        ),
        metadata={
            "work_title": "Beyond Good and Evil",
            "work_slug": "beyond_good_and_evil",
            "work_period": "late",
            "section_number": 9,
            "aphorism_number": 259,
            "aphorism_number_end": 259,
            "chunk_index": 170,
            "chunk_type": "aphorism",
        },
        rrf_score=0.029,
        rerank_score=1.8,
    ),
    HybridResult(
        id="beyond_good_and_evil_chunk_36",
        document=(
            "36. Granted that nothing is 'given' as real except our world of desires "
            "and passions, and that we cannot get down, or up, to any other 'reality' "
            "besides the reality of our drives--for thinking is only a relation of "
            "these drives to each other: is it not permitted to make the experiment "
            "and to ask the question whether this 'given' does not suffice for also "
            "understanding on the basis of this kind of thing the so-called "
            "mechanistic (or 'material') world? ... The world viewed from inside, the "
            "world defined and designated according to its 'intelligible character'--"
            "it would be 'Will to Power' and nothing else."
        ),
        metadata={
            "work_title": "Beyond Good and Evil",
            "work_slug": "beyond_good_and_evil",
            "work_period": "late",
            "section_number": 2,
            "aphorism_number": 36,
            "aphorism_number_end": 36,
            "chunk_index": 35,
            "chunk_type": "aphorism",
        },
        rrf_score=0.027,
        rerank_score=3.5,
    ),
]

_EXAMPLE_QUESTION = "What does Nietzsche mean by the will to power?"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _show_system_prompt() -> None:
    console.print(Rule("[bold cyan]System Prompt[/bold cyan]"))
    console.print(Panel(SYSTEM_PROMPT, border_style="cyan", padding=(0, 1)))


def _show_built_prompt(question: str, results: list[HybridResult]) -> str:
    prompt = build_prompt(question, results)
    console.print(Rule("[bold cyan]Built User Prompt[/bold cyan]"))
    console.print(Panel(prompt, border_style="blue", padding=(0, 1)))
    return prompt


def _assert_prompt_structure(prompt: str, question: str, results: list[HybridResult]) -> None:
    assert prompt.startswith("Source passages:"), "Prompt must start with 'Source passages:'"
    assert question in prompt, "Question must appear verbatim in prompt"
    for r in results:
        assert r.document in prompt, f"Document missing from prompt: {r.id}"
        work_title = r.metadata["work_title"]
        assert work_title in prompt, f"Work title missing: {work_title}"
    console.print("  [green]✓[/green] Prompt structure assertions passed")


def _live_query(question: str, results: list[HybridResult]) -> None:
    console.print(Rule("[bold cyan]Live Claude Answer[/bold cyan]"))
    console.print(f"[bold yellow]Question:[/bold yellow] {question}\n")

    try:
        answer = generate_answer(question, results)
        console.print(Panel(answer, title="[green]Claude's answer[/green]",
                            border_style="green", padding=(0, 1)))
        assert isinstance(answer, str) and len(answer) > 0, "Expected a non-empty answer"
        console.print("  [green]✓[/green] generate_answer returned a non-empty string")
    except Exception as exc:
        console.print(f"[red]FAIL[/red] {exc}")
        raise


def _pipeline_query(question: str) -> None:
    """Run the full pipeline: hybrid retrieval → generation."""
    console.print(Rule("[bold cyan]Full Pipeline Query[/bold cyan]"))
    console.print(f"[bold yellow]Question:[/bold yellow] {question}\n")

    import chromadb
    from retrieval.hybrid import hybrid_search
    from retrieval.sparse import BM25Index

    client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
    col = client.get_collection(config.COLLECTION_NAME)
    data = col.get(include=["documents", "metadatas"])
    bm25 = BM25Index(ids=data["ids"], documents=data["documents"], metadatas=data["metadatas"])

    results = hybrid_search(question, persist_dir=config.CHROMA_PERSIST_DIR,
                            collection_name=config.COLLECTION_NAME,
                            top_n=config.RERANK_TOP_N, bm25_index=bm25)

    console.print(f"[cyan]Retrieved {len(results)} passages via hybrid search[/cyan]")
    for i, r in enumerate(results, 1):
        aph = r.metadata.get("aphorism_number", "?")
        console.print(f"  {i}. §{aph} (rerank={r.rerank_score:.2f}): "
                      f"{' '.join(r.document.split()[:12])} …")

    answer = generate_answer(question, results)
    console.print(Panel(answer, title="[green]Claude's answer[/green]",
                        border_style="green", padding=(0, 1)))
    assert isinstance(answer, str) and len(answer) > 0
    console.print("  [green]✓[/green] Full pipeline answer received")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generation smoke test")
    parser.add_argument("--live", action="store_true",
                        help="Run live API calls even if key looks placeholder")
    args = parser.parse_args()

    has_key = bool(config.ANTHROPIC_API_KEY and
                   config.ANTHROPIC_API_KEY != "your_key_here")
    run_live = has_key or args.live

    failures: list[str] = []

    # ── 1. Show system prompt ─────────────────────────────────────────────────
    _show_system_prompt()

    # ── 2. Build and show example prompt ─────────────────────────────────────
    console.print()
    prompt = _show_built_prompt(_EXAMPLE_QUESTION, _EXAMPLE_RESULTS)

    # ── 3. Assert prompt structure ────────────────────────────────────────────
    console.print(Rule("[bold cyan]Prompt Structure Assertions[/bold cyan]"))
    try:
        _assert_prompt_structure(prompt, _EXAMPLE_QUESTION, _EXAMPLE_RESULTS)
    except AssertionError as exc:
        console.print(f"[red]FAIL[/red] {exc}")
        failures.append(str(exc))

    # ── 4. Live API calls ─────────────────────────────────────────────────────
    if run_live:
        console.print()
        try:
            _live_query(_EXAMPLE_QUESTION, _EXAMPLE_RESULTS)
        except Exception as exc:
            failures.append(str(exc))

        # Second example with different question
        console.print()
        try:
            _live_query(
                "What is the distinction between master and slave morality?",
                _EXAMPLE_RESULTS,
            )
        except Exception as exc:
            failures.append(str(exc))

        # Full pipeline query
        if config.CHROMA_PERSIST_DIR.exists():
            console.print()
            try:
                _pipeline_query("What does Nietzsche say about philosophers and truth?")
            except Exception as exc:
                console.print(f"[red]FAIL[/red] {exc}")
                failures.append(str(exc))
    else:
        console.print()
        console.print(
            "[yellow]Skipping live API calls[/yellow] — set ANTHROPIC_API_KEY in .env "
            "or pass --live to enable."
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    console.print()
    console.rule()
    if failures:
        console.print(f"[bold red]{len(failures)} assertion(s) failed.[/bold red]")
        for f in failures:
            console.print(f"  • {f}")
        sys.exit(1)
    else:
        live_note = " (including live API calls)" if run_live else " (prompt-only checks)"
        console.print(f"[bold green]All generation smoke checks passed.[/bold green]{live_note}")


if __name__ == "__main__":
    main()
