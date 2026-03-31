"""Download plain-text Nietzsche works from Project Gutenberg."""

from __future__ import annotations

import sys
from pathlib import Path

import requests
from rich.console import Console

console = Console(stderr=True)

# Gutenberg plain-text URLs keyed by output slug
GUTENBERG_SOURCES: dict[str, tuple[str, str]] = {
    "beyond_good_and_evil": (
        "Beyond Good and Evil",
        "https://www.gutenberg.org/cache/epub/4363/pg4363.txt",
    ),
}

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def fetch_text(url: str, slug: str) -> str:
    """Download plain text from *url*, streaming with progress reported to stderr.

    Args:
        url: Full URL to the plain-text file.
        slug: Human-readable identifier used only for progress messages.

    Returns:
        The decoded text content.

    Raises:
        requests.HTTPError: If the server returns a non-2xx status code.
    """
    console.print(f"[bold cyan]Fetching[/bold cyan] {slug} from {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    received = 0
    chunks: list[bytes] = []

    for chunk in response.iter_content(chunk_size=8192):
        chunks.append(chunk)
        received += len(chunk)
        if total:
            pct = received / total * 100
            console.print(
                f"  [dim]{received:,} / {total:,} bytes ({pct:.0f}%)[/dim]",
                end="\r",
            )

    console.print(f"  [green]Downloaded {received:,} bytes[/green]          ")
    return b"".join(chunks).decode("utf-8", errors="replace")


def save_work(slug: str, dest_dir: Path = RAW_DIR) -> Path:
    """Fetch a Gutenberg work by slug and save it to *dest_dir/<slug>.txt*.

    Skips the download if the file already exists.

    Args:
        slug: Key into ``GUTENBERG_SOURCES`` (e.g. ``"beyond_good_and_evil"``).
        dest_dir: Directory where the .txt file will be written.

    Returns:
        Path to the saved file.

    Raises:
        KeyError: If *slug* is not in ``GUTENBERG_SOURCES``.
        requests.HTTPError: If the download fails.
    """
    if slug not in GUTENBERG_SOURCES:
        raise KeyError(f"Unknown slug {slug!r}. Available: {list(GUTENBERG_SOURCES)}")

    title, url = GUTENBERG_SOURCES[slug]
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"{slug}.txt"

    if out_path.exists():
        console.print(
            f"[yellow]Skipping[/yellow] {title} — already exists at {out_path}"
        )
        return out_path

    text = fetch_text(url, title)
    out_path.write_text(text, encoding="utf-8")
    console.print(f"[bold green]Saved[/bold green] {title} → {out_path}")
    return out_path


def fetch_all(dest_dir: Path = RAW_DIR) -> list[Path]:
    """Fetch every work defined in ``GUTENBERG_SOURCES``.

    Args:
        dest_dir: Directory where .txt files will be written.

    Returns:
        List of paths to all saved files.
    """
    paths: list[Path] = []
    for slug in GUTENBERG_SOURCES:
        paths.append(save_work(slug, dest_dir))
    return paths


if __name__ == "__main__":
    saved = fetch_all()
    console.print(f"\n[bold]Done.[/bold] {len(saved)} file(s) ready in {RAW_DIR}")
    sys.exit(0)
