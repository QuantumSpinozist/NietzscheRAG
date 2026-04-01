"""Embed Nietzsche chunks and persist them to a ChromaDB collection.

CLI usage::

    python ingest/embed.py --work beyond_good_and_evil
    python ingest/embed.py --work beyond_good_and_evil --collection nietzsche
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chromadb
from chromadb import Collection
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import track

from ingest.chunk import Chunk, chunk_work
import config

console = Console(stderr=True)

# Metadata for every supported work: slug → (title, period, chunk_style)
WORK_REGISTRY: dict[str, tuple[str, str, str]] = {
    # ── Late period ───────────────────────────────────────────────────────────
    "beyond_good_and_evil": ("Beyond Good and Evil", "late", "aphorism"),
    "genealogy_of_morality": ("On the Genealogy of Morality", "late", "paragraph"),
    # PG 52263 bundles TI + The Antichrist; paragraph style + WORK_END_BEFORE truncation.
    "twilight_of_the_idols": ("Twilight of the Idols", "late", "paragraph"),
    "the_antichrist": ("The Antichrist", "late", "aphorism"),
    "ecce_homo": ("Ecce Homo", "late", "paragraph"),
    # PG 25012 bundles The Case of Wagner + NCW + Selected Aphorisms.
    "nietzsche_contra_wagner": ("Nietzsche contra Wagner", "late", "aphorism"),
    # ── Middle period ─────────────────────────────────────────────────────────
    "the_gay_science": ("The Gay Science", "middle", "aphorism"),
    "daybreak": ("Daybreak", "middle", "aphorism"),
    "human_all_too_human": ("Human, All Too Human", "middle", "aphorism"),
    # ── Early period ──────────────────────────────────────────────────────────
    "birth_of_tragedy": ("The Birth of Tragedy", "early", "paragraph"),
    # PG 51710: Thoughts Out of Season Part I (David Strauss + Use/Abuse of History)
    "untimely_meditations_1": ("Untimely Meditations I–II", "early", "paragraph"),
    # PG 38226: Thoughts Out of Season Part II (Schopenhauer as Educator + Wagner)
    "untimely_meditations_2": ("Untimely Meditations III–IV", "early", "paragraph"),
}

# Works whose Gutenberg file bundles multiple texts.  chunk_work() truncates
# the stripped body at the first occurrence of this string so only the
# intended work is chunked.
WORK_END_BEFORE: dict[str, str] = {
    "twilight_of_the_idols": "THE ANTICHRIST\n\nAn Attempted Criticism",
}


# ── ChromaDB helpers ──────────────────────────────────────────────────────────


def get_chroma_collection(
    persist_dir: Path = config.CHROMA_PERSIST_DIR,
    collection_name: str = config.COLLECTION_NAME,
) -> Collection:
    """Return (or create) a persistent ChromaDB collection.

    Args:
        persist_dir: Directory where ChromaDB stores its data files.
        collection_name: Name of the collection to get or create.

    Returns:
        A ChromaDB :class:`Collection` object.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(collection_name)


# ── Metadata serialisation ────────────────────────────────────────────────────


def _chunk_metadata(chunk: Chunk) -> dict:
    """Return a ChromaDB-safe metadata dict for *chunk*.

    ChromaDB metadata values must be ``str``, ``int``, ``float``, or ``bool``.
    ``None`` integer fields (e.g. ``aphorism_number`` for prose chunks) are
    stored as ``-1`` so filters can still target them unambiguously.
    """
    return {
        "work_title": chunk.work_title,
        "work_slug": chunk.work_slug,
        "work_period": chunk.work_period,
        "section_number": chunk.section_number,
        "aphorism_number": chunk.aphorism_number if chunk.aphorism_number is not None else -1,
        "aphorism_number_end": chunk.aphorism_number_end if chunk.aphorism_number_end is not None else -1,
        "chunk_index": chunk.chunk_index,
        "chunk_type": chunk.chunk_type,
    }


def _chunk_id(chunk: Chunk) -> str:
    """Return a stable, unique ChromaDB document ID for *chunk*."""
    return f"{chunk.work_slug}_chunk_{chunk.chunk_index}"


# ── Embedding ─────────────────────────────────────────────────────────────────


def embed_chunks(
    chunks: list[Chunk],
    persist_dir: Path = config.CHROMA_PERSIST_DIR,
    collection_name: str = config.COLLECTION_NAME,
    model_name: str = config.EMBEDDING_MODEL,
    batch_size: int = config.EMBED_BATCH_SIZE,
) -> int:
    """Embed *chunks* with a sentence-transformer and upsert into ChromaDB.

    Existing entries with the same ID are overwritten (upsert semantics), so
    re-ingesting a work is safe.

    Args:
        chunks: Chunks to embed, typically the output of :func:`chunk_work`.
        persist_dir: ChromaDB persistence directory.
        collection_name: Target collection name.
        model_name: HuggingFace model identifier for :class:`SentenceTransformer`.
        batch_size: Number of chunks to encode and upsert per iteration.

    Returns:
        Total number of chunks upserted.
    """
    if not chunks:
        console.print("[yellow]embed_chunks: no chunks to embed, skipping.[/yellow]")
        return 0

    collection = get_chroma_collection(persist_dir, collection_name)
    console.print(f"[cyan]Loading embedding model[/cyan] {model_name!r} …")
    model = SentenceTransformer(model_name)

    total = 0
    n_batches = (len(chunks) + batch_size - 1) // batch_size
    console.print(
        f"[cyan]Embedding[/cyan] {len(chunks)} chunks "
        f"in {n_batches} batch(es) of ≤{batch_size}"
    )

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.content for c in batch]

        embeddings: list[list[float]] = model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        ).tolist()

        collection.upsert(
            ids=[_chunk_id(c) for c in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[_chunk_metadata(c) for c in batch],
        )
        total += len(batch)
        console.print(
            f"  batch {i // batch_size + 1}/{n_batches} — "
            f"upserted {len(batch)} chunks (total {total})"
        )

    console.print(f"[bold green]Done.[/bold green] {total} chunks in {collection_name!r}")
    return total


# ── CLI entry point ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Embed a Nietzsche work and persist it to ChromaDB."
    )
    p.add_argument(
        "--work",
        required=True,
        choices=list(WORK_REGISTRY),
        metavar="SLUG",
        help=f"Work slug. Choices: {', '.join(WORK_REGISTRY)}",
    )
    p.add_argument(
        "--collection",
        default=config.COLLECTION_NAME,
        help=f"ChromaDB collection name (default: {config.COLLECTION_NAME!r})",
    )
    p.add_argument(
        "--persist-dir",
        type=Path,
        default=config.CHROMA_PERSIST_DIR,
        help=f"ChromaDB persistence directory (default: {config.CHROMA_PERSIST_DIR})",
    )
    p.add_argument(
        "--model",
        default=config.EMBEDDING_MODEL,
        help=f"SentenceTransformer model (default: {config.EMBEDDING_MODEL!r})",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=config.EMBED_BATCH_SIZE,
        help=f"Embedding batch size (default: {config.EMBED_BATCH_SIZE})",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    slug = args.work
    title, period, chunk_style = WORK_REGISTRY[slug]
    raw_path = config.RAW_DIR / f"{slug}.txt"

    if not raw_path.exists():
        console.print(f"[red]Raw file not found:[/red] {raw_path}")
        console.print("  Run: python ingest/fetch.py   (or python app.py ingest)")
        sys.exit(1)

    console.print(f"[bold]Ingesting[/bold] {title!r} ({period}, {chunk_style})")
    console.print(f"  source : {raw_path}")

    text = raw_path.read_text(encoding="utf-8")
    end_before = WORK_END_BEFORE.get(slug)
    chunks = chunk_work(text, title, slug, period, chunk_style, end_before=end_before)
    console.print(f"  chunks : {len(chunks)}")

    embed_chunks(
        chunks,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        model_name=args.model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
