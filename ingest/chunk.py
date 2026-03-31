"""Chunk Nietzsche texts into aphorisms or prose paragraphs with metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.console import Console

console = Console(stderr=True)

ChunkStyle = Literal["aphorism", "paragraph"]
Period = Literal["early", "middle", "late"]

# ── Gutenberg boundary markers ────────────────────────────────────────────────

_START_RE = re.compile(
    r"\*{3}\s*START OF THE PROJECT GUTENBERG EBOOK .+?\*{3}", re.IGNORECASE
)
_END_RE = re.compile(
    r"\*{3}\s*END OF THE PROJECT GUTENBERG EBOOK .+?\*{3}", re.IGNORECASE
)
# Secondary end marker: "End of Project Gutenberg's <title>" line
_END_SECONDARY_RE = re.compile(r"\nEnd of Project Gutenberg[^\n]*\n", re.IGNORECASE)

# ── Section detection ─────────────────────────────────────────────────────────

# "42. Some text" at the very start of a line (requires non-whitespace after)
_APHORISM_RE = re.compile(r"^(\d+)\.\s+\S", re.MULTILINE)

# "CHAPTER IV. TITLE" or "CHAPTER IV:" — Roman numeral up to 6 chars
_CHAPTER_RE = re.compile(
    r"^CHAPTER\s+((?:I|V|X|L|C|D|M){1,6})\b",
    re.MULTILINE | re.IGNORECASE,
)


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """A single retrievable passage from a Nietzsche work."""

    content: str
    work_title: str
    work_slug: str
    work_period: Period
    section_number: int           # chapter / part number (0 = preamble/preface)
    aphorism_number: int | None   # first (or only) aphorism number in this chunk
    aphorism_number_end: int | None  # last aphorism number; equals aphorism_number
                                     # when no merging occurred, None for prose
    chunk_index: int              # position within the work
    chunk_type: ChunkStyle

    def to_dict(self) -> dict:
        """Return a plain dict suitable for ChromaDB metadata storage."""
        return {
            "content": self.content,
            "work_title": self.work_title,
            "work_slug": self.work_slug,
            "work_period": self.work_period,
            "section_number": self.section_number,
            "aphorism_number": self.aphorism_number,
            "aphorism_number_end": self.aphorism_number_end,
            "chunk_index": self.chunk_index,
            "chunk_type": self.chunk_type,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _token_count(text: str) -> int:
    """Whitespace-based token count — fast and dependency-free."""
    return len(text.split())


def _tail_tokens(text: str, n: int) -> str:
    """Return the last *n* whitespace tokens of *text* as a single string."""
    tokens = text.split()
    return " ".join(tokens[-n:]) if len(tokens) > n else text


def _roman_to_int(roman: str) -> int:
    """Convert a Roman numeral string (e.g. ``'XIV'``) to an integer."""
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total, prev = 0, 0
    for ch in reversed(roman.upper()):
        v = vals.get(ch, 0)
        total += v if v >= prev else -v
        prev = v
    return total


# ── Gutenberg cleanup ─────────────────────────────────────────────────────────


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove the Project Gutenberg header and footer from *text*.

    Args:
        text: Raw text as downloaded from Project Gutenberg.

    Returns:
        The bare work content, stripped of leading/trailing whitespace.
    """
    start_m = _START_RE.search(text)
    if start_m:
        text = text[start_m.end():]
    end_m = _END_RE.search(text)
    if end_m:
        text = text[: end_m.start()]
    # Also strip the informal "End of Project Gutenberg's …" line that some
    # editions include just before the formal *** END *** marker.
    sec_m = _END_SECONDARY_RE.search(text)
    if sec_m:
        text = text[: sec_m.start()]
    return text.strip()


# ── Aphoristic chunking ───────────────────────────────────────────────────────


def chunk_aphoristic(
    text: str,
    work_title: str,
    work_slug: str,
    work_period: Period,
    min_tokens: int = 50,
) -> list[Chunk]:
    """Chunk an aphoristic work by numbered section.

    Each ``N. ...`` line begins a new aphorism chunk.  Aphorisms shorter than
    *min_tokens* are merged into the following one.  When aphorisms are merged,
    ``aphorism_number`` holds the *first* number and ``aphorism_number_end``
    holds the *last*, preserving every original number from the source text.
    Text before the first numbered aphorism (preface, table of contents) is
    included as a preamble chunk with ``section_number=0,
    aphorism_number=None``.

    Args:
        text: Gutenberg-stripped work text.
        work_title: Human-readable title, e.g. ``"Beyond Good and Evil"``.
        work_slug: Filename slug, e.g. ``"beyond_good_and_evil"``.
        work_period: ``"early"``, ``"middle"``, or ``"late"``.
        min_tokens: Aphorisms below this token count are merged with the next.

    Returns:
        List of :class:`Chunk` objects in document order.
    """
    # Collect chapter and aphorism event positions, sorted by text position
    chapter_events: list[tuple[int, int]] = sorted(
        (m.start(), _roman_to_int(m.group(1))) for m in _CHAPTER_RE.finditer(text)
    )
    aphorism_events: list[tuple[int, int]] = [
        (m.start(), int(m.group(1))) for m in _APHORISM_RE.finditer(text)
    ]

    def _chapter_at(pos: int) -> int:
        """Return the chapter number that was in effect at *pos*."""
        ch = 0
        for ch_pos, ch_num in chapter_events:
            if ch_pos <= pos:
                ch = ch_num
            else:
                break
        return ch

    # ── Collect raw (chapter, aphorism_num, content) triples ─────────────────
    # Entry type: (chapter: int, aph_num: int | None, content: str)
    raw: list[tuple[int, int | None, str]] = []

    # Preamble: everything before the first numbered aphorism
    first_aph_pos = aphorism_events[0][0] if aphorism_events else len(text)
    preamble = text[:first_aph_pos].strip()
    if preamble and _token_count(preamble) >= min_tokens:
        raw.append((0, None, preamble))

    # Numbered aphorisms — slice from one marker to the next
    for i, (pos, aph_num) in enumerate(aphorism_events):
        end = aphorism_events[i + 1][0] if i + 1 < len(aphorism_events) else len(text)
        content = text[pos:end].strip()
        raw.append((_chapter_at(pos), aph_num, content))

    # ── Merge short aphorisms, preserving original numbering ─────────────────
    # Each entry: (chapter, aph_start, aph_end, content)
    # aph_start / aph_end are None only for the preamble chunk.
    merged: list[tuple[int, int | None, int | None, str]] = []
    i = 0
    while i < len(raw):
        ch, aph_start, content = raw[i]
        aph_end = aph_start  # track the last aphorism number absorbed
        while _token_count(content) < min_tokens and i + 1 < len(raw):
            i += 1
            _, next_aph, next_content = raw[i]
            content = content + "\n\n" + next_content
            # Update end pointer only for real aphorism numbers
            if next_aph is not None:
                aph_end = next_aph
        merged.append((ch, aph_start, aph_end, content))
        i += 1

    return [
        Chunk(
            content=content,
            work_title=work_title,
            work_slug=work_slug,
            work_period=work_period,
            section_number=ch,
            aphorism_number=aph_start,
            aphorism_number_end=aph_end,
            chunk_index=idx,
            chunk_type="aphorism",
        )
        for idx, (ch, aph_start, aph_end, content) in enumerate(merged)
    ]


# ── Prose / paragraph chunking ────────────────────────────────────────────────


def chunk_prose(
    text: str,
    work_title: str,
    work_slug: str,
    work_period: Period,
    target_tokens: int = 300,
    overlap_tokens: int = 100,
) -> list[Chunk]:
    """Chunk a prose/essay work by paragraph with token overlap.

    Paragraphs are accumulated until adding the next would exceed
    *target_tokens*, at which point the current window is emitted and the
    next chunk begins with *overlap_tokens* tokens carried over from the end
    of the previous chunk.

    Args:
        text: Gutenberg-stripped work text.
        work_title: Human-readable title.
        work_slug: Filename slug.
        work_period: ``"early"``, ``"middle"``, or ``"late"``.
        target_tokens: Soft token ceiling per chunk.
        overlap_tokens: Tokens to carry over between consecutive chunks.

    Returns:
        List of :class:`Chunk` objects in document order.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0
    chunk_index = 0
    section_number = 0

    for para in paragraphs:
        # Chapter headers advance the section counter but are not content
        ch_m = _CHAPTER_RE.match(para)
        if ch_m and _token_count(para) < 20:
            section_number = _roman_to_int(ch_m.group(1))
            continue

        para_tok = _token_count(para)

        if buf and buf_tokens + para_tok > target_tokens:
            content = "\n\n".join(buf)
            chunks.append(
                Chunk(
                    content=content,
                    work_title=work_title,
                    work_slug=work_slug,
                    work_period=work_period,
                    section_number=section_number,
                    aphorism_number=None,
                    aphorism_number_end=None,
                    chunk_index=chunk_index,
                    chunk_type="paragraph",
                )
            )
            chunk_index += 1

            # Seed next chunk with tail overlap from the emitted content
            overlap_text = _tail_tokens(content, overlap_tokens)
            buf = [overlap_text] if overlap_text else []
            buf_tokens = _token_count(overlap_text)

        buf.append(para)
        buf_tokens += para_tok

    # Flush remaining content
    if buf:
        chunks.append(
            Chunk(
                content="\n\n".join(buf),
                work_title=work_title,
                work_slug=work_slug,
                work_period=work_period,
                section_number=section_number,
                aphorism_number=None,
                aphorism_number_end=None,
                chunk_index=chunk_index,
                chunk_type="paragraph",
            )
        )

    return chunks


# ── Public API ────────────────────────────────────────────────────────────────


def chunk_work(
    text: str,
    work_title: str,
    work_slug: str,
    work_period: Period,
    chunk_style: ChunkStyle,
    min_aphorism_tokens: int = 50,
    target_prose_tokens: int = 300,
    overlap_tokens: int = 100,
) -> list[Chunk]:
    """Strip Gutenberg boilerplate and chunk a Nietzsche work.

    Args:
        text: Raw text as downloaded from Project Gutenberg.
        work_title: Human-readable title, e.g. ``"Beyond Good and Evil"``.
        work_slug: Filename slug, e.g. ``"beyond_good_and_evil"``.
        work_period: ``"early"``, ``"middle"``, or ``"late"``.
        chunk_style: ``"aphorism"`` for numbered-section works,
            ``"paragraph"`` for essay-style prose.
        min_aphorism_tokens: Minimum token count before merging short aphorisms.
        target_prose_tokens: Target token count for prose chunks.
        overlap_tokens: Token overlap between consecutive prose chunks.

    Returns:
        List of :class:`Chunk` objects ready for embedding.
    """
    clean = strip_gutenberg_boilerplate(text)
    console.print(
        f"[cyan]Chunking[/cyan] {work_title!r} as [bold]{chunk_style}[/bold] "
        f"({_token_count(clean):,} tokens after boilerplate strip)"
    )

    if chunk_style == "aphorism":
        chunks = chunk_aphoristic(
            clean, work_title, work_slug, work_period,
            min_tokens=min_aphorism_tokens,
        )
    else:
        chunks = chunk_prose(
            clean, work_title, work_slug, work_period,
            target_tokens=target_prose_tokens,
            overlap_tokens=overlap_tokens,
        )

    console.print(f"  → [green]{len(chunks)} chunks[/green] produced")
    return chunks


def chunk_file(
    path: Path,
    work_title: str,
    work_slug: str,
    work_period: Period,
    chunk_style: ChunkStyle,
    **kwargs,
) -> list[Chunk]:
    """Read *path* and chunk its contents.

    Convenience wrapper around :func:`chunk_work`.

    Args:
        path: Path to the plain-text ``.txt`` file.
        work_title: Human-readable title.
        work_slug: Filename slug.
        work_period: ``"early"``, ``"middle"``, or ``"late"``.
        chunk_style: ``"aphorism"`` or ``"paragraph"``.
        **kwargs: Forwarded to :func:`chunk_work`.

    Returns:
        List of :class:`Chunk` objects.
    """
    text = path.read_text(encoding="utf-8")
    return chunk_work(text, work_title, work_slug, work_period, chunk_style, **kwargs)
