"""Prompt builder and Claude API call for Nietzsche RAG generation."""

from __future__ import annotations

import re
import anthropic

import config
from retrieval.hybrid import HybridResult


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a philosophical research assistant specialising in Friedrich Nietzsche.\n"
    "Answer the user's question using ONLY the provided source passages.\n"
    "For every claim you make, cite the source work and section number in brackets, "
    "e.g. [BGE §36].\n"
    "If the passages do not contain enough information to answer, say so explicitly.\n"
    "Distinguish between what Nietzsche literally says and scholarly interpretation.\n"
    "Do not editorialize or inject views not supported by the passages."
)


# ── Prompt builder ────────────────────────────────────────────────────────────


def build_prompt(question: str, results: list[HybridResult]) -> str:
    """Build the user-turn prompt from retrieved passages and a question.

    This is a pure function — it never calls the API.

    Args:
        question: The user's natural-language question.
        results: Retrieved passages from :func:`hybrid_search`, in ranked order.

    Returns:
        A formatted string ready to send as the ``user`` message to Claude.
    """
    parts: list[str] = ["Source passages:"]

    for r in results:
        work_title = r.metadata.get("work_title", "Unknown")
        section = r.metadata.get("section_number", "?")
        aph = r.metadata.get("aphorism_number")
        aph_end = r.metadata.get("aphorism_number_end", aph)

        # Build a human-readable source label
        if aph is not None and aph != -1:
            if aph_end is not None and aph_end != aph and aph_end != -1:
                source_label = f"{work_title}, §{aph}–{aph_end}"
            else:
                source_label = f"{work_title}, §{aph}"
        else:
            source_label = f"{work_title}, ch.{section}"

        parts.append("---")
        parts.append(r.document)
        parts.append(f"Source: {source_label}")

    parts.append("---")
    parts.append("")
    parts.append(f"Question: {question}")

    return "\n".join(parts)


# ── Claude API call ───────────────────────────────────────────────────────────


def generate_answer(
    question: str,
    results: list[HybridResult],
    model: str = config.GENERATION_MODEL,
    max_tokens: int = 1024,
) -> str:
    """Send the prompt to Claude and return the generated answer.

    Args:
        question: The user's natural-language question.
        results: Retrieved passages to ground the answer.
        model: Claude model identifier (default: ``config.GENERATION_MODEL``).
        max_tokens: Maximum tokens in the response.

    Returns:
        The assistant's text response as a plain string.

    Raises:
        anthropic.APIError: On any API-level error.
    """
    prompt = build_prompt(question, results)

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


# ── Citation parsing ──────────────────────────────────────────────────────────


def parse_used_chunk_ids(answer: str, results: list[HybridResult]) -> set[str]:
    """Return the IDs of chunks whose section number is cited in *answer*.

    Scans for ``§N`` patterns in the generated answer and matches them against
    the ``aphorism_number`` or ``section_number`` metadata of each result.

    Args:
        answer: The generated answer text (may contain ``§N`` citations).
        results: The same result list that was passed to :func:`generate_answer`.

    Returns:
        Set of chunk IDs that were cited at least once.
    """
    cited: set[int] = {int(m) for m in re.findall(r"§(\d+)", answer)}
    used: set[str] = set()
    for r in results:
        aph = r.metadata.get("aphorism_number")
        sec = r.metadata.get("section_number")
        if (aph is not None and aph != -1 and int(aph) in cited) or (
            sec is not None and int(sec) in cited
        ):
            used.add(r.id)
    return used
