"""HyDE — Hypothetical Document Embeddings for improved dense retrieval.

Instead of embedding the raw user question, we ask Claude to write a short
passage *in Nietzsche's own voice* that would answer the question, then embed
that hypothetical passage.  The resulting embedding sits much closer to the
actual corpus chunks than a bare question embedding would, which is especially
valuable for aphoristic works where the question and the canonical passage
share few surface tokens.

Reference: Gao et al. (2022) "Precise Zero-Shot Dense Retrieval without
Relevance Labels" (HyDE paper).
"""

from __future__ import annotations

import anthropic

import config

_HYDE_SYSTEM = (
    "You are Friedrich Nietzsche. Write a single short passage — 2 to 4 sentences — "
    "in your own authentic philosophical voice that directly addresses the question asked. "
    "Use your characteristic aphoristic style: bold declarations, rhetorical questions, "
    "sharp inversions. Draw on your actual vocabulary where appropriate "
    "(will to power, eternal recurrence, ressentiment, Übermensch, master/slave morality, "
    "amor fati, Dionysian, revaluation of values, etc.). "
    "Do NOT explain or summarise — write as you would in Beyond Good and Evil or "
    "The Gay Science. Do not add any preamble or attribution."
)


def generate_hypothetical_passage(
    question: str,
    model: str = config.HYDE_MODEL,
    max_tokens: int = 200,
) -> str:
    """Ask Claude to write a hypothetical Nietzsche passage for *question*.

    The returned text is intended to be embedded (not shown to the user) — it
    serves as a better dense-search proxy than the raw question text.

    Args:
        question: The user's natural-language question.
        model: Claude model to use for generation (default: haiku for speed).
        max_tokens: Maximum length of the hypothetical passage.

    Returns:
        A short passage written in Nietzsche's voice.
    """
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_HYDE_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )
    return message.content[0].text
