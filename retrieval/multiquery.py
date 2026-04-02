"""Multi-query expansion: generate N paraphrases of a question via Claude Haiku,
retrieve dense candidates for each, then union all lists before RRF merging."""

from __future__ import annotations

import anthropic

import config

_MULTIQUERY_SYSTEM = (
    "You are a helpful assistant. Your task is to rephrase a philosophical question "
    "into a small number of alternative formulations that preserve the meaning but "
    "use different vocabulary and emphasis. Each rephrasing should be a standalone "
    "question that could independently retrieve the relevant passage from a corpus. "
    "Focus on varying the philosophical terminology — for example, 'eternal recurrence' "
    "could become 'the thought of eternal return' or 'the heaviest burden thought experiment'. "
    "Output ONLY the rephrased questions, one per line, with no numbering, preamble, or explanation."
)


def generate_query_variants(
    question: str,
    n: int = 2,
    model: str = config.HYDE_MODEL,
    max_tokens: int = 150,
) -> list[str]:
    """Generate *n* paraphrased variants of *question* using Claude Haiku.

    Args:
        question: Original user question.
        n: Number of variants to generate (default 2).
        model: Claude model for generation (default: haiku, cheap/fast).
        max_tokens: Max tokens for the response.

    Returns:
        List of variant strings (length <= n; may be shorter if the model
        returns fewer lines).
    """
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_MULTIQUERY_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Generate exactly {n} rephrased versions of this question:\n\n{question}",
            }
        ],
    )
    raw = message.content[0].text.strip()
    variants = [line.strip() for line in raw.splitlines() if line.strip()]
    return variants[:n]
