"""Evaluation set: 10 questions with ground-truth chunk identifiers.

Each entry specifies:
  - question: the natural-language query
  - ground_truth: list of (work_slug, aphorism_number) pairs.
    A retrieval run is considered a *hit* if ANY ground-truth chunk
    appears within the top-k results.

Ground truth was established by locating the specific aphorism in the
raw corpus text that most directly answers each question.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalItem:
    question: str
    # (work_slug, aphorism_number) — aphorism_number=-1 for prose chunks
    ground_truth: list[tuple[str, int]]
    # Human note about what the correct answer is
    note: str = ""


EVAL_SET: list[EvalItem] = [
    EvalItem(
        question="What is Nietzsche's announcement of the death of God?",
        ground_truth=[("the_gay_science", 125)],
        note="GS §125 'The Madman' — the lantern-in-the-marketplace passage",
    ),
    EvalItem(
        question="What is the eternal recurrence or the greatest weight?",
        ground_truth=[("the_gay_science", 341)],
        note="GS §341 'The Heaviest Burden' — the demon thought experiment",
    ),
    EvalItem(
        question="What does the death of God mean for modern European culture?",
        ground_truth=[("the_gay_science", 343)],
        note="GS §343 'What our Cheerfulness Signifies' — consequences of God's death",
    ),
    EvalItem(
        question="What is Nietzsche's distinction between noble morality and slave morality?",
        ground_truth=[("beyond_good_and_evil", 260)],
        note="BGE §260 — the master/slave morality taxonomy",
    ),
    EvalItem(
        question="What does Nietzsche say about the philosopher of the future?",
        ground_truth=[("beyond_good_and_evil", 212)],
        note="BGE §212 — philosophers as commanders and legislators",
    ),
    EvalItem(
        question="What is Nietzsche's critique of Christianity and the religion of pity?",
        ground_truth=[("the_antichrist", 7)],
        note="AC §7 — pity as anti-life, opposition to tonic passions",
    ),
    EvalItem(
        question="What is the will to power and how does it relate to self-preservation?",
        ground_truth=[("beyond_good_and_evil", 13)],
        note="BGE §13 — will to power as more fundamental than self-preservation instinct",
    ),
    EvalItem(
        question="What does Nietzsche say about amor fati and loving what is necessary?",
        ground_truth=[("the_gay_science", 276)],
        note="GS §276 'For the New Year' — 'I want to learn more and more to see what is necessary as beautiful'",
    ),
    EvalItem(
        question="What is Nietzsche's view on consciousness and the genius of the species?",
        ground_truth=[("the_gay_science", 354)],
        note="GS §354 — consciousness as herd instinct, perspectivism of species",
    ),
    EvalItem(
        question="What does Nietzsche say about the revaluation of all values?",
        ground_truth=[("the_antichrist", 62)],
        note="AC §62 — revaluation of all values, the anti-Christian project",
    ),
]
