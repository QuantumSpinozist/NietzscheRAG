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
        ground_truth=[
            ("the_gay_science", 125),   # primary: 'The Madman' — the lantern passage
            ("the_gay_science", 343),   # secondary: 'What our Cheerfulness Signifies' — also opens with the death of God
        ],
        note="GS §125 'The Madman'; GS §343 reciprocally relevant as it opens by naming the event",
    ),
    EvalItem(
        question="What is the eternal recurrence or the greatest weight?",
        ground_truth=[("the_gay_science", 341)],
        note="GS §341 'The Heaviest Burden' — the demon thought experiment; uniquely canonical",
    ),
    EvalItem(
        question="What does the death of God mean for modern European culture?",
        ground_truth=[
            ("the_gay_science", 343),   # primary: 'What our Cheerfulness Signifies'
            ("the_gay_science", 125),   # secondary: 'The Madman' describes the cultural crisis directly
        ],
        note="GS §343 is the direct answer; GS §125 describes the same cultural rupture",
    ),
    EvalItem(
        question="What is Nietzsche's distinction between noble morality and slave morality?",
        ground_truth=[("beyond_good_and_evil", 260)],
        note="BGE §260 — the master/slave morality taxonomy; uniquely canonical",
    ),
    EvalItem(
        question="What does Nietzsche say about the philosopher of the future?",
        ground_truth=[
            ("beyond_good_and_evil", 212),  # primary: philosophers as commanders and legislators
            ("beyond_good_and_evil", 211),  # secondary: 'A New Species of Philosophers' — direct precursor
        ],
        note="BGE §212 is canonical; §211 names the new species and is part of the same argument",
    ),
    EvalItem(
        question="What is Nietzsche's critique of Christianity and the religion of pity?",
        ground_truth=[("the_antichrist", 7)],
        note="AC §7 — pity as anti-life, opposition to tonic passions; uniquely canonical",
    ),
    EvalItem(
        question="What is the will to power and how does it relate to self-preservation?",
        ground_truth=[
            ("beyond_good_and_evil", 13),   # primary: will to power as more fundamental than self-preservation
            ("beyond_good_and_evil", 36),   # secondary: will to power as the intelligible character of existence
        ],
        note="BGE §13 is the Darwinian critique; BGE §36 is the positive metaphysical formulation",
    ),
    EvalItem(
        question="What does Nietzsche say about amor fati and loving what is necessary?",
        ground_truth=[("the_gay_science", 276)],
        note="GS §276 'For the New Year' — first occurrence of amor fati as a term; uniquely canonical",
    ),
    EvalItem(
        question="What is Nietzsche's view on consciousness and the genius of the species?",
        ground_truth=[("the_gay_science", 354)],
        note="GS §354 — consciousness as herd instinct, perspectivism of species; uniquely canonical",
    ),
    EvalItem(
        question="What does Nietzsche say about the revaluation of all values?",
        ground_truth=[
            ("the_antichrist", 62),  # primary: the programmatic statement of the anti-Christian revaluation
            ("the_antichrist", 61),  # secondary: immediate setup — condemns the Church and names the counter-movement
        ],
        note="AC §62 is the climax; §61 names the same project and is part of the same argument",
    ),
]
