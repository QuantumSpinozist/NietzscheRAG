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
    EvalItem(
        question="What is Nietzsche's critique of free will and the concept of causa sui?",
        ground_truth=[
            ("beyond_good_and_evil", 21),   # primary: 'causa sui' as the best self-contradiction — canonical free will critique
        ],
        note="BGE §21 — the causa sui passage; Nietzsche dismantles both free and unfree will as metaphysical errors",
    ),
    EvalItem(
        question="How did knowledge and logic emerge from false beliefs according to Nietzsche?",
        ground_truth=[
            ("the_gay_science", 110),   # primary: 'Origin of Knowledge' — false beliefs as incorporated errors underlying logic
            ("the_gay_science", 111),   # secondary: 'Origin of the Logical' — directly continues the argument
        ],
        note="GS §110 'Origin of Knowledge'; GS §111 'Origin of the Logical' continues the same argument",
    ),
    EvalItem(
        question="What does Nietzsche say about nobility and the pathos of distance?",
        ground_truth=[
            ("beyond_good_and_evil", 257),  # primary: 'What is noble?' — the founding act of aristocratic society
            ("beyond_good_and_evil", 259),  # secondary: life as appropriation / will to power as the noble principle
        ],
        note="BGE §257 defines nobility via the pathos of distance; BGE §259 grounds it in will to power",
    ),
    EvalItem(
        question="What does Nietzsche say about great suffering as a precondition of greatness?",
        ground_truth=[
            ("beyond_good_and_evil", 225),  # primary: 'The discipline of suffering, of great suffering'
        ],
        note="BGE §225 — the discipline of great suffering; Nietzsche's most direct statement on suffering and depth",
    ),
    EvalItem(
        question="What does Nietzsche say about the value of untruth and false judgments as conditions of life?",
        ground_truth=[
            ("beyond_good_and_evil", 4),    # primary: 'the falseness of a judgment is not an objection to it'
            ("beyond_good_and_evil", 34),   # secondary: perspectivism — every philosophy is a confession of its author
        ],
        note="BGE §4 is the epistemological provocation; BGE §34 extends it to perspectivism",
    ),
]
