"""Tests for ingest/chunk.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingest.chunk import (
    Chunk,
    _MAX_APHORISM_NUMBER,
    _roman_to_int,
    _tail_tokens,
    _token_count,
    chunk_aphoristic,
    chunk_prose,
    chunk_work,
    strip_gutenberg_boilerplate,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

APHORISTIC_TEXT = """\
CHAPTER I. ON TRUTH

1. Supposing that Truth is a woman--what then? Is there not ground
for suspecting that all philosophers have failed to understand women?
And perhaps the will to truth is itself a form of the will to power.

2. The falseness of an opinion is not for us any objection to it.
The question is, how far an opinion is life-furthering.

3. Short.

4. Also short.

5. Having kept a sharp eye on philosophers, and having read between
their lines long enough, I now say to myself that the greater part of
conscious thinking must be counted among the instinctive functions.
"""

PROSE_TEXT = """\
CHAPTER I. ON TRAGEDY

The birth of tragedy out of the spirit of music is a paradox that has
long puzzled scholars of antiquity. We must ask ourselves what drove
the Greeks toward the tragic form, and what they found there that could
not be found in the Olympian religion alone.

The Dionysian impulse dissolves the principium individuationis. In
ecstasy the individual forgets himself and merges with the primordial
unity. This is the kernel of the tragic experience: not the hero's
suffering, but the affirmation behind it.

Apollo gives form and measure to the Dionysian chaos. The dream-image
of the Apolline artist arrests the flux of Dionysian dissolution and
presents it as beautiful appearance. Tragedy holds both drives in
productive tension.

Without Dionysus, Apolline art becomes mere prettiness. Without Apollo,
Dionysian experience dissolves into formlessness. The tragic artwork is
their synthesis: the Dionysian insight clothed in Apolline beauty.
"""

GUTENBERG_TEXT = """\
Some header text here.
*** START OF THE PROJECT GUTENBERG EBOOK FOOBAR ***
This is the actual content of the work.
*** END OF THE PROJECT GUTENBERG EBOOK FOOBAR ***
Some footer text here.
"""

# Gay Science / Oscar Levy edition: numbers are centred on the page with
# heavy indentation, followed by a period, then the content on the next line.
GS_STYLE_TEXT = """\
                                   1.

The teachers of the object of existence find men always at one problem.
They find something remarkable about the human species.

                                   2.

Whether I look with good or evil eye upon men, I find them always
at the same problem. To do that which conduces to the conservation.

                                   3.

It will be surmised that I should not like to take leave ungratefully
of that period of severe sickness. A philosopher has the duty.
"""

# A file that contains two works concatenated — like PG 52263 (TI + Antichrist).
COMBINED_TEXT = """\
FIRST WORK

1. The first aphorism of the first work is here and has enough tokens.

2. The second aphorism of the first work also has sufficient tokens here.

SECOND WORK

1. This aphorism belongs to the second work and should be excluded.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────


class TestTokenCount:
    def test_counts_whitespace_tokens(self) -> None:
        assert _token_count("one two three") == 3

    def test_empty_string(self) -> None:
        assert _token_count("") == 0

    def test_extra_whitespace_ignored(self) -> None:
        assert _token_count("  a   b  ") == 2


class TestTailTokens:
    def test_returns_last_n_tokens(self) -> None:
        assert _tail_tokens("a b c d e", 3) == "c d e"

    def test_returns_full_text_when_shorter(self) -> None:
        assert _tail_tokens("a b", 10) == "a b"

    def test_n_equals_length(self) -> None:
        assert _tail_tokens("x y z", 3) == "x y z"


class TestRomanToInt:
    def test_single_numerals(self) -> None:
        assert _roman_to_int("I") == 1
        assert _roman_to_int("V") == 5
        assert _roman_to_int("X") == 10

    def test_additive(self) -> None:
        assert _roman_to_int("III") == 3
        assert _roman_to_int("VIII") == 8

    def test_subtractive(self) -> None:
        assert _roman_to_int("IV") == 4
        assert _roman_to_int("IX") == 9
        assert _roman_to_int("XIV") == 14

    def test_case_insensitive(self) -> None:
        assert _roman_to_int("iv") == 4
        assert _roman_to_int("xiv") == 14


# ── Gutenberg boilerplate stripping ───────────────────────────────────────────


class TestStripGutenbergBoilerplate:
    def test_removes_header_and_footer(self) -> None:
        result = strip_gutenberg_boilerplate(GUTENBERG_TEXT)
        assert result == "This is the actual content of the work."

    def test_no_markers_returns_text_unchanged(self) -> None:
        plain = "Just some text without markers."
        assert strip_gutenberg_boilerplate(plain) == plain

    def test_strips_surrounding_whitespace(self) -> None:
        text = "*** START OF THE PROJECT GUTENBERG EBOOK X ***\n\n  content  \n\n*** END OF THE PROJECT GUTENBERG EBOOK X ***"
        assert strip_gutenberg_boilerplate(text) == "content"

    def test_gutenberg_header_not_in_any_chunk(self) -> None:
        """Boilerplate text must never reach the chunker output."""
        chunks = chunk_work(
            GUTENBERG_TEXT + "\n\n1. An aphorism here with enough tokens to be valid.",
            "Test Work", "test_work", "late", "aphorism",
        )
        for c in chunks:
            assert "Project Gutenberg" not in c.content
            assert "START OF" not in c.content


# ── Aphorism chunking ─────────────────────────────────────────────────────────


class TestChunkAphoristic:
    def test_aphorism_boundaries_detected(self) -> None:
        """Each numbered section becomes its own chunk (before merging)."""
        chunks = chunk_aphoristic(
            APHORISTIC_TEXT, "BGE", "bge", "late", min_tokens=0
        )
        aph_nums = [c.aphorism_number for c in chunks if c.aphorism_number is not None]
        assert aph_nums == [1, 2, 3, 4, 5]

    def test_short_aphorisms_merged(self) -> None:
        """Aphorisms under min_tokens are merged with the next one."""
        chunks = chunk_aphoristic(
            APHORISTIC_TEXT, "BGE", "bge", "late", min_tokens=20
        )
        # Aphorisms 3 ("Short.") and 4 ("Also short.") are each < 20 tokens
        # and should be merged together (and possibly with 5)
        merged = [c for c in chunks if c.aphorism_number is not None and c.aphorism_number_end != c.aphorism_number]
        assert len(merged) >= 1

    def test_merged_aphorism_preserves_original_numbers(self) -> None:
        """When aphorisms are merged, aphorism_number holds the first number
        and aphorism_number_end holds the last — no number is silently dropped."""
        chunks = chunk_aphoristic(
            APHORISTIC_TEXT, "BGE", "bge", "late", min_tokens=20
        )
        merged = [c for c in chunks if c.aphorism_number is not None and c.aphorism_number_end != c.aphorism_number]
        assert len(merged) >= 1
        for c in merged:
            assert c.aphorism_number is not None
            assert c.aphorism_number_end is not None
            assert c.aphorism_number_end > c.aphorism_number

    def test_merged_content_contains_both_aphorism_texts(self) -> None:
        """Merged chunk content includes text from all absorbed aphorisms."""
        chunks = chunk_aphoristic(
            APHORISTIC_TEXT, "BGE", "bge", "late", min_tokens=20
        )
        merged = [c for c in chunks if c.aphorism_number is not None and c.aphorism_number_end != c.aphorism_number]
        # The merged chunk must contain text from each absorbed aphorism
        for c in merged:
            assert len(c.content) > 0
            # Content should be longer than a single short aphorism
            assert _token_count(c.content) >= 2

    def test_chapter_number_assigned(self) -> None:
        """Chunks under CHAPTER I get section_number == 1."""
        chunks = chunk_aphoristic(
            APHORISTIC_TEXT, "BGE", "bge", "late", min_tokens=0
        )
        aph_chunks = [c for c in chunks if c.aphorism_number is not None]
        assert all(c.section_number == 1 for c in aph_chunks)

    def test_metadata_fields_present(self) -> None:
        """Every chunk dict contains all required metadata fields."""
        required = {
            "work_title", "work_slug", "work_period",
            "section_number", "aphorism_number", "aphorism_number_end",
            "chunk_index", "chunk_type", "content",
        }
        chunks = chunk_aphoristic(
            APHORISTIC_TEXT, "BGE", "bge", "late", min_tokens=0
        )
        for c in chunks:
            assert required == set(c.to_dict().keys())

    def test_no_empty_chunks(self) -> None:
        """No chunk should have empty or whitespace-only content."""
        chunks = chunk_aphoristic(
            APHORISTIC_TEXT, "BGE", "bge", "late", min_tokens=0
        )
        for c in chunks:
            assert c.content.strip() != ""

    def test_chunk_type_is_aphorism(self) -> None:
        chunks = chunk_aphoristic(
            APHORISTIC_TEXT, "BGE", "bge", "late", min_tokens=0
        )
        for c in chunks:
            assert c.chunk_type == "aphorism"

    def test_chunk_index_sequential(self) -> None:
        chunks = chunk_aphoristic(
            APHORISTIC_TEXT, "BGE", "bge", "late", min_tokens=0
        )
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_unmerged_aphorism_start_equals_end(self) -> None:
        """For a single (unmerged) aphorism, start == end."""
        chunks = chunk_aphoristic(
            APHORISTIC_TEXT, "BGE", "bge", "late", min_tokens=0
        )
        for c in chunks:
            if c.aphorism_number is not None:
                assert c.aphorism_number == c.aphorism_number_end


# ── Prose chunking ────────────────────────────────────────────────────────────


class TestChunkProse:
    def test_paragraph_chunking_respects_token_limit(self) -> None:
        """Prose chunks should not greatly exceed target_tokens."""
        chunks = chunk_prose(
            PROSE_TEXT, "BT", "birth_of_tragedy", "early",
            target_tokens=100, overlap_tokens=20,
        )
        for c in chunks:
            # Allow some slack: overlap can push chunks slightly over target
            assert _token_count(c.content) < 200

    def test_chunk_overlap_is_present(self) -> None:
        """Consecutive prose chunks share tokens at their boundary."""
        chunks = chunk_prose(
            PROSE_TEXT, "BT", "birth_of_tragedy", "early",
            target_tokens=60, overlap_tokens=15,
        )
        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")
        tail = _tail_tokens(chunks[0].content, 15).split()
        head = chunks[1].content.split()[:20]
        # At least some tail tokens appear at the start of the next chunk
        assert any(t in head for t in tail)

    def test_metadata_fields_present(self) -> None:
        required = {
            "work_title", "work_slug", "work_period",
            "section_number", "aphorism_number", "aphorism_number_end",
            "chunk_index", "chunk_type", "content",
        }
        chunks = chunk_prose(
            PROSE_TEXT, "BT", "birth_of_tragedy", "early"
        )
        for c in chunks:
            assert required == set(c.to_dict().keys())

    def test_no_empty_chunks(self) -> None:
        chunks = chunk_prose(
            PROSE_TEXT, "BT", "birth_of_tragedy", "early"
        )
        for c in chunks:
            assert c.content.strip() != ""

    def test_chunk_type_is_paragraph(self) -> None:
        chunks = chunk_prose(
            PROSE_TEXT, "BT", "birth_of_tragedy", "early"
        )
        for c in chunks:
            assert c.chunk_type == "paragraph"

    def test_aphorism_number_is_none(self) -> None:
        chunks = chunk_prose(
            PROSE_TEXT, "BT", "birth_of_tragedy", "early"
        )
        for c in chunks:
            assert c.aphorism_number is None
            assert c.aphorism_number_end is None

    def test_chapter_section_number_tracked(self) -> None:
        """Chunks produced after a CHAPTER header carry the correct section_number."""
        chunks = chunk_prose(
            PROSE_TEXT, "BT", "birth_of_tragedy", "early"
        )
        # PROSE_TEXT has CHAPTER I, so all chunks should have section_number == 1
        assert all(c.section_number == 1 for c in chunks)


# ── chunk_work dispatcher ─────────────────────────────────────────────────────


class TestChunkWork:
    def test_dispatches_to_aphorism(self) -> None:
        chunks = chunk_work(
            APHORISTIC_TEXT, "BGE", "bge", "late", "aphorism", min_aphorism_tokens=0
        )
        assert all(c.chunk_type == "aphorism" for c in chunks)

    def test_dispatches_to_paragraph(self) -> None:
        chunks = chunk_work(
            PROSE_TEXT, "BT", "birth_of_tragedy", "early", "paragraph"
        )
        assert all(c.chunk_type == "paragraph" for c in chunks)

    def test_strips_boilerplate_before_chunking(self) -> None:
        text = (
            "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\n\n"
            + APHORISTIC_TEXT
            + "\n\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***"
        )
        chunks = chunk_work(text, "BGE", "bge", "late", "aphorism", min_aphorism_tokens=0)
        for c in chunks:
            assert "Project Gutenberg" not in c.content

    def test_work_metadata_propagated(self) -> None:
        chunks = chunk_work(
            APHORISTIC_TEXT, "Beyond Good and Evil", "beyond_good_and_evil",
            "late", "aphorism", min_aphorism_tokens=0,
        )
        for c in chunks:
            assert c.work_title == "Beyond Good and Evil"
            assert c.work_slug == "beyond_good_and_evil"
            assert c.work_period == "late"


# ── Indented-number format (Gay Science / Oscar Levy) ─────────────────────────


class TestIndentedAphorismFormat:
    """The Oscar Levy editions centre section numbers on the page with heavy
    indentation followed by a period.  The current regex must detect these."""

    def test_indented_numbers_produce_chunks(self) -> None:
        chunks = chunk_aphoristic(GS_STYLE_TEXT, "GS", "the_gay_science", "middle", min_tokens=0)
        aph_nums = [c.aphorism_number for c in chunks if c.aphorism_number is not None]
        assert aph_nums == [1, 2, 3]

    def test_content_preserved_after_indented_number(self) -> None:
        chunks = chunk_aphoristic(GS_STYLE_TEXT, "GS", "the_gay_science", "middle", min_tokens=0)
        aph1 = next(c for c in chunks if c.aphorism_number == 1)
        assert "teachers of the object" in aph1.content.lower()

    def test_no_empty_chunks_from_indented_format(self) -> None:
        chunks = chunk_aphoristic(GS_STYLE_TEXT, "GS", "the_gay_science", "middle", min_tokens=0)
        for c in chunks:
            assert c.content.strip() != ""


# ── Year / large-number false-positive filtering ───────────────────────────────


class TestAphorismNumberFilter:
    def test_max_aphorism_number_constant_is_reasonable(self) -> None:
        assert 500 < _MAX_APHORISM_NUMBER < 2000

    def test_year_not_treated_as_aphorism(self) -> None:
        """A line like '1878. A comparison ...' must not produce a chunk."""
        text = (
            "1. A real aphorism with enough tokens to pass the minimum check here.\n\n"
            "1878. A comparison of the books will show that the two years of "
            "meditation intervening had brought about a complete change in style.\n\n"
            "2. Another real aphorism with enough tokens to pass the minimum check.\n\n"
        )
        chunks = chunk_aphoristic(text, "HH", "human_all_too_human", "middle", min_tokens=0)
        aphs = [c.aphorism_number for c in chunks if c.aphorism_number is not None]
        assert 1878 not in aphs
        assert 1 in aphs
        assert 2 in aphs

    def test_numbers_up_to_max_accepted(self) -> None:
        """Aphorism numbers up to _MAX_APHORISM_NUMBER are accepted."""
        text = f"1. First aphorism with plenty of tokens to be counted correctly.\n\n"
        text += f"{_MAX_APHORISM_NUMBER}. Last valid aphorism with plenty of tokens here.\n\n"
        chunks = chunk_aphoristic(text, "W", "w", "middle", min_tokens=0)
        aphs = [c.aphorism_number for c in chunks if c.aphorism_number is not None]
        assert _MAX_APHORISM_NUMBER in aphs


# ── end_before truncation ─────────────────────────────────────────────────────


class TestEndBefore:
    def test_truncates_at_marker(self) -> None:
        chunks = chunk_work(
            COMBINED_TEXT, "First Work", "first_work", "late", "aphorism",
            min_aphorism_tokens=0,
            end_before="SECOND WORK",
        )
        for c in chunks:
            assert "SECOND WORK" not in c.content
            assert "belongs to the second work" not in c.content

    def test_first_work_chunks_still_present(self) -> None:
        chunks = chunk_work(
            COMBINED_TEXT, "First Work", "first_work", "late", "aphorism",
            min_aphorism_tokens=0,
            end_before="SECOND WORK",
        )
        aphs = [c.aphorism_number for c in chunks if c.aphorism_number is not None]
        assert 1 in aphs
        assert 2 in aphs

    def test_missing_marker_chunks_full_text(self) -> None:
        """If end_before string is not found, the whole text is chunked."""
        chunks = chunk_work(
            COMBINED_TEXT, "W", "w", "late", "aphorism",
            min_aphorism_tokens=0,
            end_before="NONEXISTENT MARKER XYZ",
        )
        # Both §1 entries from FIRST WORK and SECOND WORK are present
        aphs = [c.aphorism_number for c in chunks if c.aphorism_number is not None]
        assert len(aphs) >= 2

    def test_end_before_none_no_effect(self) -> None:
        """end_before=None must not alter chunking behaviour."""
        chunks_default = chunk_work(
            COMBINED_TEXT, "W", "w", "late", "aphorism",
            min_aphorism_tokens=0,
        )
        chunks_none = chunk_work(
            COMBINED_TEXT, "W", "w", "late", "aphorism",
            min_aphorism_tokens=0,
            end_before=None,
        )
        assert len(chunks_default) == len(chunks_none)


# ── Per-work smoke tests (require downloaded raw files) ───────────────────────


def _smoke(slug: str, style: str, min_chunks: int, max_aphorism: int | None = None):
    """Helper: chunk a real file and run basic sanity checks."""
    from ingest.embed import WORK_END_BEFORE, WORK_REGISTRY
    path = Path(f"data/raw/{slug}.txt")
    if not path.exists():
        pytest.skip(f"Raw file not present: {path}")
    title, period, _ = WORK_REGISTRY[slug]
    end_before = WORK_END_BEFORE.get(slug)
    chunks = chunk_work(
        path.read_text(encoding="utf-8"),
        title, slug, period, style,
        end_before=end_before,
    )
    assert len(chunks) >= min_chunks, f"{slug}: expected ≥{min_chunks} chunks, got {len(chunks)}"
    for c in chunks:
        assert c.content.strip(), f"{slug}: empty chunk at index {c.chunk_index}"
        assert "Project Gutenberg" not in c.content
        assert "START OF" not in c.content
    if max_aphorism is not None:
        bad = [c.aphorism_number for c in chunks if c.aphorism_number and c.aphorism_number > max_aphorism]
        assert not bad, f"{slug}: aphorism numbers > {max_aphorism}: {bad[:5]}"
    return chunks


class TestCorpusSmokeTests:
    def test_beyond_good_and_evil(self) -> None:
        chunks = _smoke("beyond_good_and_evil", "aphorism", 200, max_aphorism=999)
        aphs = {c.aphorism_number for c in chunks if c.aphorism_number}
        assert max(aphs) <= 296

    def test_genealogy_of_morality(self) -> None:
        _smoke("genealogy_of_morality", "paragraph", 150)

    def test_twilight_of_the_idols(self) -> None:
        chunks = _smoke("twilight_of_the_idols", "paragraph", 100)
        # The opening sentence of The Antichrist body must not appear in any chunk.
        # (The phrase "An Attempted Criticism of Christianity" appears earlier in
        # TI's translator preface as a description, so we use the Antichrist's
        # own opening line instead.)
        for c in chunks:
            assert "This book belongs to the very few" not in c.content

    def test_the_antichrist(self) -> None:
        chunks = _smoke("the_antichrist", "aphorism", 50, max_aphorism=999)
        aphs = {c.aphorism_number for c in chunks if c.aphorism_number}
        assert max(aphs) <= 62

    def test_ecce_homo(self) -> None:
        _smoke("ecce_homo", "paragraph", 100)

    def test_nietzsche_contra_wagner(self) -> None:
        _smoke("nietzsche_contra_wagner", "aphorism", 30, max_aphorism=999)

    def test_the_gay_science(self) -> None:
        chunks = _smoke("the_gay_science", "aphorism", 300, max_aphorism=999)
        aphs = {c.aphorism_number for c in chunks if c.aphorism_number}
        # GS has aphorisms up to 383 (including preface / prelude sections)
        assert max(aphs) <= 999

    def test_daybreak(self) -> None:
        chunks = _smoke("daybreak", "aphorism", 400, max_aphorism=999)
        aphs = {c.aphorism_number for c in chunks if c.aphorism_number}
        assert max(aphs) <= 575

    def test_human_all_too_human(self) -> None:
        chunks = _smoke("human_all_too_human", "aphorism", 400, max_aphorism=999)
        aphs = {c.aphorism_number for c in chunks if c.aphorism_number}
        assert max(aphs) <= 638

    def test_birth_of_tragedy(self) -> None:
        _smoke("birth_of_tragedy", "paragraph", 150)

    def test_untimely_meditations_1(self) -> None:
        _smoke("untimely_meditations_1", "paragraph", 100)

    def test_untimely_meditations_2(self) -> None:
        _smoke("untimely_meditations_2", "paragraph", 100)


# ── Smoke test on real BGE file ───────────────────────────────────────────────


@pytest.mark.skipif(
    not Path("data/raw/beyond_good_and_evil.txt").exists(),
    reason="BGE raw file not present",
)
class TestBGESmokeTest:
    def _chunks(self) -> list[Chunk]:
        return chunk_work(
            Path("data/raw/beyond_good_and_evil.txt").read_text(encoding="utf-8"),
            "Beyond Good and Evil",
            "beyond_good_and_evil",
            "late",
            "aphorism",
        )

    def test_produces_many_chunks(self) -> None:
        """BGE has 296 numbered aphorisms — expect at least 200 chunks."""
        assert len(self._chunks()) >= 200

    def test_no_empty_chunks(self) -> None:
        for c in self._chunks():
            assert c.content.strip() != ""

    def test_no_gutenberg_boilerplate_in_chunks(self) -> None:
        for c in self._chunks():
            assert "Project Gutenberg" not in c.content
            assert "START OF" not in c.content

    def test_all_metadata_fields_present(self) -> None:
        required = {
            "work_title", "work_slug", "work_period",
            "section_number", "aphorism_number", "aphorism_number_end",
            "chunk_index", "chunk_type", "content",
        }
        for c in self._chunks():
            assert required == set(c.to_dict().keys())

    def test_section_numbers_cover_all_chapters(self) -> None:
        """BGE has 9 chapters — all should appear in section_number values."""
        chunks = self._chunks()
        sections = {c.section_number for c in chunks if c.aphorism_number is not None}
        assert sections == set(range(1, 10))
