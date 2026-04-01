"""Tests for ingest/embed.py — all model and vector store calls are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ingest.chunk import Chunk
from ingest.embed import (
    WORK_END_BEFORE,
    WORK_REGISTRY,
    _chunk_id,
    _chunk_metadata,
    embed_chunks,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_chunk(
    slug: str = "beyond_good_and_evil",
    idx: int = 0,
    aph: int | None = 1,
) -> Chunk:
    return Chunk(
        content="Supposing that Truth is a woman--what then?",
        work_title="Beyond Good and Evil",
        work_slug=slug,
        work_period="late",
        section_number=1,
        aphorism_number=aph,
        aphorism_number_end=aph,
        chunk_index=idx,
        chunk_type="aphorism",
    )


def _prose_chunk(idx: int = 0) -> Chunk:
    return Chunk(
        content="The birth of tragedy out of the spirit of music.",
        work_title="Birth of Tragedy",
        work_slug="birth_of_tragedy",
        work_period="early",
        section_number=1,
        aphorism_number=None,
        aphorism_number_end=None,
        chunk_index=idx,
        chunk_type="paragraph",
    )


def _mock_model(embedding_dim: int = 768) -> MagicMock:
    """Fake SentenceTransformer: encode() returns a numpy-like with .tolist()."""
    model = MagicMock()
    model.encode = MagicMock(
        side_effect=lambda texts, **kw: MagicMock(
            tolist=lambda: [[0.1] * embedding_dim for _ in texts]
        )
    )
    return model


# ── _chunk_metadata ───────────────────────────────────────────────────────────


class TestChunkMetadata:
    def test_all_fields_present(self) -> None:
        meta = _chunk_metadata(_make_chunk())
        expected = {
            "work_title", "work_slug", "work_period",
            "section_number", "aphorism_number", "aphorism_number_end",
            "chunk_index", "chunk_type",
        }
        assert set(meta.keys()) == expected

    def test_none_aphorism_serialised_as_minus_one(self) -> None:
        """ChromaDB rejects None; prose chunks must store -1 for integer fields."""
        meta = _chunk_metadata(_prose_chunk())
        assert meta["aphorism_number"] == -1
        assert meta["aphorism_number_end"] == -1

    def test_real_aphorism_number_preserved(self) -> None:
        meta = _chunk_metadata(_make_chunk(aph=42))
        assert meta["aphorism_number"] == 42
        assert meta["aphorism_number_end"] == 42

    def test_all_values_are_chromadb_safe_types(self) -> None:
        """Every value must be str, int, float, or bool — never None."""
        for chunk in [_make_chunk(), _prose_chunk()]:
            for v in _chunk_metadata(chunk).values():
                assert isinstance(v, (str, int, float, bool))


# ── _chunk_id ─────────────────────────────────────────────────────────────────


class TestChunkId:
    def test_format(self) -> None:
        assert _chunk_id(_make_chunk(slug="bge", idx=7)) == "bge_chunk_7"

    def test_different_indices_give_different_ids(self) -> None:
        assert _chunk_id(_make_chunk(idx=0)) != _chunk_id(_make_chunk(idx=1))

    def test_different_slugs_give_different_ids(self) -> None:
        assert _chunk_id(_make_chunk(slug="a", idx=0)) != _chunk_id(_make_chunk(slug="b", idx=0))

    def test_id_is_stable(self) -> None:
        chunk = _make_chunk(slug="bge", idx=3)
        assert _chunk_id(chunk) == _chunk_id(chunk)


# ── embed_chunks ──────────────────────────────────────────────────────────────


class TestEmbedChunks:
    # ── as per CLAUDE.md spec ────────────────────────────────────────────────

    def test_embed_calls_model(self) -> None:
        """SentenceTransformer.encode is called with the chunk texts."""
        mock_store = MagicMock()
        mock_model = _mock_model()

        with patch("ingest.embed.get_vector_store", return_value=mock_store), \
             patch("ingest.embed.SentenceTransformer", return_value=mock_model):
            embed_chunks([_make_chunk()])

        mock_model.encode.assert_called_once()
        texts_arg = mock_model.encode.call_args[0][0]
        assert texts_arg == ["Supposing that Truth is a woman--what then?"]

    def test_chunks_added_to_store(self) -> None:
        """All chunks are stored via store_chunks."""
        mock_store = MagicMock()
        mock_model = _mock_model()

        with patch("ingest.embed.get_vector_store", return_value=mock_store), \
             patch("ingest.embed.SentenceTransformer", return_value=mock_model):
            embed_chunks([_make_chunk()])

        mock_store.store_chunks.assert_called_once()

    # ── additional coverage ──────────────────────────────────────────────────

    def test_empty_list_returns_zero(self) -> None:
        with patch("ingest.embed.get_vector_store") as mock_store_fn, \
             patch("ingest.embed.SentenceTransformer"):
            result = embed_chunks([])
        assert result == 0
        mock_store_fn.assert_not_called()  # no store opened for empty input

    def test_returns_count_of_chunks(self) -> None:
        chunks = [_make_chunk(idx=i) for i in range(5)]
        mock_store = MagicMock()
        mock_model = _mock_model()

        with patch("ingest.embed.get_vector_store", return_value=mock_store), \
             patch("ingest.embed.SentenceTransformer", return_value=mock_model):
            result = embed_chunks(chunks)

        assert result == 5

    def test_upsert_ids_match_chunk_ids(self) -> None:
        chunks = [_make_chunk(idx=0), _make_chunk(idx=1)]
        mock_store = MagicMock()
        mock_model = _mock_model()

        with patch("ingest.embed.get_vector_store", return_value=mock_store), \
             patch("ingest.embed.SentenceTransformer", return_value=mock_model):
            embed_chunks(chunks)

        chunk_dicts = mock_store.store_chunks.call_args[0][0]
        ids_passed = [d["id"] for d in chunk_dicts]
        assert ids_passed == [_chunk_id(c) for c in chunks]

    def test_upsert_documents_match_content(self) -> None:
        chunks = [_make_chunk(idx=0), _make_chunk(idx=1)]
        mock_store = MagicMock()
        mock_model = _mock_model()

        with patch("ingest.embed.get_vector_store", return_value=mock_store), \
             patch("ingest.embed.SentenceTransformer", return_value=mock_model):
            embed_chunks(chunks)

        chunk_dicts = mock_store.store_chunks.call_args[0][0]
        docs = [d["content"] for d in chunk_dicts]
        assert docs == [c.content for c in chunks]

    def test_upsert_metadatas_have_no_none_values(self) -> None:
        chunks = [_make_chunk(), _prose_chunk()]
        mock_store = MagicMock()
        mock_model = _mock_model()

        with patch("ingest.embed.get_vector_store", return_value=mock_store), \
             patch("ingest.embed.SentenceTransformer", return_value=mock_model):
            embed_chunks(chunks)

        chunk_dicts = mock_store.store_chunks.call_args[0][0]
        metadatas = [{k: v for k, v in d.items() if k not in ("id", "content")} for d in chunk_dicts]
        for meta in metadatas:
            for v in meta.values():
                assert v is not None

    def test_batch_size_controls_upsert_calls(self) -> None:
        """7 chunks with batch_size=3 → 3 store_chunks calls (3 + 3 + 1)."""
        chunks = [_make_chunk(idx=i) for i in range(7)]
        mock_store = MagicMock()
        mock_model = _mock_model()

        with patch("ingest.embed.get_vector_store", return_value=mock_store), \
             patch("ingest.embed.SentenceTransformer", return_value=mock_model):
            embed_chunks(chunks, batch_size=3)

        assert mock_store.store_chunks.call_count == 3

    def test_encode_called_once_per_batch(self) -> None:
        chunks = [_make_chunk(idx=i) for i in range(4)]
        mock_store = MagicMock()
        mock_model = _mock_model()

        with patch("ingest.embed.get_vector_store", return_value=mock_store), \
             patch("ingest.embed.SentenceTransformer", return_value=mock_model):
            embed_chunks(chunks, batch_size=2)

        assert mock_model.encode.call_count == 2

    def test_embeddings_passed_to_store(self) -> None:
        """The embedding vectors returned by encode are forwarded to store_chunks."""
        chunk = _make_chunk()
        mock_store = MagicMock()
        mock_model = _mock_model(embedding_dim=768)

        with patch("ingest.embed.get_vector_store", return_value=mock_store), \
             patch("ingest.embed.SentenceTransformer", return_value=mock_model):
            embed_chunks([chunk])

        embeddings = mock_store.store_chunks.call_args[0][1]
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768

    def test_model_name_passed_to_sentence_transformer(self) -> None:
        mock_store = MagicMock()

        with patch("ingest.embed.get_vector_store", return_value=mock_store), \
             patch("ingest.embed.SentenceTransformer", return_value=_mock_model()) as mock_cls:
            embed_chunks([_make_chunk()], model_name="custom/model")

        mock_cls.assert_called_once_with("custom/model")


# ── WORK_REGISTRY coverage ────────────────────────────────────────────────────

EXPECTED_REGISTRY_SLUGS = {
    # Late period
    "beyond_good_and_evil",
    "genealogy_of_morality",
    "twilight_of_the_idols",
    "the_antichrist",
    "ecce_homo",
    "nietzsche_contra_wagner",
    # Middle period
    "the_gay_science",
    "daybreak",
    "human_all_too_human",
    # Early period
    "birth_of_tragedy",
    "untimely_meditations_1",
    "untimely_meditations_2",
}

VALID_PERIODS = {"early", "middle", "late"}
VALID_STYLES = {"aphorism", "paragraph"}


class TestWorkRegistry:
    def test_all_expected_slugs_present(self) -> None:
        missing = EXPECTED_REGISTRY_SLUGS - set(WORK_REGISTRY)
        assert not missing, f"Missing slugs in WORK_REGISTRY: {missing}"

    def test_each_entry_has_three_fields(self) -> None:
        for slug, entry in WORK_REGISTRY.items():
            assert len(entry) == 3, f"{slug}: expected (title, period, style), got {entry!r}"

    def test_titles_are_non_empty_strings(self) -> None:
        for slug, (title, _, _) in WORK_REGISTRY.items():
            assert isinstance(title, str) and title, f"{slug}: empty or non-string title"

    def test_periods_are_valid(self) -> None:
        for slug, (_, period, _) in WORK_REGISTRY.items():
            assert period in VALID_PERIODS, f"{slug}: invalid period {period!r}"

    def test_chunk_styles_are_valid(self) -> None:
        for slug, (_, _, style) in WORK_REGISTRY.items():
            assert style in VALID_STYLES, f"{slug}: invalid chunk style {style!r}"

    def test_late_period_works_registered(self) -> None:
        late = {s for s, (_, p, _) in WORK_REGISTRY.items() if p == "late"}
        assert late >= {
            "beyond_good_and_evil", "genealogy_of_morality", "twilight_of_the_idols",
            "the_antichrist", "ecce_homo", "nietzsche_contra_wagner",
        }

    def test_middle_period_works_registered(self) -> None:
        middle = {s for s, (_, p, _) in WORK_REGISTRY.items() if p == "middle"}
        assert middle >= {"the_gay_science", "daybreak", "human_all_too_human"}

    def test_early_period_works_registered(self) -> None:
        early = {s for s, (_, p, _) in WORK_REGISTRY.items() if p == "early"}
        assert early >= {"birth_of_tragedy", "untimely_meditations_1", "untimely_meditations_2"}

    def test_twilight_uses_paragraph_style(self) -> None:
        """TI has non-unique local aphorism numbers — must use paragraph chunking."""
        _, _, style = WORK_REGISTRY["twilight_of_the_idols"]
        assert style == "paragraph"

    def test_prose_works_use_paragraph_style(self) -> None:
        prose_slugs = {
            "genealogy_of_morality", "twilight_of_the_idols", "ecce_homo",
            "birth_of_tragedy", "untimely_meditations_1", "untimely_meditations_2",
        }
        for slug in prose_slugs:
            _, _, style = WORK_REGISTRY[slug]
            assert style == "paragraph", f"{slug}: expected paragraph, got {style!r}"


# ── WORK_END_BEFORE ───────────────────────────────────────────────────────────


class TestWorkEndBefore:
    def test_twilight_has_end_before_entry(self) -> None:
        """PG 52263 bundles TI + The Antichrist — truncation marker must be set."""
        assert "twilight_of_the_idols" in WORK_END_BEFORE

    def test_twilight_marker_targets_antichrist_body(self) -> None:
        marker = WORK_END_BEFORE["twilight_of_the_idols"]
        assert "ANTICHRIST" in marker.upper()

    def test_all_end_before_slugs_are_in_registry(self) -> None:
        for slug in WORK_END_BEFORE:
            assert slug in WORK_REGISTRY, f"WORK_END_BEFORE slug {slug!r} not in WORK_REGISTRY"

    def test_all_end_before_values_are_non_empty_strings(self) -> None:
        for slug, marker in WORK_END_BEFORE.items():
            assert isinstance(marker, str) and marker, f"{slug}: empty or non-string marker"
