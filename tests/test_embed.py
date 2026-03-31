"""Tests for ingest/embed.py — all model and ChromaDB calls are mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from ingest.chunk import Chunk
from ingest.embed import _chunk_id, _chunk_metadata, embed_chunks, get_chroma_collection


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
    import numpy as np

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


# ── get_chroma_collection ─────────────────────────────────────────────────────


class TestGetChromaCollection:
    def test_creates_collection(self, tmp_path: Path) -> None:
        """get_chroma_collection returns a usable collection object."""
        with patch("ingest.embed.chromadb.PersistentClient") as mock_client_cls:
            mock_col = MagicMock()
            mock_client_cls.return_value.get_or_create_collection.return_value = mock_col
            result = get_chroma_collection(tmp_path, "test")
        mock_client_cls.assert_called_once_with(path=str(tmp_path))
        mock_client_cls.return_value.get_or_create_collection.assert_called_once_with("test")
        assert result is mock_col

    def test_creates_persist_dir_if_missing(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "chroma"
        with patch("ingest.embed.chromadb.PersistentClient"):
            get_chroma_collection(deep, "test")
        assert deep.exists()


# ── embed_chunks ──────────────────────────────────────────────────────────────


class TestEmbedChunks:
    def _patches(self, tmp_path: Path):
        """Context manager that mocks both SentenceTransformer and get_chroma_collection."""
        mock_col = MagicMock()
        mock_model = _mock_model()

        col_patch = patch("ingest.embed.get_chroma_collection", return_value=mock_col)
        model_patch = patch("ingest.embed.SentenceTransformer", return_value=mock_model)
        return col_patch, model_patch, mock_col, mock_model

    # ── as per CLAUDE.md spec ────────────────────────────────────────────────

    def test_embed_calls_model(self, tmp_path: Path) -> None:
        """SentenceTransformer.encode is called with the chunk texts."""
        mock_col = MagicMock()
        mock_model = _mock_model()

        with patch("ingest.embed.get_chroma_collection", return_value=mock_col), \
             patch("ingest.embed.SentenceTransformer", return_value=mock_model):
            embed_chunks([_make_chunk()], persist_dir=tmp_path)

        mock_model.encode.assert_called_once()
        # The first positional argument is the list of texts
        texts_arg = mock_model.encode.call_args[0][0]
        assert texts_arg == ["Supposing that Truth is a woman--what then?"]

    def test_chunks_added_to_collection(self, tmp_path: Path) -> None:
        """All chunks are upserted into the ChromaDB collection."""
        mock_col = MagicMock()
        mock_model = _mock_model()

        with patch("ingest.embed.get_chroma_collection", return_value=mock_col), \
             patch("ingest.embed.SentenceTransformer", return_value=mock_model):
            embed_chunks([_make_chunk()], persist_dir=tmp_path)

        mock_col.upsert.assert_called_once()

    # ── additional coverage ──────────────────────────────────────────────────

    def test_empty_list_returns_zero(self, tmp_path: Path) -> None:
        with patch("ingest.embed.get_chroma_collection") as mock_col_fn, \
             patch("ingest.embed.SentenceTransformer"):
            result = embed_chunks([], persist_dir=tmp_path)
        assert result == 0
        mock_col_fn.assert_not_called()  # no collection opened for empty input

    def test_returns_count_of_chunks(self, tmp_path: Path) -> None:
        chunks = [_make_chunk(idx=i) for i in range(5)]
        mock_col = MagicMock()
        mock_model = _mock_model()

        with patch("ingest.embed.get_chroma_collection", return_value=mock_col), \
             patch("ingest.embed.SentenceTransformer", return_value=mock_model):
            result = embed_chunks(chunks, persist_dir=tmp_path)

        assert result == 5

    def test_upsert_ids_match_chunk_ids(self, tmp_path: Path) -> None:
        chunks = [_make_chunk(idx=0), _make_chunk(idx=1)]
        mock_col = MagicMock()
        mock_model = _mock_model()

        with patch("ingest.embed.get_chroma_collection", return_value=mock_col), \
             patch("ingest.embed.SentenceTransformer", return_value=mock_model):
            embed_chunks(chunks, persist_dir=tmp_path)

        ids_passed = mock_col.upsert.call_args.kwargs["ids"]
        assert ids_passed == [_chunk_id(c) for c in chunks]

    def test_upsert_documents_match_content(self, tmp_path: Path) -> None:
        chunks = [_make_chunk(idx=0), _make_chunk(idx=1)]
        mock_col = MagicMock()
        mock_model = _mock_model()

        with patch("ingest.embed.get_chroma_collection", return_value=mock_col), \
             patch("ingest.embed.SentenceTransformer", return_value=mock_model):
            embed_chunks(chunks, persist_dir=tmp_path)

        docs = mock_col.upsert.call_args.kwargs["documents"]
        assert docs == [c.content for c in chunks]

    def test_upsert_metadatas_have_no_none_values(self, tmp_path: Path) -> None:
        chunks = [_make_chunk(), _prose_chunk()]
        mock_col = MagicMock()
        mock_model = _mock_model()

        with patch("ingest.embed.get_chroma_collection", return_value=mock_col), \
             patch("ingest.embed.SentenceTransformer", return_value=mock_model):
            embed_chunks(chunks, persist_dir=tmp_path)

        metadatas = mock_col.upsert.call_args.kwargs["metadatas"]
        for meta in metadatas:
            for v in meta.values():
                assert v is not None

    def test_batch_size_controls_upsert_calls(self, tmp_path: Path) -> None:
        """7 chunks with batch_size=3 → 3 upsert calls (3 + 3 + 1)."""
        chunks = [_make_chunk(idx=i) for i in range(7)]
        mock_col = MagicMock()
        mock_model = _mock_model()

        with patch("ingest.embed.get_chroma_collection", return_value=mock_col), \
             patch("ingest.embed.SentenceTransformer", return_value=mock_model):
            embed_chunks(chunks, persist_dir=tmp_path, batch_size=3)

        assert mock_col.upsert.call_count == 3

    def test_encode_called_once_per_batch(self, tmp_path: Path) -> None:
        chunks = [_make_chunk(idx=i) for i in range(4)]
        mock_col = MagicMock()
        mock_model = _mock_model()

        with patch("ingest.embed.get_chroma_collection", return_value=mock_col), \
             patch("ingest.embed.SentenceTransformer", return_value=mock_model):
            embed_chunks(chunks, persist_dir=tmp_path, batch_size=2)

        assert mock_model.encode.call_count == 2

    def test_embeddings_passed_to_upsert(self, tmp_path: Path) -> None:
        """The embedding vectors returned by encode are forwarded to upsert."""
        chunk = _make_chunk()
        mock_col = MagicMock()
        mock_model = _mock_model(embedding_dim=768)

        with patch("ingest.embed.get_chroma_collection", return_value=mock_col), \
             patch("ingest.embed.SentenceTransformer", return_value=mock_model):
            embed_chunks([chunk], persist_dir=tmp_path)

        embeddings = mock_col.upsert.call_args.kwargs["embeddings"]
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768

    def test_model_name_passed_to_sentence_transformer(self, tmp_path: Path) -> None:
        mock_col = MagicMock()

        with patch("ingest.embed.get_chroma_collection", return_value=mock_col), \
             patch("ingest.embed.SentenceTransformer", return_value=_mock_model()) as mock_cls:
            embed_chunks([_make_chunk()], persist_dir=tmp_path, model_name="custom/model")

        mock_cls.assert_called_once_with("custom/model")

    def test_collection_name_forwarded(self, tmp_path: Path) -> None:
        with patch("ingest.embed.get_chroma_collection", return_value=MagicMock()) as mock_fn, \
             patch("ingest.embed.SentenceTransformer", return_value=_mock_model()):
            embed_chunks([_make_chunk()], persist_dir=tmp_path, collection_name="my_col")

        assert mock_fn.call_args.kwargs.get("collection_name") == "my_col" or \
               mock_fn.call_args.args[1] == "my_col"
