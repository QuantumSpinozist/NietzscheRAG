"""Tests for ingest/fetch.py — all network calls are mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from ingest.fetch import GUTENBERG_SOURCES, fetch_all, fetch_text, save_work


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(body: str, content_length: int | None = None) -> MagicMock:
    """Return a mock requests.Response that streams *body* in 8 KB chunks."""
    encoded = body.encode("utf-8")
    headers = {}
    if content_length is not None:
        headers["content-length"] = str(content_length)

    resp = MagicMock()
    resp.headers = headers
    resp.raise_for_status = MagicMock()
    resp.iter_content = MagicMock(
        return_value=iter(
            [encoded[i : i + 8192] for i in range(0, len(encoded), 8192)] or [b""]
        )
    )
    return resp


# ---------------------------------------------------------------------------
# fetch_text
# ---------------------------------------------------------------------------

class TestFetchText:
    def test_returns_decoded_content(self) -> None:
        """fetch_text decodes and returns the full response body."""
        body = "Jenseits von Gut und Böse"
        with patch("ingest.fetch.requests.get", return_value=_mock_response(body)):
            result = fetch_text("https://example.com/bge.txt", "bge")
        assert result == body

    def test_raises_on_http_error(self) -> None:
        """fetch_text propagates HTTPError from raise_for_status."""
        resp = _mock_response("")
        resp.raise_for_status.side_effect = requests.HTTPError("404")
        with patch("ingest.fetch.requests.get", return_value=resp):
            with pytest.raises(requests.HTTPError):
                fetch_text("https://example.com/missing.txt", "missing")

    def test_progress_reported_when_content_length_present(self) -> None:
        """fetch_text calls iter_content when content-length header is present."""
        body = "A" * 20_000
        resp = _mock_response(body, content_length=len(body))
        with patch("ingest.fetch.requests.get", return_value=resp):
            result = fetch_text("https://example.com/large.txt", "large")
        assert len(result) == 20_000

    def test_handles_missing_content_length(self) -> None:
        """fetch_text works correctly when content-length header is absent."""
        body = "short text"
        with patch("ingest.fetch.requests.get", return_value=_mock_response(body)):
            result = fetch_text("https://example.com/short.txt", "short")
        assert result == body


# ---------------------------------------------------------------------------
# save_work
# ---------------------------------------------------------------------------

class TestSaveWork:
    def test_saves_file_to_dest_dir(self, tmp_path: Path) -> None:
        """save_work writes the fetched text to <slug>.txt in dest_dir."""
        body = "Beyond Good and Evil content"
        slug = "beyond_good_and_evil"
        resp = _mock_response(body)

        with patch("ingest.fetch.requests.get", return_value=resp):
            out = save_work(slug, dest_dir=tmp_path)

        assert out == tmp_path / f"{slug}.txt"
        assert out.read_text(encoding="utf-8") == body

    def test_skips_download_if_file_exists(self, tmp_path: Path) -> None:
        """save_work does not issue a network request when the file already exists."""
        slug = "beyond_good_and_evil"
        existing = tmp_path / f"{slug}.txt"
        existing.write_text("cached content", encoding="utf-8")

        with patch("ingest.fetch.requests.get") as mock_get:
            out = save_work(slug, dest_dir=tmp_path)

        mock_get.assert_not_called()
        assert out == existing
        assert existing.read_text() == "cached content"  # unchanged

    def test_creates_dest_dir_if_missing(self, tmp_path: Path) -> None:
        """save_work creates dest_dir (and parents) when it does not exist."""
        slug = "beyond_good_and_evil"
        deep_dir = tmp_path / "a" / "b" / "raw"

        with patch("ingest.fetch.requests.get", return_value=_mock_response("text")):
            out = save_work(slug, dest_dir=deep_dir)

        assert out.exists()

    def test_raises_on_unknown_slug(self, tmp_path: Path) -> None:
        """save_work raises KeyError for slugs not in GUTENBERG_SOURCES."""
        with pytest.raises(KeyError, match="unknown_work"):
            save_work("unknown_work", dest_dir=tmp_path)

    def test_returned_path_matches_slug(self, tmp_path: Path) -> None:
        """save_work return value has the expected filename."""
        slug = "beyond_good_and_evil"
        with patch("ingest.fetch.requests.get", return_value=_mock_response("x")):
            out = save_work(slug, dest_dir=tmp_path)
        assert out.name == f"{slug}.txt"


# ---------------------------------------------------------------------------
# fetch_all
# ---------------------------------------------------------------------------

class TestFetchAll:
    def test_returns_one_path_per_source(self, tmp_path: Path) -> None:
        """fetch_all returns exactly as many paths as GUTENBERG_SOURCES entries."""
        with patch("ingest.fetch.requests.get", return_value=_mock_response("text")):
            paths = fetch_all(dest_dir=tmp_path)
        assert len(paths) == len(GUTENBERG_SOURCES)

    def test_all_files_written(self, tmp_path: Path) -> None:
        """fetch_all writes a .txt file for every defined slug."""
        with patch("ingest.fetch.requests.get", return_value=_mock_response("text")):
            paths = fetch_all(dest_dir=tmp_path)
        for p in paths:
            assert p.exists()
            assert p.suffix == ".txt"

    def test_slugs_match_sources(self, tmp_path: Path) -> None:
        """fetch_all creates files whose stems match GUTENBERG_SOURCES keys."""
        with patch("ingest.fetch.requests.get", return_value=_mock_response("text")):
            paths = fetch_all(dest_dir=tmp_path)
        stems = {p.stem for p in paths}
        assert stems == set(GUTENBERG_SOURCES.keys())


# ---------------------------------------------------------------------------
# GUTENBERG_SOURCES coverage — all expected works must be registered
# ---------------------------------------------------------------------------

EXPECTED_SLUGS = {
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


class TestGutenbergSourcesCoverage:
    def test_all_expected_slugs_present(self) -> None:
        """Every corpus slug listed in CLAUDE.md must appear in GUTENBERG_SOURCES."""
        missing = EXPECTED_SLUGS - set(GUTENBERG_SOURCES.keys())
        assert not missing, f"Missing slugs: {missing}"

    def test_each_entry_has_title_and_url(self) -> None:
        """Every entry is a (title, url) 2-tuple with non-empty strings."""
        for slug, (title, url) in GUTENBERG_SOURCES.items():
            assert isinstance(title, str) and title, f"{slug}: empty title"
            assert isinstance(url, str) and url.startswith("https://"), f"{slug}: bad url {url!r}"

    def test_urls_point_to_gutenberg(self) -> None:
        """All URLs are Project Gutenberg cache URLs."""
        for slug, (_, url) in GUTENBERG_SOURCES.items():
            assert "gutenberg.org" in url, f"{slug}: URL not on gutenberg.org"

    @pytest.mark.parametrize("slug", sorted(EXPECTED_SLUGS))
    def test_save_work_succeeds_for_slug(self, slug: str, tmp_path: Path) -> None:
        """save_work writes a file for every expected slug (network mocked)."""
        with patch("ingest.fetch.requests.get", return_value=_mock_response("text")):
            out = save_work(slug, dest_dir=tmp_path)
        assert out.exists()
        assert out.stem == slug
