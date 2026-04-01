"""Tests for the FastAPI backend — all pipeline and model calls are mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_SOURCE = {
    "work_title": "Beyond Good and Evil",
    "work_slug": "beyond_good_and_evil",
    "section_number": 36,
    "chunk_type": "aphorism",
    "content": "Supposing truth is a woman—what then?",
    "similarity": 0.87,
}

SAMPLE_PIPELINE_RESULT = {
    "answer": "Nietzsche argues that truth is perspectival [BGE §36].",
    "sources": [SAMPLE_SOURCE],
}


@pytest.fixture()
def client(mocker):
    """TestClient with heavy startup dependencies mocked out.

    Sets ``config.INGEST_TOKEN`` to ``"test-token"`` so auth tests work without
    a real ``.env`` file.
    """
    # Mock model loading in the lifespan
    mocker.patch("api.main.SentenceTransformer", return_value=MagicMock())
    mocker.patch("api.main.CrossEncoder", return_value=MagicMock())

    # Mock vector store so BM25 index builds from empty corpus
    mock_store = MagicMock()
    mock_store.get_all_documents.return_value = {
        "ids": [],
        "documents": [],
        "metadatas": [],
    }
    mocker.patch("api.main.get_vector_store", return_value=mock_store)

    # Ensure a known ingest token is configured for auth tests
    import config as _config
    mocker.patch.object(_config, "INGEST_TOKEN", "test-token")

    from api.main import app

    with TestClient(app) as c:
        yield c


# ── POST /query ───────────────────────────────────────────────────────────────


class TestQueryEndpoint:
    def test_returns_200_with_answer(self, client, mocker) -> None:
        mocker.patch(
            "api.routes.query.run_pipeline",
            return_value=SAMPLE_PIPELINE_RESULT,
        )
        res = client.post("/query", json={"question": "What is the will to power?"})
        assert res.status_code == 200
        assert "answer" in res.json()

    def test_response_contains_sources(self, client, mocker) -> None:
        mocker.patch(
            "api.routes.query.run_pipeline",
            return_value=SAMPLE_PIPELINE_RESULT,
        )
        res = client.post("/query", json={"question": "What is eternal recurrence?"})
        body = res.json()
        assert isinstance(body["sources"], list)
        assert len(body["sources"]) == 1
        assert body["sources"][0]["work_slug"] == "beyond_good_and_evil"

    def test_missing_question_returns_422(self, client) -> None:
        res = client.post("/query", json={})
        assert res.status_code == 422

    def test_filter_period_forwarded(self, client, mocker) -> None:
        mock_pipeline = mocker.patch(
            "api.routes.query.run_pipeline",
            return_value=SAMPLE_PIPELINE_RESULT,
        )
        client.post(
            "/query",
            json={"question": "Eternal recurrence?", "filter_period": "late"},
        )
        _, kwargs = mock_pipeline.call_args
        # run_pipeline is called with positional args
        args = mock_pipeline.call_args[0]
        assert args[1] == "late"  # filter_period

    def test_filter_slug_forwarded(self, client, mocker) -> None:
        mock_pipeline = mocker.patch(
            "api.routes.query.run_pipeline",
            return_value=SAMPLE_PIPELINE_RESULT,
        )
        client.post(
            "/query",
            json={"question": "Socrates?", "filter_slug": "twilight_of_the_idols"},
        )
        args = mock_pipeline.call_args[0]
        assert args[2] == "twilight_of_the_idols"  # filter_slug

    def test_invalid_period_returns_422(self, client) -> None:
        res = client.post(
            "/query",
            json={"question": "Test?", "filter_period": "ancient"},
        )
        assert res.status_code == 422

    def test_answer_is_string(self, client, mocker) -> None:
        mocker.patch(
            "api.routes.query.run_pipeline",
            return_value=SAMPLE_PIPELINE_RESULT,
        )
        res = client.post("/query", json={"question": "What is nihilism?"})
        assert isinstance(res.json()["answer"], str)


# ── POST /ingest ──────────────────────────────────────────────────────────────


class TestIngestEndpoint:
    def test_missing_token_returns_401(self, client) -> None:
        res = client.post("/ingest")
        assert res.status_code == 401

    def test_wrong_token_returns_401(self, client) -> None:
        res = client.post("/ingest", headers={"X-Ingest-Token": "wrong"})
        assert res.status_code == 401

    def test_valid_token_returns_200(self, client, mocker) -> None:
        mocker.patch("api.routes.ingest.run_ingest", return_value=42)
        res = client.post("/ingest", headers={"X-Ingest-Token": "test-token"})
        assert res.status_code == 200

    def test_valid_token_returns_status_started(self, client, mocker) -> None:
        mocker.patch("api.routes.ingest.run_ingest", return_value=42)
        res = client.post("/ingest", headers={"X-Ingest-Token": "test-token"})
        assert res.json()["status"] == "started"
