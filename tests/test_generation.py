"""Tests for generation/claude.py — no live API calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from generation.claude import SYSTEM_PROMPT, build_prompt, generate_answer
from retrieval.hybrid import HybridResult


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _result(
    work_title: str = "Beyond Good and Evil",
    section_number: int = 1,
    aphorism_number: int | None = 36,
    aphorism_number_end: int | None = None,
    document: str = "The will to power is the fundamental drive.",
    work_slug: str = "beyond_good_and_evil",
    chunk_index: int = 0,
) -> HybridResult:
    if aphorism_number_end is None:
        aphorism_number_end = aphorism_number
    return HybridResult(
        id=f"{work_slug}_chunk_{chunk_index}",
        document=document,
        metadata={
            "work_title": work_title,
            "work_slug": work_slug,
            "work_period": "late",
            "section_number": section_number,
            "aphorism_number": aphorism_number if aphorism_number is not None else -1,
            "aphorism_number_end": aphorism_number_end if aphorism_number_end is not None else -1,
            "chunk_index": chunk_index,
            "chunk_type": "aphorism",
        },
        rrf_score=0.1,
        rerank_score=1.0,
    )


def _three_results() -> list[HybridResult]:
    return [
        _result(
            document="Supposing that Truth is a woman--what then?",
            aphorism_number=1, chunk_index=0,
        ),
        _result(
            document="The will to truth which will still tempt us.",
            aphorism_number=2, chunk_index=1,
        ),
        _result(
            document="In the philosopher there is nothing whatever impersonal.",
            aphorism_number=6, chunk_index=2,
        ),
    ]


# ── build_prompt ──────────────────────────────────────────────────────────────


class TestBuildPrompt:
    def test_prompt_contains_all_passages(self) -> None:
        """Given 3 results, all 3 documents appear in the built prompt."""
        results = _three_results()
        prompt = build_prompt("What is truth?", results)
        for r in results:
            assert r.document in prompt

    def test_prompt_contains_source_citations(self) -> None:
        """Each passage block includes work_title and section/aphorism number."""
        results = _three_results()
        prompt = build_prompt("What is truth?", results)
        for r in results:
            work_title = r.metadata["work_title"]
            aph = r.metadata["aphorism_number"]
            assert work_title in prompt
            assert str(aph) in prompt

    def test_prompt_contains_question(self) -> None:
        """The user question appears verbatim at the end of the prompt."""
        question = "What does Nietzsche mean by the will to power?"
        prompt = build_prompt(question, _three_results())
        assert question in prompt
        # Question should be near the end
        assert prompt.rstrip().endswith(question)

    def test_no_api_call_in_prompt_builder(self) -> None:
        """build_prompt() is a pure function — it must not call the Anthropic API."""
        with patch("generation.claude.anthropic.Anthropic") as mock_cls:
            build_prompt("Any question?", _three_results())
        mock_cls.assert_not_called()

    def test_returns_string(self) -> None:
        assert isinstance(build_prompt("q", _three_results()), str)

    def test_source_label_uses_aphorism_number(self) -> None:
        r = _result(work_title="BGE", aphorism_number=36, section_number=2)
        prompt = build_prompt("q", [r])
        assert "§36" in prompt

    def test_source_label_range_for_merged_aphorisms(self) -> None:
        r = _result(aphorism_number=3, aphorism_number_end=5)
        prompt = build_prompt("q", [r])
        assert "§3" in prompt
        assert "5" in prompt

    def test_source_label_falls_back_to_chapter_for_prose(self) -> None:
        """Prose chunks (aphorism_number == -1) cite by chapter number."""
        r = _result(aphorism_number=None, section_number=3)
        # -1 is stored in metadata by _result helper
        prompt = build_prompt("q", [r])
        assert "ch.3" in prompt

    def test_empty_results_produces_prompt_with_question(self) -> None:
        question = "What is the eternal recurrence?"
        prompt = build_prompt(question, [])
        assert question in prompt

    def test_prompt_structure_has_separator_lines(self) -> None:
        prompt = build_prompt("q", [_result()])
        assert "---" in prompt

    def test_prompt_starts_with_source_passages_header(self) -> None:
        prompt = build_prompt("q", [_result()])
        assert prompt.startswith("Source passages:")

    def test_multiple_results_all_cited(self) -> None:
        results = [
            _result(work_title="BGE", aphorism_number=1, chunk_index=0),
            _result(work_title="BGE", aphorism_number=100, chunk_index=1),
            _result(work_title="BGE", aphorism_number=260, chunk_index=2),
        ]
        prompt = build_prompt("morality", results)
        assert "§1" in prompt
        assert "§100" in prompt
        assert "§260" in prompt

    def test_system_prompt_not_in_user_prompt(self) -> None:
        """build_prompt() returns the user turn only, not the system prompt."""
        prompt = build_prompt("q", [_result()])
        assert "philosophical research assistant" not in prompt


# ── generate_answer ───────────────────────────────────────────────────────────


class TestGenerateAnswer:
    def _mock_client(self, response_text: str = "Mocked answer.") -> MagicMock:
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text=response_text)]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_msg
        return mock_client

    def test_generation_returns_string(self) -> None:
        mock_client = self._mock_client("Mocked answer.")
        with patch("generation.claude.anthropic.Anthropic", return_value=mock_client):
            result = generate_answer("What is the will to power?", _three_results())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generation_returns_model_text(self) -> None:
        expected = "Nietzsche defines will to power as the fundamental drive of life."
        mock_client = self._mock_client(expected)
        with patch("generation.claude.anthropic.Anthropic", return_value=mock_client):
            result = generate_answer("What is the will to power?", _three_results())
        assert result == expected

    def test_api_called_once(self) -> None:
        mock_client = self._mock_client()
        with patch("generation.claude.anthropic.Anthropic", return_value=mock_client):
            generate_answer("q", _three_results())
        mock_client.messages.create.assert_called_once()

    def test_system_prompt_passed_to_api(self) -> None:
        mock_client = self._mock_client()
        with patch("generation.claude.anthropic.Anthropic", return_value=mock_client):
            generate_answer("q", _three_results())
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == SYSTEM_PROMPT

    def test_user_prompt_contains_question(self) -> None:
        question = "What is eternal recurrence?"
        mock_client = self._mock_client()
        with patch("generation.claude.anthropic.Anthropic", return_value=mock_client):
            generate_answer(question, _three_results())
        messages = mock_client.messages.create.call_args.kwargs["messages"]
        user_content = messages[0]["content"]
        assert question in user_content

    def test_user_prompt_contains_passages(self) -> None:
        results = _three_results()
        mock_client = self._mock_client()
        with patch("generation.claude.anthropic.Anthropic", return_value=mock_client):
            generate_answer("q", results)
        messages = mock_client.messages.create.call_args.kwargs["messages"]
        user_content = messages[0]["content"]
        for r in results:
            assert r.document in user_content

    def test_model_name_forwarded(self) -> None:
        mock_client = self._mock_client()
        with patch("generation.claude.anthropic.Anthropic", return_value=mock_client):
            generate_answer("q", _three_results(), model="claude-custom-model")
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-custom-model"

    def test_max_tokens_forwarded(self) -> None:
        mock_client = self._mock_client()
        with patch("generation.claude.anthropic.Anthropic", return_value=mock_client):
            generate_answer("q", _three_results(), max_tokens=512)
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 512

    def test_default_max_tokens_is_1024(self) -> None:
        mock_client = self._mock_client()
        with patch("generation.claude.anthropic.Anthropic", return_value=mock_client):
            generate_answer("q", _three_results())
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 1024

    def test_messages_role_is_user(self) -> None:
        mock_client = self._mock_client()
        with patch("generation.claude.anthropic.Anthropic", return_value=mock_client):
            generate_answer("q", _three_results())
        messages = mock_client.messages.create.call_args.kwargs["messages"]
        assert messages[0]["role"] == "user"


# ── SYSTEM_PROMPT content ─────────────────────────────────────────────────────


class TestSystemPrompt:
    def test_system_prompt_is_string(self) -> None:
        assert isinstance(SYSTEM_PROMPT, str)

    def test_system_prompt_mentions_nietzsche(self) -> None:
        assert "Nietzsche" in SYSTEM_PROMPT

    def test_system_prompt_requires_citations(self) -> None:
        assert "cite" in SYSTEM_PROMPT.lower() or "§" in SYSTEM_PROMPT

    def test_system_prompt_forbids_editorializing(self) -> None:
        assert "editorialize" in SYSTEM_PROMPT.lower() or "only" in SYSTEM_PROMPT.lower()
