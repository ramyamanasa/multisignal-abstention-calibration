"""Tests for src/alignment/modes.py — pedagogical mode prompts and rewriting.

Run with:
    pytest tests/test_alignment.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.alignment.modes import apply_mode, get_system_prompt


# ---------------------------------------------------------------------------
# get_system_prompt
# ---------------------------------------------------------------------------

class TestGetSystemPrompt:
    """Unit tests for get_system_prompt — no I/O, pure logic."""

    @pytest.mark.parametrize("mode", ["beginner", "exam", "deep"])
    def test_returns_non_empty_string_for_each_valid_mode(self, mode: str):
        """get_system_prompt must return a non-empty string for every supported mode."""
        result = get_system_prompt(mode)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_raises_value_error_for_unknown_mode(self):
        """get_system_prompt must raise ValueError for an unrecognised mode string."""
        with pytest.raises(ValueError, match="Unknown mode"):
            get_system_prompt("advanced")

    def test_raises_value_error_for_empty_string(self):
        """get_system_prompt must raise ValueError for an empty string."""
        with pytest.raises(ValueError):
            get_system_prompt("")

    def test_each_mode_returns_distinct_prompt(self):
        """Each mode must produce a different system prompt."""
        prompts = {mode: get_system_prompt(mode) for mode in ("beginner", "exam", "deep")}
        assert len(set(prompts.values())) == 3

    @pytest.mark.parametrize("mode", ["beginner", "exam", "deep"])
    def test_prompt_is_deterministic(self, mode: str):
        """Calling get_system_prompt twice for the same mode must return identical strings."""
        assert get_system_prompt(mode) == get_system_prompt(mode)


# ---------------------------------------------------------------------------
# apply_mode
# ---------------------------------------------------------------------------

class TestApplyMode:
    """Unit tests for apply_mode — all Ollama HTTP calls are mocked."""

    def _mock_response(self, text: str) -> MagicMock:
        """Build a fake requests.Response that mimics the Ollama JSON structure."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": text}
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    @patch("src.alignment.modes.requests.post")
    def test_returns_rewritten_answer(self, mock_post: MagicMock):
        """apply_mode must return the 'response' field from the Ollama reply."""
        expected = "Rewritten answer."
        mock_post.return_value = self._mock_response(f"  {expected}  ")

        result = apply_mode(
            answer="Raw draft.",
            query="What is backprop?",
            mode="beginner",
        )

        assert result == expected

    @patch("src.alignment.modes.requests.post")
    def test_calls_ollama_exactly_once(self, mock_post: MagicMock):
        """apply_mode must issue exactly one POST to the Ollama API."""
        mock_post.return_value = self._mock_response("ok")

        apply_mode(answer="draft", query="question", mode="exam")

        mock_post.assert_called_once()

    @patch("src.alignment.modes.requests.post")
    def test_uses_ollama_base_url_env_var(self, mock_post: MagicMock):
        """apply_mode must use OLLAMA_BASE_URL from the environment when set."""
        mock_post.return_value = self._mock_response("ok")

        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://remote:11434"}):
            apply_mode(answer="draft", query="question", mode="deep")

        call_url = mock_post.call_args[0][0]
        assert call_url.startswith("http://remote:11434")

    @patch("src.alignment.modes.requests.post")
    def test_explicit_ollama_url_overrides_env(self, mock_post: MagicMock):
        """Explicit ollama_url argument must take precedence over the env var."""
        mock_post.return_value = self._mock_response("ok")

        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://env-host:11434"}):
            apply_mode(
                answer="draft",
                query="question",
                mode="beginner",
                ollama_url="http://explicit:11434",
            )

        call_url = mock_post.call_args[0][0]
        assert "explicit" in call_url
        assert "env-host" not in call_url

    @patch("src.alignment.modes.requests.post")
    def test_raises_value_error_for_invalid_mode(self, mock_post: MagicMock):
        """apply_mode must raise ValueError before calling Ollama for an invalid mode."""
        with pytest.raises(ValueError, match="Unknown mode"):
            apply_mode(answer="draft", query="question", mode="invalid")

        mock_post.assert_not_called()

    @patch("src.alignment.modes.requests.post")
    def test_system_prompt_included_in_request_body(self, mock_post: MagicMock):
        """The Ollama prompt payload must contain part of the mode system prompt."""
        mock_post.return_value = self._mock_response("ok")

        apply_mode(answer="draft", query="question", mode="exam")

        payload = mock_post.call_args[1]["json"]
        assert "prompt" in payload
        # "exam" system prompt mentions bullet points
        assert "bullet" in payload["prompt"].lower()
