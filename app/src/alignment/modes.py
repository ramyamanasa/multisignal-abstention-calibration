"""Pedagogical mode system prompts and answer rewriting for lectureOS."""

from __future__ import annotations

import os

import requests

_OLLAMA_DEFAULT = "http://localhost:11434"
_MODEL = "qwen2.5:7b-instruct"

_SYSTEM_PROMPTS: dict[str, str] = {
    "beginner": (
        "You are a patient tutor explaining concepts to a student with no prior background. "
        "Use short sentences. Avoid jargon — define every technical term when first used. "
        "Ground every idea in a concrete, everyday analogy before stating the formal concept."
    ),
    "exam": (
        "You are a concise exam-prep assistant. "
        "Structure your answer as bullet points. "
        "**Bold** every key term on first use. "
        "Provide tight, one-sentence definitions — no tangents."
    ),
    "deep": (
        "You are an expert instructor writing for an advanced audience. "
        "Provide full technical depth: include derivations, proofs, or pseudocode where relevant. "
        "Actively cross-reference related concepts and note important edge cases or open questions."
    ),
}


def get_system_prompt(mode: str) -> str:
    """Return the system prompt for the given pedagogical mode.

    Args:
        mode: One of ``"beginner"``, ``"exam"``, or ``"deep"``.

    Returns:
        A non-empty system prompt string tailored to the mode.

    Raises:
        ValueError: If ``mode`` is not one of the supported values.
    """
    if mode not in _SYSTEM_PROMPTS:
        supported = ", ".join(f'"{m}"' for m in _SYSTEM_PROMPTS)
        raise ValueError(f"Unknown mode {mode!r}. Supported modes: {supported}.")
    return _SYSTEM_PROMPTS[mode]


def apply_mode(
    answer: str,
    query: str,
    mode: str,
    ollama_url: str | None = None,
) -> str:
    """Rewrite a raw LLM answer using the system prompt for the given mode.

    Calls the local Ollama instance with the mode-specific system prompt so
    the model reshapes the answer's tone, structure, and vocabulary.

    Args:
        answer: The raw answer text to rewrite.
        query: The original student question, used as context for rewriting.
        mode: One of ``"beginner"``, ``"exam"``, or ``"deep"``.
        ollama_url: Base URL for the Ollama API. Defaults to the
            ``OLLAMA_BASE_URL`` environment variable, falling back to
            ``http://localhost:11434``.

    Returns:
        The rewritten answer as a plain string.

    Raises:
        ValueError: If ``mode`` is not recognised.
        requests.HTTPError: If the Ollama API returns a non-2xx status.
    """
    system_prompt = get_system_prompt(mode)  # raises ValueError for bad mode
    base_url = ollama_url or os.getenv("OLLAMA_BASE_URL", _OLLAMA_DEFAULT)

    prompt = (
        f"{system_prompt}\n\n"
        f"Original question: {query}\n\n"
        f"Draft answer:\n{answer}\n\n"
        "Rewrite the draft answer according to the instructions above."
    )
    payload = {"model": _MODEL, "prompt": prompt, "stream": False}
    resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"].strip()
