"""Tests for src/agents/graph.py — CALAS LangGraph pipeline.

Run with:
    pytest tests/test_agents.py -v
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agents.graph import (
    CALASState,
    build_graph,
    calas_graph,
    classify_node,
    route_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(**overrides) -> dict:
    """Return a minimal valid pipeline state, with optional field overrides."""
    base: dict = {
        "query": "",
        "chunks": [],
        "answer": "",
        "confidence": 0.0,
        "action": "answer",
        "mode": "beginner",
        "signals": {},
        "hallucination_prob": 0.0,
        "corpus_confidence": 1.0,
        "has_slides": True,
        "abstain_reason": "",
    }
    return {**base, **overrides}


def _mock_pipeline_result(hal_prob: float, decision: str) -> dict:
    return {
        "hallucination_probability": hal_prob,
        "decision": decision,
        "entropy_signals":      {"mean_entropy": 0.1, "max_entropy": 0.2, "entity_entropy": 0.1},
        "consistency_signals":  {"semantic_inconsistency": 0.3},
        "disagreement_signals": {"cross_model_disagreement": 0.2},
    }


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

class TestGraphCompilation:
    """Verify the graph can be built and compiled without errors."""

    def test_build_graph_returns_compiled_graph(self):
        graph = build_graph()
        assert graph is not None

    def test_module_level_calas_graph_is_available(self):
        assert calas_graph is not None


# ---------------------------------------------------------------------------
# classify_node
# ---------------------------------------------------------------------------

class TestClassifyNode:
    """Unit tests for classify_node — all Groq calls are mocked."""

    @patch("src.agents.graph._ollama_chat", return_value="ambiguous")
    def test_vague_one_word_query_returns_clarify(self, mock_chat):
        """A one-word query like 'explain' must produce action='clarify'."""
        result = classify_node(_state(query="explain"))

        assert result["action"] == "clarify"
        mock_chat.assert_called_once()

    @patch("src.agents.graph._ollama_chat", return_value="clear")
    def test_well_formed_question_returns_answer(self, mock_chat):
        """A clear coursework question must produce action='answer'."""
        result = classify_node(_state(query="What is gradient descent?"))

        assert result["action"] == "answer"

    @patch("src.agents.graph._ollama_chat", return_value="clear")
    def test_classify_preserves_other_state_fields(self, mock_chat):
        """classify_node must not mutate unrelated state fields."""
        original = _state(query="Explain backpropagation.", mode="exam", confidence=0.9)
        result = classify_node(original)

        assert result["mode"] == "exam"
        assert result["confidence"] == 0.9
        assert result["query"] == "Explain backpropagation."


# ---------------------------------------------------------------------------
# route_node
# ---------------------------------------------------------------------------

class TestRouteNode:
    """Unit tests for route_node — run_pipeline is mocked."""

    @patch("src.abstention.pipeline.run_pipeline")
    def test_high_hallucination_probability_returns_abstain(self, mock_run):
        """route_node must set action='abstain' when hallucination_prob >= threshold."""
        mock_run.return_value = _mock_pipeline_result(hal_prob=0.9, decision="abstain")
        result = route_node(_state(query="What is backpropagation?"))

        assert result["action"] == "abstain"

    @patch("src.abstention.pipeline.run_pipeline")
    def test_low_hallucination_probability_returns_answer(self, mock_run):
        """route_node must set action='answer' when hallucination_prob < threshold."""
        mock_run.return_value = _mock_pipeline_result(hal_prob=0.2, decision="answer")
        result = route_node(_state(query="What is gradient descent?"))

        assert result["action"] == "answer"

    @patch("src.abstention.pipeline.run_pipeline")
    def test_confidence_is_inverse_of_hallucination_prob(self, mock_run):
        """confidence must equal 1 - hallucination_probability."""
        mock_run.return_value = _mock_pipeline_result(hal_prob=0.3, decision="answer")
        result = route_node(_state(query="What is attention?"))

        assert abs(result["confidence"] - 0.7) < 1e-4

    @patch("src.abstention.pipeline.run_pipeline")
    def test_signals_stored_in_state(self, mock_run):
        """route_node must merge all three signal dicts into state['signals']."""
        mock_run.return_value = _mock_pipeline_result(hal_prob=0.4, decision="answer")
        result = route_node(_state(query="test"))

        assert "mean_entropy" in result["signals"]
        assert "semantic_inconsistency" in result["signals"]
        assert "cross_model_disagreement" in result["signals"]

    @patch("src.abstention.pipeline.run_pipeline")
    def test_hallucination_prob_stored_in_state(self, mock_run):
        """route_node must write hallucination_prob into state."""
        mock_run.return_value = _mock_pipeline_result(hal_prob=0.75, decision="abstain")
        result = route_node(_state(query="test"))

        assert result["hallucination_prob"] == 0.75

    @patch("src.abstention.pipeline.run_pipeline")
    def test_route_preserves_query_and_mode(self, mock_run):
        """route_node must not alter query or mode."""
        mock_run.return_value = _mock_pipeline_result(hal_prob=0.1, decision="answer")
        original = _state(query="test query", mode="deep")
        result = route_node(original)

        assert result["query"] == "test query"
        assert result["mode"] == "deep"

    def test_low_corpus_confidence_hard_abstains_without_pipeline(self):
        """route_node must hard-abstain immediately when corpus_confidence < 0.5.

        No pipeline calls should be made — this is a cheap retrieval-level guard
        that fires before any LLM calls when uploaded slides clearly don't cover
        the topic.
        """
        off_topic_chunks = [
            {"text": "Backpropagation computes gradients.",
             "metadata": {"source": "lec1.pdf", "page": 1}, "score": 0.01},
        ]
        result = route_node(_state(
            query="Who won the 2024 Super Bowl?",
            chunks=off_topic_chunks,
            corpus_confidence=0.4,
        ))

        assert result["action"] == "abstain"
        assert result["hallucination_prob"] == 0.95
        assert result["confidence"] == 0.05
        assert result["signals"]["mean_entropy"] == 0.0
        assert result["signals"]["semantic_inconsistency"] == 0.0

    def test_empty_chunks_bypass_corpus_confidence_guard(self):
        """route_node must not trigger the corpus guard when chunks=[] (no slides uploaded).

        With no slides, corpus_confidence defaults to 1.0 and the guard is skipped.
        """
        with patch("src.abstention.pipeline.run_pipeline") as mock_run:
            mock_run.return_value = _mock_pipeline_result(hal_prob=0.2, decision="answer")
            result = route_node(_state(query="What is attention?", chunks=[], corpus_confidence=1.0))

        assert result["action"] == "answer"
        mock_run.assert_called_once()

    def test_no_slides_youtube_chunks_hard_abstains(self):
        """route_node must hard-abstain with reason 'no_slides' when has_slides=False
        and all retrieved chunks come from the YouTube pre-built corpus.

        No pipeline calls should be made -- this guard fires before any LLM call.
        """
        youtube_chunks = [
            {"text": "Attention allows models to focus on relevant input parts.",
             "metadata": {"source": "youtube:abc123", "page": 0}, "score": 0.04},
            {"text": "Self-attention computes queries, keys, and values.",
             "metadata": {"source": "youtube:abc123", "page": 0}, "score": 0.03},
        ]
        result = route_node(_state(
            query="What is the attention mechanism?",
            chunks=youtube_chunks,
            corpus_confidence=0.9,
            has_slides=False,
        ))

        assert result["action"] == "abstain"
        assert result["abstain_reason"] == "no_slides"
        assert result["hallucination_prob"] == 0.95
        assert result["confidence"] == 0.05

    def test_slides_uploaded_youtube_chunks_do_not_trigger_no_slides_guard(self):
        """When has_slides=True, the YouTube-source guard must not fire
        even if the chunk metadata happens to contain a YouTube source.
        """
        youtube_chunks = [
            {"text": "Some transcript content.",
             "metadata": {"source": "youtube:abc123", "page": 0}, "score": 0.04},
        ]
        with patch("src.abstention.pipeline.run_pipeline_with_context") as mock_run:
            mock_run.return_value = _mock_pipeline_result(hal_prob=0.1, decision="answer")
            result = route_node(_state(
                query="What is attention?",
                chunks=youtube_chunks,
                corpus_confidence=0.9,
                has_slides=True,
            ))

        assert result["action"] == "answer"
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# Full graph — generate_node skipped when action is abstain
# ---------------------------------------------------------------------------

class TestFullGraphAbstainPath:
    """Integration tests that run the compiled graph end-to-end with mocks."""

    @patch("src.agents.graph._ollama_chat")
    @patch("src.abstention.pipeline.run_pipeline")
    def test_generate_node_skipped_when_route_abstains(self, mock_run, mock_chat):
        """generate_node must not be called when route_node produces action='abstain'.

        Pipeline trace:
          classify_node  → LLM returns 'clear'  (call #1)
          retrieve_node  → no collection → chunks=[]
          route_node     → run_pipeline → hal_prob=0.9 → action='abstain'
          END            (generate_node and verify_node never execute)

        Total expected LLM calls: exactly 1.
        """
        mock_chat.return_value = "clear"
        mock_run.return_value = _mock_pipeline_result(hal_prob=0.9, decision="abstain")

        initial = _state(query="What is backpropagation?")
        result = calas_graph.invoke(initial)

        assert mock_chat.call_count == 1, (
            f"Expected 1 LLM call (classify only), got {mock_chat.call_count}"
        )
        assert result["action"] == "abstain"
        assert result["answer"] == ""

    @patch("src.agents.graph._ollama_chat")
    def test_clarify_path_ends_without_retrieval(self, mock_chat):
        """Graph must terminate at classify_node when the query is ambiguous.

        Pipeline trace:
          classify_node  → LLM returns 'ambiguous' → action='clarify'
          END            (retrieve_node never runs)

        Total expected LLM calls: exactly 1.
        """
        mock_chat.return_value = "ambiguous"

        initial = _state(query="explain")
        result = calas_graph.invoke(initial)

        assert mock_chat.call_count == 1
        assert result["action"] == "clarify"
        assert result["chunks"] == []

    @patch("src.agents.graph._ollama_chat")
    @patch("src.abstention.pipeline.run_pipeline_with_context")
    def test_full_answer_path_calls_llm_three_times(self, mock_run, mock_chat):
        """Full answer path must call the LLM for classify, generate, and verify.

        Pipeline trace:
          classify_node  → 'clear'                        (LLM call #1)
          retrieve_node  → patched stub   (no LLM call)
          route_node     → run_pipeline_with_context      → action='answer'
          generate_node  → 'The answer is X.'             (LLM call #2)
          verify_node    → 'yes'                          (LLM call #3)

        route_node calls run_pipeline_with_context (not run_pipeline) because
        the patched retrieve_node injects non-empty chunks.
        """
        mock_chat.side_effect = ["clear", "The answer is X.", "yes"]
        mock_run.return_value = _mock_pipeline_result(hal_prob=0.1, decision="answer")

        good_chunks = [
            {"text": "Backprop computes gradients via chain rule.",
             "metadata": {"source": "lec1.pdf", "page": 3},
             "score": 0.02},
        ]

        def _fake_make_retrieve(collection, all_chunks):
            def _node(state: dict) -> dict:
                return {**state, "chunks": good_chunks}
            return _node

        with patch("src.agents.graph._make_retrieve_node", side_effect=_fake_make_retrieve):
            graph = build_graph()
            result = graph.invoke(_state(query="Explain backpropagation."))

        assert mock_chat.call_count == 3
        assert result["action"] == "answer"
        assert result["answer"] == "The answer is X."
        assert result["confidence"] > 0.0

    @patch("src.agents.graph._ollama_chat")
    @patch("src.abstention.pipeline.run_pipeline_with_context")
    def test_verify_node_downgrades_ungrounded_answer(self, mock_run, mock_chat):
        """verify_node must set action='escalate' and reduce confidence when answer is ungrounded."""
        mock_chat.side_effect = ["clear", "Some hallucinated answer.", "no"]
        mock_run.return_value = _mock_pipeline_result(hal_prob=0.1, decision="answer")

        good_chunks = [
            {"text": "Backprop computes gradients.",
             "metadata": {"source": "lec1.pdf", "page": 1},
             "score": 0.02},
        ]

        def _fake_make_retrieve(collection, all_chunks):
            def _node(state: dict) -> dict:
                return {**state, "chunks": good_chunks}
            return _node

        with patch("src.agents.graph._make_retrieve_node", side_effect=_fake_make_retrieve):
            graph = build_graph()
            result = graph.invoke(_state(query="Explain backpropagation."))

        assert result["action"] == "escalate"
        assert result["confidence"] < 0.8
