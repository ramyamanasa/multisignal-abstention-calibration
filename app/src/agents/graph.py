"""LangGraph StateGraph for the CALAS orchestration pipeline."""

from __future__ import annotations

import os
import time
from typing import Literal

from groq import Groq
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.retrieval.retrieve import hybrid_retrieve

_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
ABSTENTION_THRESHOLD = 0.75


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CALASState(TypedDict):
    """Shared state passed between all nodes in the CALAS pipeline."""

    query: str
    chunks: list[dict]
    answer: str
    confidence: float
    action: Literal["answer", "abstain", "clarify", "escalate"]
    mode: Literal["beginner", "exam", "deep"]
    signals: dict
    hallucination_prob: float
    corpus_confidence: float  # mean cosine similarity of top retrieved chunks (0-1)
    has_slides: bool          # True when the user uploaded a PDF/PPTX for this query
    abstain_reason: str       # "no_slides" | "not_covered" | "uncertain" | ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ollama_chat(prompt: str) -> str:
    """Call Groq API; function name preserved for test-mock compatibility.

    Args:
        prompt: The full prompt string to send to the model.

    Returns:
        The model's response as a plain string.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=_GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def classify_node(state: CALASState) -> CALASState:
    """Classify the query as clear, ambiguous, or out of scope."""
    prompt = (
        "Your only job is to decide whether a student's question has CLEAR INTENT.\n\n"
        "Reply 'clear' if the question has a specific, understandable topic — even if it\n"
        "might not be covered in the course. Short subject questions like 'What are LLMs?'\n"
        "or 'How does attention work?' are CLEAR. Questions about unrelated topics are also\n"
        "CLEAR — topic relevance is decided later, not here.\n\n"
        "Reply 'ambiguous' ONLY when the question is genuinely uninterpretable without more\n"
        "context — e.g. 'explain that', 'tell me more', 'what about it?', 'huh?'.\n\n"
        "Examples:\n"
        "  'What are LLMs?' → clear\n"
        "  'How does vLLM implement PagedAttention?' → clear\n"
        "  'What is 2+2?' → clear\n"
        "  'explain that' → ambiguous\n"
        "  'tell me more' → ambiguous\n"
        "  'what do you mean?' → ambiguous\n\n"
        f"Question: {state['query']}\n\n"
        "Reply with only one word: clear or ambiguous."
    )
    label = _ollama_chat(prompt).lower().strip().rstrip(".")

    if "ambiguous" in label:
        return {**state, "action": "clarify"}
    return {**state, "action": "answer"}


def _chunks_from_collection(collection) -> list[dict]:
    """Reconstruct a list of chunk dicts from a ChromaDB collection."""
    result = collection.get(include=["documents", "metadatas"])
    return [
        {"text": doc, **meta}
        for doc, meta in zip(result["documents"], result["metadatas"])
    ]


def _make_retrieve_node(collection, all_chunks: list[dict]):
    """Return a retrieve_node closure bound to a specific index."""
    if collection is not None and not all_chunks:
        all_chunks = _chunks_from_collection(collection)

    def retrieve_node(state: CALASState) -> CALASState:
        if collection is None or not all_chunks:
            return {**state, "chunks": [], "corpus_confidence": 1.0}
        chunks = hybrid_retrieve(state["query"], collection, all_chunks, k=5)
        corpus_confidence = getattr(chunks, "corpus_confidence", 1.0)
        return {**state, "chunks": chunks, "corpus_confidence": corpus_confidence}

    return retrieve_node


_ZERO_SIGNALS = {
    "mean_entropy": 0.0, "max_entropy": 0.0, "entity_entropy": 0.0,
    "semantic_inconsistency": 0.0, "cross_model_disagreement": 0.0,
}


def route_node(state: CALASState) -> CALASState:
    """Use the multi-signal meta-classifier to decide whether to answer or abstain.

    Three layered guards before the LLM pipeline:

    Guard 1 (no_slides): No PDF uploaded and all retrieved chunks are from the
    YouTube pre-built corpus. Never answer general knowledge questions from training
    data -- require the user to upload slides first.

    Guard 2 (not_covered): Slides were uploaded but corpus_confidence < 0.65,
    meaning the top retrieved chunks have low cosine similarity to the query.
    The slides don't cover this topic.

    Guard 3 (uncertain): Signals from the 3-signal pipeline exceed the abstention
    threshold -- the model found relevant content but cannot answer confidently.
    """
    chunks = state.get("chunks", [])
    corpus_confidence = state.get("corpus_confidence", 1.0)
    has_slides = state.get("has_slides", False)

    # Guard 1: no slides uploaded and corpus is entirely YouTube transcripts
    if not has_slides and chunks:
        all_youtube = all(
            c.get("metadata", {}).get("source", "").startswith("youtube:")
            for c in chunks
        )
        if all_youtube:
            return {
                **state,
                "action": "abstain",
                "confidence": 0.05,
                "hallucination_prob": 0.95,
                "signals": _ZERO_SIGNALS,
                "abstain_reason": "no_slides",
            }

    # Guard 2: slides uploaded but don't cover the queried topic
    if chunks and corpus_confidence < 0.65:
        return {
            **state,
            "action": "abstain",
            "confidence": 0.05,
            "hallucination_prob": 0.95,
            "signals": _ZERO_SIGNALS,
            "abstain_reason": "not_covered",
        }

    if chunks:
        from src.abstention.pipeline import run_pipeline_with_context
        context = "\n\n".join(c["text"] for c in chunks)
        result = run_pipeline_with_context(state["query"], context, threshold=ABSTENTION_THRESHOLD)
    else:
        from src.abstention.pipeline import run_pipeline
        result = run_pipeline(state["query"], threshold=ABSTENTION_THRESHOLD)

    hal_prob = result["hallucination_probability"]
    action = result["decision"]
    all_signals = {
        **result.get("entropy_signals", {}),
        **result.get("consistency_signals", {}),
        **result.get("disagreement_signals", {}),
    }
    return {
        **state,
        "action": action,
        "confidence": round(1.0 - hal_prob, 4),
        "hallucination_prob": hal_prob,
        "signals": all_signals,
        "abstain_reason": "uncertain" if action == "abstain" else "",
    }


def generate_node(state: CALASState) -> CALASState:
    """Generate an answer grounded in the retrieved chunks via Groq."""
    mode_instructions = {
        "beginner": "Use simple language, define technical terms, and give an analogy.",
        "exam": "Be concise and precise. Use bullet points for key facts.",
        "deep": "Provide a thorough explanation with reasoning and edge cases.",
    }
    instruction = mode_instructions.get(state.get("mode", "beginner"), mode_instructions["beginner"])

    context_block = "\n\n".join(
        f"[{i+1}] (source: {c['metadata'].get('source','?')}, p.{c['metadata'].get('page','?')})\n{c['text']}"
        for i, c in enumerate(state["chunks"])
    )

    prompt = (
        f"You are a helpful teaching assistant.\n{instruction}\n\n"
        f"Context from lecture materials:\n{context_block}\n\n"
        f"Student question: {state['query']}\n\n"
        "Answer (cite sources by number):"
    )

    answer = _ollama_chat(prompt)
    return {**state, "answer": answer}


def verify_node(state: CALASState) -> CALASState:
    """Check whether the generated answer is grounded in the retrieved chunks."""
    context_block = "\n\n".join(c["text"] for c in state["chunks"])
    prompt = (
        "You are a grounding checker. Your job is to catch CLEAR HALLUCINATIONS — "
        "specific factual claims in the answer that directly contradict or are entirely "
        "absent from the context below.\n\n"
        "Reply 'no' ONLY when the answer makes a concrete factual assertion that has NO "
        "basis anywhere in the context.\n\n"
        "Reply 'yes' when:\n"
        "  - The answer is mostly grounded in the context, even if it adds a brief analogy,\n"
        "    a connecting inference, or a clarifying sentence.\n"
        "  - Minor elaboration that does not introduce conflicting facts is present.\n"
        "  - The answer uses simpler language or rephrases content from the context.\n\n"
        "Context:\n"
        f"{context_block}\n\n"
        f"Answer:\n{state['answer']}\n\n"
        "Reply with only one word: yes or no."
    )
    verdict = _ollama_chat(prompt).lower().strip().rstrip(".")

    if "no" in verdict:
        return {
            **state,
            "confidence": round(state["confidence"] * 0.6, 4),
            "action": "escalate",
        }
    return state


# ---------------------------------------------------------------------------
# Routing functions (edges)
# ---------------------------------------------------------------------------

def _after_classify(state: CALASState) -> str:
    if state["action"] == "clarify":
        return END
    if state["action"] == "abstain":
        return END
    return "retrieve_node"


def _after_route(state: CALASState) -> str:
    if state["action"] == "answer":
        return "generate_node"
    return END


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(collection=None, all_chunks: list[dict] | None = None) -> StateGraph:
    """Build and compile the CALAS LangGraph StateGraph."""
    graph = StateGraph(CALASState)

    graph.add_node("classify_node", classify_node)
    graph.add_node("retrieve_node", _make_retrieve_node(collection, all_chunks or []))
    graph.add_node("route_node", route_node)
    graph.add_node("generate_node", generate_node)
    graph.add_node("verify_node", verify_node)

    graph.add_edge(START, "classify_node")
    graph.add_conditional_edges("classify_node", _after_classify)
    graph.add_edge("retrieve_node", "route_node")
    graph.add_conditional_edges("route_node", _after_route)
    graph.add_edge("generate_node", "verify_node")
    graph.add_edge("verify_node", END)

    return graph.compile()


# Module-level compiled graph (no index — callers should use build_graph).
calas_graph = build_graph()
