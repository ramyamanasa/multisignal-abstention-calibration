"""
Module: pipeline.py (lectureOS abstention)
Adapted from multisignal-abstention-calibration/src/pipeline.py
Uses lazy initialization so opt-125m is not loaded at import time.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Allow sibling imports within src/abstention/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generation import (
    load_model,
    generate_with_logprobs,
    generate_samples,
    generate_primary_answer,
    generate_model2_answer,
)
from signals import (
    compute_entropy_signal,
    compute_consistency_signal,
    compute_disagreement_signal,
    build_feature_vector,
)
from fusion import load_classifier, predict_with_abstention

MODEL_PATH = Path(__file__).parent / "meta_clf.pkl"

_OPT_MAX_CONTEXT_CHARS = 800

_clf = None
_initialized = False


def _ensure_initialized():
    global _clf, _initialized
    if _initialized:
        return
    load_model()
    _clf = load_classifier(str(MODEL_PATH))
    _initialized = True


def run_pipeline(question: str, threshold: float = 0.5) -> dict:
    """
    Full 3-signal pipeline for a single input question.

    Returns dict with keys:
        question                  (str)
        answer                    (str)   from llama-3.3-70b-versatile (secondary)
        primary_answer            (str)   from llama-3.1-8b-instant (primary)
        entropy_signals           (dict)
        consistency_signals       (dict)
        disagreement_signals      (dict)
        feature_vector            (list of 5 floats)
        hallucination_probability (float)
        decision                  (str)   'answer' or 'abstain'
        threshold                 (float)
        samples                   (list)
    """
    _ensure_initialized()

    result = generate_with_logprobs(question)
    entropy_signals = compute_entropy_signal(
        result["token_logprobs"], result["tokens"]
    )

    samples = generate_samples(question, n=5, temperature=1.0)
    consistency_signals = compute_consistency_signal(samples)

    primary_answer = generate_primary_answer(question)
    secondary_answer = generate_model2_answer(question)
    disagreement_signals = compute_disagreement_signal(secondary_answer, primary_answer)

    fv = build_feature_vector(
        entropy_signals, consistency_signals, disagreement_signals
    )

    X = fv.reshape(1, -1)
    decisions, probs = predict_with_abstention(_clf, X, threshold=threshold)

    return {
        "question":                  question,
        "answer":                    secondary_answer,
        "primary_answer":            primary_answer,
        "entropy_signals":           entropy_signals,
        "consistency_signals":       consistency_signals,
        "disagreement_signals":      disagreement_signals,
        "feature_vector":            fv.tolist(),
        "hallucination_probability": round(probs[0], 4),
        "decision":                  decisions[0],
        "threshold":                 threshold,
        "samples":                   samples,
    }


def run_pipeline_with_context(question: str, context: str, threshold: float = 0.5) -> dict:
    """Context-aware 3-signal pipeline.

    All three signals are computed on context-grounded prompts so that
    questions answerable from the retrieved slides yield LOW uncertainty
    (models converge) while off-topic questions yield HIGH uncertainty
    (models diverge on irrelevant context).

    Returns the same keys as run_pipeline.
    """
    _ensure_initialized()

    truncated_context = context[:_OPT_MAX_CONTEXT_CHARS]
    opt_prompt  = f"Based on this context:\n{truncated_context}\n\nAnswer this question: {question}"
    groq_prompt = f"Based on this context:\n{context}\n\nAnswer this question: {question}"

    result = generate_with_logprobs(opt_prompt)
    entropy_signals = compute_entropy_signal(result["token_logprobs"], result["tokens"])

    samples = generate_samples(groq_prompt, n=5, temperature=1.0)
    consistency_signals = compute_consistency_signal(samples)

    primary_answer   = generate_primary_answer(groq_prompt)
    secondary_answer = generate_model2_answer(groq_prompt)
    disagreement_signals = compute_disagreement_signal(secondary_answer, primary_answer)

    fv = build_feature_vector(entropy_signals, consistency_signals, disagreement_signals)
    X  = fv.reshape(1, -1)
    decisions, probs = predict_with_abstention(_clf, X, threshold=threshold)

    return {
        "question":                  question,
        "answer":                    secondary_answer,
        "primary_answer":            primary_answer,
        "entropy_signals":           entropy_signals,
        "consistency_signals":       consistency_signals,
        "disagreement_signals":      disagreement_signals,
        "feature_vector":            fv.tolist(),
        "hallucination_probability": round(probs[0], 4),
        "decision":                  decisions[0],
        "threshold":                 threshold,
        "samples":                   samples,
    }
