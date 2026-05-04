"""
Module: pipeline.py
Owner: Person C (Integration + Demo)
Responsibility: End-to-end function for a single query.
"""

import os
import sys
import numpy as np

# Allow imports from src/
sys.path.insert(0, os.path.dirname(__file__))

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

# Load model and classifier once at import time
print("Initializing pipeline...")
load_model()
clf = load_classifier()
print("Pipeline ready.")


def run_pipeline(question: str, threshold: float = 0.95) -> dict:
    """
    Full pipeline for a single input question.

    Returns dict with keys:
        question                  (str)
        answer                    (str)   from opt-125m
        groq_answer               (str)   from Groq
        entropy_signals           (dict)
        consistency_signals       (dict)
        disagreement_signals      (dict)
        feature_vector            (list of 5 floats)
        hallucination_probability (float)
        decision                  (str)   'answer' or 'abstain'
        threshold                 (float)
    """
    # Signal 1: opt-125m answer + token log probs
    result         = generate_with_logprobs(question)
    entropy_signals = compute_entropy_signal(
        result["token_logprobs"], result["tokens"]
    )

    # Signal 2: Groq consistency sampling
    samples = generate_samples(question, n=5, temperature=1.0)

    consistency_signals = compute_consistency_signal(samples)

    # Signal 3: cross-model disagreement (primary vs secondary Groq)
    primary_answer       = generate_primary_answer(question)
    secondary_answer     = generate_model2_answer(question)
    disagreement_signals = compute_disagreement_signal(secondary_answer, primary_answer)

    # Feature vector
    fv = build_feature_vector(
        entropy_signals, consistency_signals, disagreement_signals
    )

    # Fusion model prediction
    X = fv.reshape(1, -1)
    decisions, probs = predict_with_abstention(clf, X, threshold=threshold)

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


if __name__ == "__main__":
    TEST_QUESTIONS = [
        "Who wrote the play Hamlet?",
        "What is the capital of Australia?",
        "What year did World War I end?",
        "Who invented the telephone?",
        "What is the largest ocean on Earth?",
    ]

    print("\n" + "="*60)
    for q in TEST_QUESTIONS:
        print(f"\nQuestion: {q}")
        result = run_pipeline(q, threshold=0.95)
        print(f"Answer:      {result['answer']}")
        print(f"P(halluc):   {result['hallucination_probability']}")
        print(f"Decision:    {result['decision'].upper()}")
        print(f"Entropy:     mean={result['entropy_signals']['mean_entropy']:.4f}  "
              f"max={result['entropy_signals']['max_entropy']:.4f}")
        print(f"Consistency: {result['consistency_signals']['semantic_inconsistency']:.4f}")
        print(f"Disagreement:{result['disagreement_signals']['cross_model_disagreement']:.4f}")
        print("-"*60)