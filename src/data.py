"""
Module: data.py
Owner: Person A (Signals)
Responsibility: Dataset loading, answer correctness labeling, feature CSV construction.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

from generation import load_model, generate_with_logprobs, generate_samples
from signals import (
    compute_entropy_signal,
    compute_consistency_signal,
    compute_disagreement_signal,
    build_feature_vector,
)


# ---------------------------------------------------------------------------
# Correctness scoring
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    return text.lower().strip().rstrip(".")


def is_correct(predicted: str, references: list) -> int:
    """
    Returns 1 if predicted answer matches any reference answer, 0 otherwise.
    Uses token-level overlap (simple but standard for TriviaQA).
    """
    pred_tokens = set(normalize(predicted).split())
    for ref in references:
        ref_tokens = set(normalize(ref).split())
        if len(pred_tokens & ref_tokens) > 0:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------

def build_feature_dataset(
    n_questions: int = 100,
    n_samples: int = 5,
    output_path: str = "../data/processed/features.csv",
    log_path: str = "../data/processed/raw_outputs.jsonl",
):
    """
    Loads TriviaQA, runs the full signal pipeline on n_questions,
    saves feature CSV and raw outputs.
    """
    print(f"Loading TriviaQA...")
    dataset = load_dataset("trivia_qa", "rc.nocontext", split="train")

    print(f"Running pipeline on {n_questions} questions...")
    tokenizer, model = load_model("facebook/opt-125m")

    rows = []
    raw_logs = []

    for i in tqdm(range(n_questions)):
        item = dataset[i]
        question = item["question"]
        references = item["answer"]["aliases"]

        # Generation
        result = generate_with_logprobs(question, tokenizer, model)
        samples = generate_samples(question, tokenizer, model, n=n_samples)

        # Correctness label
        label = is_correct(result["answer_text"], references)

        # Signals
        entropy_signals      = compute_entropy_signal(result["token_logprobs"], result["tokens"])
        consistency_signals  = compute_consistency_signal(samples)
        disagreement_signals = compute_disagreement_signal(
            result["answer_text"], samples[0] if samples else ""
        )

        fv = build_feature_vector(entropy_signals, consistency_signals, disagreement_signals)

        row = {
            "question_id":             i,
            "question":                question,
            "predicted_answer":        result["answer_text"],
            "is_hallucination":        1 - label,   # 1 = wrong, 0 = correct
            "mean_entropy":            fv[0],
            "max_entropy":             fv[1],
            "entity_entropy":          fv[2],
            "semantic_inconsistency":  fv[3],
            "cross_model_disagreement": fv[4],
        }
        rows.append(row)

        # Raw log for debugging
        raw_logs.append({
            "question_id": i,
            "question":    question,
            "references":  references[:3],
            "predicted":   result["answer_text"],
            "label":       label,
            "samples":     samples,
            "token_logprobs": result["token_logprobs"],
        })

    # Save
    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Feature CSV saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nLabel distribution:")
    print(df["is_hallucination"].value_counts())
    print(f"\nFeature preview:")
    print(df[["mean_entropy","max_entropy","entity_entropy",
              "semantic_inconsistency","cross_model_disagreement"]].describe())

    with open(log_path, "w") as f:
        for entry in raw_logs:
            f.write(json.dumps(entry) + "\n")
    print(f"Raw outputs saved to {log_path}")

    return df


if __name__ == "__main__":
    df = build_feature_dataset(
        n_questions=100,
        n_samples=5,
        output_path="../data/processed/features.csv",
        log_path="../data/processed/raw_outputs.jsonl",
    )