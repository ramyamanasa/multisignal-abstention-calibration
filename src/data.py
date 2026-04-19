"""
Module: data.py
Owner: Person A (Signals)
Responsibility: HaluEval dataset loading, signal computation, feature CSV construction.

Uses HaluEval qa_samples which provides:
- question: the input question
- answer: a pre-generated answer (may be hallucinated)
- hallucination: 'yes' or 'no' label
- knowledge: reference context

We compute 3 signals on each (question, answer) pair:
- Signal 1: token entropy from opt-125m scoring the answer
- Signal 2: semantic consistency of Groq samples
- Signal 3: cross-model disagreement between opt-125m and Groq answers
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

from generation import load_model, generate_with_logprobs, generate_samples, generate_model2_answer
from signals import (
    compute_entropy_signal,
    compute_consistency_signal,
    compute_disagreement_signal,
    build_feature_vector,
)


# ---------------------------------------------------------------------------
# Signal 1: score a provided answer with opt-125m
# ---------------------------------------------------------------------------

def score_answer_with_logprobs(question: str, answer: str, local_tokenizer, local_model) -> dict:
    """
    Scores a pre-existing answer using opt-125m token log probs.
    This is different from generate_with_logprobs which generates its own answer.

    Returns dict with keys: answer_text, token_logprobs, tokens
    """
    import torch

    prompt = f"Question: {question}\nAnswer: {answer}"
    prompt_only = f"Question: {question}\nAnswer:"

    inputs     = local_tokenizer(prompt,      return_tensors="pt")
    input_len  = local_tokenizer(prompt_only, return_tensors="pt")["input_ids"].shape[1]

    with torch.no_grad():
        outputs = local_model(**inputs)
        logits  = outputs.logits  # (1, seq_len, vocab_size)

    import torch.nn.functional as F
    log_probs_all = F.log_softmax(logits[0], dim=-1)
    input_ids     = inputs["input_ids"][0]

    token_logprobs = []
    tokens         = []

    for i in range(input_len, len(input_ids)):
        token_id = input_ids[i].item()
        lp       = log_probs_all[i - 1, token_id].item()
        token_logprobs.append(lp)
        tokens.append(local_tokenizer.decode([token_id]))

    return {
        "answer_text":    answer,
        "token_logprobs": token_logprobs,
        "tokens":         tokens,
    }


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------

def build_feature_dataset(
    n_questions: int = 200,
    n_samples: int = 5,
    output_path: str = "../data/processed/features.csv",
    log_path: str = "../data/processed/raw_outputs.jsonl",
):
    """
    Loads HaluEval, computes signals on provided answers,
    saves feature CSV and raw outputs.
    """
    print("Loading HaluEval qa_samples...")
    dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    # Balance the dataset: equal hallucinated and non-hallucinated
    hal_examples     = [x for x in dataset if x["hallucination"] == "yes"]
    nonhal_examples  = [x for x in dataset if x["hallucination"] == "no"]

    n_each = n_questions // 2
    selected = hal_examples[:n_each] + nonhal_examples[:n_each]
    print(f"Selected {len(selected)} examples ({n_each} hallucinated, {n_each} non-hallucinated)")

    # Load local model for Signal 1
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading opt-125m for Signal 1 scoring...")
    local_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    local_model     = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m", dtype=torch.float32
    )
    local_model.eval()
    print("opt-125m loaded.")

    rows     = []
    raw_logs = []

    for i, item in enumerate(tqdm(selected)):
        question     = item["question"]
        answer       = item["answer"]
        is_hallucination = 1 if item["hallucination"] == "yes" else 0

        try:
            # Signal 1: score the provided answer with opt-125m
            scored = score_answer_with_logprobs(
                question, answer, local_tokenizer, local_model
            )
            entropy_signals = compute_entropy_signal(
                scored["token_logprobs"], scored["tokens"]
            )

            # Signal 2: Groq consistency sampling
            samples = generate_samples(question, n=n_samples)
            consistency_signals = compute_consistency_signal(samples)

            # Signal 3: disagreement between Groq answer and opt-125m answer
            groq_answer    = generate_model2_answer(question)
            opt_answer     = generate_with_logprobs(question)["answer_text"]
            disagreement_signals = compute_disagreement_signal(groq_answer, opt_answer)

            fv = build_feature_vector(
                entropy_signals, consistency_signals, disagreement_signals
            )

            row = {
                "question_id":              i,
                "question":                 question,
                "answer":                   answer,
                "is_hallucination":         is_hallucination,
                "mean_entropy":             fv[0],
                "max_entropy":              fv[1],
                "entity_entropy":           fv[2],
                "semantic_inconsistency":   fv[3],
                "cross_model_disagreement": fv[4],
            }
            rows.append(row)

            raw_logs.append({
                "question_id":   i,
                "question":      question,
                "answer":        answer,
                "label":         is_hallucination,
                "groq_answer":   groq_answer,
                "opt_answer":    opt_answer,
                "samples":       samples,
                "token_logprobs": scored["token_logprobs"],
            })

        except Exception as e:
            print(f"Error on question {i}: {e}")
            continue

    # Save
    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nFeature CSV saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nLabel distribution:")
    print(df["is_hallucination"].value_counts())
    print(f"\nFeature preview:")
    print(df[["mean_entropy", "max_entropy", "entity_entropy",
              "semantic_inconsistency", "cross_model_disagreement"]].describe())

    with open(log_path, "w") as f:
        for entry in raw_logs:
            f.write(json.dumps(entry) + "\n")
    print(f"Raw outputs saved to {log_path}")

    return df


if __name__ == "__main__":
    df = build_feature_dataset(
        n_questions=200,
        n_samples=5,
        output_path="../data/processed/features.csv",
        log_path="../data/processed/raw_outputs.jsonl",
    )