"""
OOD Evaluation on TriviaQA
Tests whether signals trained on HaluEval transfer to a different dataset.
Directly addresses Research Question 2 from the proposal.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from generation import load_model, generate_with_logprobs, generate_samples, generate_model2_answer
from signals import (
    compute_entropy_signal,
    compute_consistency_signal,
    compute_disagreement_signal,
    build_feature_vector,
)
from evaluation import compute_auroc, compute_ece, plot_reliability_diagram


FEATURE_COLS = [
    "mean_entropy",
    "max_entropy",
    "entity_entropy",
    "semantic_inconsistency",
    "cross_model_disagreement",
]


def normalize_answer(text: str) -> str:
    return text.lower().strip().rstrip(".,?!")


def is_correct(predicted: str, references: list) -> int:
    pred_tokens = set(normalize_answer(predicted).split())
    for ref in references:
        ref_tokens = set(normalize_answer(ref).split())
        if len(pred_tokens & ref_tokens) > 0:
            return 1
    return 0


def build_ood_features(
    n_questions: int = 150,
    n_samples: int = 5,
    output_path: str = "../data/processed/ood_features.csv",
    log_path: str = "../data/processed/ood_raw_outputs.jsonl",
):
    print("Loading TriviaQA...")
    dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading opt-125m for Signal 1 scoring...")
    local_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    local_model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m", torch_dtype=torch.float32
)
    local_model.eval()

    rows = []
    raw_logs = []
    n_correct = 0
    n_hallucination = 0
    target_each = n_questions // 2

    print(f"Targeting {target_each} correct and {target_each} hallucinated examples...")

    for i, item in enumerate(tqdm(dataset)):
        if n_correct >= target_each and n_hallucination >= target_each:
            break

        question   = item["question"]
        references = item["answer"]["aliases"]

        try:
            # Get Groq answer
            groq_answer = generate_model2_answer(question)
            label       = is_correct(groq_answer, references)

            # Skip if we already have enough of this label
            if label == 1 and n_correct >= target_each:
                continue
            if label == 0 and n_hallucination >= target_each:
                continue

            # Signal 1: score the Groq answer with opt-125m
            prompt     = f"Question: {question}\nAnswer: {groq_answer}"
            prompt_only = f"Question: {question}\nAnswer:"
            inputs     = local_tokenizer(prompt, return_tensors="pt")
            input_len  = local_tokenizer(prompt_only, return_tensors="pt")["input_ids"].shape[1]

            import torch.nn.functional as F
            with torch.no_grad():
                outputs = local_model(**inputs)
                logits  = outputs.logits

            log_probs_all = F.log_softmax(logits[0], dim=-1)
            input_ids     = inputs["input_ids"][0]

            token_logprobs = []
            tokens         = []
            for j in range(input_len, len(input_ids)):
                tid = input_ids[j].item()
                lp  = log_probs_all[j - 1, tid].item()
                token_logprobs.append(lp)
                tokens.append(local_tokenizer.decode([tid]))

            entropy_signals     = compute_entropy_signal(token_logprobs, tokens)
            samples             = generate_samples(question, n=n_samples, temperature=1.0)
            consistency_signals = compute_consistency_signal(samples)
            opt_answer          = generate_with_logprobs(question)["answer_text"]
            disagreement_signals = compute_disagreement_signal(groq_answer, opt_answer)

            fv = build_feature_vector(
                entropy_signals, consistency_signals, disagreement_signals
            )

            is_hallucination = 1 - label
            if label == 1:
                n_correct += 1
            else:
                n_hallucination += 1

            rows.append({
                "question_id":              len(rows),
                "question":                 question,
                "answer":                   groq_answer,
                "is_hallucination":         is_hallucination,
                "mean_entropy":             fv[0],
                "max_entropy":              fv[1],
                "entity_entropy":           fv[2],
                "semantic_inconsistency":   fv[3],
                "cross_model_disagreement": fv[4],
            })

            raw_logs.append({
                "question_id":    len(raw_logs),
                "question":       question,
                "groq_answer":    groq_answer,
                "references":     references[:3],
                "label":          label,
                "samples":        samples,
                "token_logprobs": token_logprobs,
            })

        except Exception as e:
            print(f"Error on question {i}: {e}")
            continue

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nOOD feature CSV saved: {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Label distribution:\n{df['is_hallucination'].value_counts()}")

    with open(log_path, "w") as f:
        for entry in raw_logs:
            f.write(json.dumps(entry) + "\n")

    return df


def evaluate_ood(
    ood_features_path: str = "../data/processed/ood_features.csv",
    clf_path:          str = "../models/meta_clf.pkl",
):
    print("\nLoading trained classifier...")
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)

    df = pd.read_csv(ood_features_path)
    df["cross_model_disagreement"] = df["cross_model_disagreement"].clip(0, 1)

    X = df[FEATURE_COLS].values
    y = df["is_hallucination"].values

    probs = clf.predict_proba(X)[:, 1]

    auroc = compute_auroc(y, probs)
    ece   = compute_ece(y, probs)

    # Bootstrap CI
    n_bootstrap = 1000
    aucs = []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y), len(y), replace=True)
        try:
            aucs.append(roc_auc_score(y[idx], probs[idx]))
        except:
            pass
    aucs = np.array(aucs)

    print(f"\nOOD Evaluation Results (TriviaQA):")
    print(f"  AUROC: {auroc:.4f}  95% CI: [{np.percentile(aucs,2.5):.4f}, {np.percentile(aucs,97.5):.4f}]")
    print(f"  ECE:   {ece:.4f}")
    print(f"  N:     {len(y)}")

    plot_reliability_diagram(
        y, probs,
        save_path="../data/processed/ood_reliability_diagram.png"
    )

    # Save results
    results = {
        "dataset":  "TriviaQA",
        "n":        len(y),
        "auroc":    round(auroc, 4),
        "ece":      round(ece, 4),
        "auroc_ci": [round(np.percentile(aucs,2.5),4), round(np.percentile(aucs,97.5),4)],
    }
    with open("../experiments/exp003_ood_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print("OOD results saved to experiments/exp003_ood_eval.json")

    return results


if __name__ == "__main__":
    load_model()

    # Step 1: Build OOD features
    df = build_ood_features(
        n_questions=150,
        n_samples=5,
        output_path="../data/processed/ood_features.csv",
        log_path="../data/processed/ood_raw_outputs.jsonl",
    )

    # Step 2: Evaluate
    results = evaluate_ood()