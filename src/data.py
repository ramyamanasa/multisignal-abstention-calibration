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
- Signal 1: token entropy from opt-125m scoring the provided answer
- Signal 2: semantic consistency of Groq llama-3.1-8b-instant samples
- Signal 3: cross-model disagreement — llama-3.1-8b-instant vs llama-3.3-70b-versatile

Balance guarantee: examples are interleaved (hal[0], correct[0], hal[1], correct[1], ...)
so every checkpoint snapshot is balanced regardless of when the run stops.
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

from generation import load_model, generate_samples, generate_primary_answer, generate_model2_answer
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
    n_questions: int = 1000,
    n_samples: int = 5,
    output_path: str = "../data/processed/features.csv",
    log_path: str = "../data/processed/raw_outputs.jsonl",
    checkpoint_path: str = "../data/processed/features_checkpoint.csv",
):
    """
    Loads HaluEval, computes signals on provided answers,
    saves feature CSV and raw outputs.

    Balance strategy: examples are interleaved (hal, correct, hal, correct, ...)
    so checkpoints are balanced at every 50-row boundary.

    n_correct and n_hallucinated are SESSION-ONLY counters (not loaded from
    previous checkpoint). The final balance check uses these session counters.
    If |n_correct - n_hallucinated| > 50 the run saves to features_unbalanced.csv
    instead of features.csv and prints a WARNING.

    Resumes from checkpoint_path if it exists (skips processed question_ids).
    Checkpoints are only written when BOTH labels have been seen in this session.
    Sleeps 2 seconds after each Groq call group.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Loading HaluEval qa_samples...")
    dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    hal_examples    = [x for x in dataset if x["hallucination"] == "yes"]
    nonhal_examples = [x for x in dataset if x["hallucination"] == "no"]
    n_each = n_questions // 2
    print(f"Available: {len(hal_examples)} hallucinated, {len(nonhal_examples)} correct")
    print(f"Target:    {n_each} of each = {n_each * 2} total")

    # Interleave so every checkpoint snapshot is balanced
    selected = []
    for h, c in zip(hal_examples[:n_each], nonhal_examples[:n_each]):
        selected.append(h)
        selected.append(c)
    print(f"Interleaved: {len(selected)} examples  "
          f"(pattern: hal[0], correct[0], hal[1], correct[1], ...)")

    # ── Resume: load processed IDs and prior rows from checkpoint ──────────────
    processed_ids = set()
    prior_rows    = []
    if Path(checkpoint_path).exists():
        df_ckpt       = pd.read_csv(checkpoint_path)
        processed_ids = set(df_ckpt["question_id"].tolist())
        prior_rows    = df_ckpt.to_dict("records")
        n_h_prior = sum(1 for r in prior_rows if r["is_hallucination"] == 1)
        n_c_prior = sum(1 for r in prior_rows if r["is_hallucination"] == 0)
        print(f"Resuming:  {len(processed_ids)} already done "
              f"({n_h_prior} hal, {n_c_prior} correct in checkpoint)")
        if abs(n_h_prior - n_c_prior) > 50:
            print(f"WARNING: checkpoint is imbalanced by {abs(n_h_prior - n_c_prior)} — "
                  f"interleaving will correct this session")

    n_to_process = len(selected) - len(processed_ids)
    print(f"Remaining: {n_to_process} examples to process\n")

    # ── Load opt-125m for Signal 1 ────────────────────────────────────────────
    print("Loading opt-125m for Signal 1 scoring...")
    local_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    local_model     = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m", dtype=torch.float32
    )
    local_model.eval()
    print("opt-125m loaded.\n")

    # SESSION-ONLY counters — do not include prior_rows
    rows           = []
    n_hallucinated = 0
    n_correct      = 0
    start_time     = time.time()
    n_processed    = 0

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a")

    try:
        with tqdm(total=n_to_process, desc="Processing") as pbar:
            for i, item in enumerate(selected):

                # ── Skip already-processed examples ───────────────────────────
                if i in processed_ids:
                    continue

                question         = item["question"]
                answer           = item["answer"]
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
                    samples = generate_samples(question, n=n_samples, temperature=1.0)
                    time.sleep(2)
                    consistency_signals = compute_consistency_signal(samples)

                    # Signal 3: primary vs secondary Groq
                    primary_answer   = generate_primary_answer(question)
                    time.sleep(2)
                    secondary_answer = generate_model2_answer(question)
                    time.sleep(2)
                    disagreement_signals = compute_disagreement_signal(
                        secondary_answer, primary_answer
                    )

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

                    # SESSION-ONLY balance counters
                    if is_hallucination == 1:
                        n_hallucinated += 1
                    else:
                        n_correct += 1

                    log_entry = {
                        "question_id":      i,
                        "question":         question,
                        "answer":           answer,
                        "label":            is_hallucination,
                        "primary_answer":   primary_answer,
                        "secondary_answer": secondary_answer,
                        "samples":          samples,
                        "token_logprobs":   scored["token_logprobs"],
                    }
                    log_file.write(json.dumps(log_entry) + "\n")
                    log_file.flush()

                    n_processed += 1

                    # ── Progress ───────────────────────────────────────────────
                    elapsed   = time.time() - start_time
                    rate      = n_processed / elapsed
                    remaining = n_to_process - n_processed
                    eta_s     = remaining / rate if rate > 0 else 0
                    eta_str   = (f"{eta_s / 3600:.1f}h" if eta_s >= 3600
                                 else f"{eta_s / 60:.1f}m" if eta_s >= 60
                                 else f"{eta_s:.0f}s")
                    pbar.set_postfix({
                        "hal": n_hallucinated,
                        "cor": n_correct,
                        "ETA": eta_str,
                    })
                    pbar.update(1)

                    # ── Checkpoint every 50 new rows ───────────────────────────
                    # Only save when BOTH labels have been seen in THIS session.
                    if n_processed % 50 == 0:
                        if n_correct > 0 and n_hallucinated > 0:
                            all_so_far = prior_rows + rows
                            pd.DataFrame(all_so_far).to_csv(checkpoint_path, index=False)
                            tqdm.write(
                                f"[checkpoint] {len(all_so_far)} total rows  "
                                f"session: {n_hallucinated} hal, {n_correct} correct"
                            )
                            if abs(n_hallucinated - n_correct) > 50:
                                tqdm.write(
                                    f"[WARNING] Session imbalance: "
                                    f"{abs(n_hallucinated - n_correct)} row gap"
                                )
                        else:
                            tqdm.write(
                                f"[checkpoint skipped] only one label so far  "
                                f"(hal={n_hallucinated}, cor={n_correct})"
                            )

                except Exception as e:
                    tqdm.write(f"[error] question {i}: {e}")
                    continue

    finally:
        log_file.close()

    # ── Final balance check using SESSION-ONLY counters ───────────────────────
    print(f"\n{'='*60}")
    print(f"Session totals: {n_hallucinated} hallucinated, {n_correct} correct")
    imbalance = abs(n_correct - n_hallucinated)

    all_rows = prior_rows + rows
    df = pd.DataFrame(all_rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if imbalance > 50:
        unbalanced_path = str(Path(output_path).parent / "features_unbalanced.csv")
        df.to_csv(unbalanced_path, index=False)
        print(f"WARNING: session imbalanced by {imbalance} rows "
              f"({n_hallucinated} hal vs {n_correct} correct)")
        print(f"WARNING: do not use this for training")
        print(f"Saved {len(df)} rows → {unbalanced_path}")
    else:
        df.to_csv(output_path, index=False)
        df.to_csv(checkpoint_path, index=False)
        print(f"SUCCESS: balanced dataset  "
              f"({n_hallucinated} hal, {n_correct} correct, diff={imbalance})")
        print(f"Saved {len(df)} rows → {output_path}")

    print(f"\nFeature preview:")
    print(df[["mean_entropy", "max_entropy", "entity_entropy",
              "semantic_inconsistency", "cross_model_disagreement"]].describe())
    print(f"Raw outputs → {log_path}")

    return df


if __name__ == "__main__":
    df = build_feature_dataset(
        n_questions=1000,
        n_samples=5,
        output_path="../data/processed/features.csv",
        log_path="../data/processed/raw_outputs.jsonl",
        checkpoint_path="../data/processed/features_checkpoint.csv",
    )
