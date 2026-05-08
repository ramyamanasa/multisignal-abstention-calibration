"""Evaluate the local Ollama model on data/eval/mmlu_ml.jsonl.

Each question is presented as a 4-choice multiple-choice prompt.
The model is asked to reply with a single letter (A–D).
Overall accuracy is reported at the end.

Usage:
    python scripts/eval_mmlu.py
    python scripts/eval_mmlu.py --file data/eval/mmlu_cs.jsonl
    python scripts/eval_mmlu.py --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
DEFAULT_FILE = Path(__file__).parent.parent / "data" / "eval" / "mmlu_ml.jsonl"
LABELS = ("A", "B", "C", "D")

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
def _ollama_chat(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# ---------------------------------------------------------------------------
# Prompt & parsing
# ---------------------------------------------------------------------------
def build_prompt(question: str, choices: list[str]) -> str:
    choices_block = "\n".join(f"{label}. {text}" for label, text in zip(LABELS, choices))
    return (
        "You are answering a machine learning multiple-choice exam.\n"
        "Read the question and the four options carefully, then reply with "
        "ONLY the single letter (A, B, C, or D) that corresponds to the correct answer. "
        "Do not explain your reasoning.\n\n"
        f"Question: {question}\n\n"
        f"{choices_block}\n\n"
        "Answer:"
    )


def parse_choice(response: str) -> str | None:
    """Return the first A/B/C/D found in the model response, or None."""
    match = re.search(r"\b([A-D])\b", response.upper())
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------
def evaluate(questions: list[dict], verbose: bool) -> dict:
    correct = 0
    unparsed = 0
    results: list[dict] = []

    for i, item in enumerate(questions, 1):
        question = item["question"]
        choices  = item["choices"]
        gold_idx = item["answer"]
        gold_lbl = LABELS[gold_idx]

        prompt   = build_prompt(question, choices)
        raw      = _ollama_chat(prompt)
        pred_lbl = parse_choice(raw)

        is_correct = pred_lbl == gold_lbl
        if pred_lbl is None:
            unparsed += 1
        elif is_correct:
            correct += 1

        results.append({
            "i":         i,
            "question":  question,
            "choices":   choices,
            "gold":      gold_lbl,
            "pred":      pred_lbl or "?",
            "correct":   is_correct,
            "raw":       raw,
        })

        # progress tick
        tick = "✓" if is_correct else ("?" if pred_lbl is None else "✗")
        print(f"[{i:>3}/{len(questions)}] {tick}  gold={gold_lbl}  pred={pred_lbl or '?'}")

        if verbose:
            q_short = textwrap.shorten(question, width=80, placeholder="…")
            print(f"      Q : {q_short}")
            for lbl, text in zip(LABELS, choices):
                marker = " ◄" if lbl == gold_lbl else ""
                print(f"      {lbl}: {text}{marker}")
            print(f"      raw response: {raw[:120]}")
            print()

    total    = len(questions)
    accuracy = correct / total if total else 0.0
    return {
        "total":    total,
        "correct":  correct,
        "wrong":    total - correct - unparsed,
        "unparsed": unparsed,
        "accuracy": accuracy,
        "results":  results,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_report(stats: dict, file: Path) -> None:
    total    = stats["total"]
    correct  = stats["correct"]
    wrong    = stats["wrong"]
    unparsed = stats["unparsed"]
    accuracy = stats["accuracy"]

    print(f"\n{'='*55}")
    print(f"  MMLU EVAL — {file.name}")
    print(f"{'='*55}")
    print(f"  Model          : {MODEL}")
    print(f"  Questions      : {total}")
    print(f"  Correct        : {correct}")
    print(f"  Wrong          : {wrong}")
    print(f"  Unparsed resp  : {unparsed}")
    print(f"  Accuracy       : {correct}/{total} = {accuracy:.1%}")
    print(f"{'='*55}")

    # ── per-question breakdown table ─────────────────────────────────────────
    print(f"\n{'#':>4}  {'Gold'}  {'Pred'}  {'':6}  Question")
    print("-" * 75)
    for r in stats["results"]:
        status = "CORRECT" if r["correct"] else ("UNPARSED" if r["pred"] == "?" else "wrong  ")
        q_short = textwrap.shorten(r["question"], width=55, placeholder="…")
        print(f"{r['i']:>4}  {r['gold']:^4}  {r['pred']:^4}  {status}  {q_short}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="MMLU accuracy eval against local Ollama model")
    parser.add_argument("--file",    type=Path, default=DEFAULT_FILE, help="Path to MMLU .jsonl file")
    parser.add_argument("--verbose", action="store_true",             help="Print full choices and raw response per question")
    args = parser.parse_args()

    if not args.file.exists():
        sys.exit(f"File not found: {args.file}")

    questions = load_jsonl(args.file)
    print(f"Loaded {len(questions)} questions from {args.file}")
    print(f"Model : {MODEL} @ {OLLAMA_BASE_URL}\n")

    stats = evaluate(questions, verbose=args.verbose)
    print_report(stats, args.file)


if __name__ == "__main__":
    main()
