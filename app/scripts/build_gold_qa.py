"""Curate data/eval/gold_candidates.jsonl → data/eval/gold_qa.jsonl.

Delete rules
------------
* Answer is a single word, bare number, or date
* Answer is under 10 words
* Answer contains only vague spatial references ("here", "at the top here", …)
* Answer is a type annotation or function/variable identifier
* Question asks what will be discussed / covered next

Keep + improve rules
--------------------
* Answer explains a concept (passes all delete rules)
* If kept answer is < 15 words → Ollama expands it to a full sentence

Abstention questions are appended at the end.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
SRC  = Path(__file__).parent.parent / "data" / "eval" / "gold_candidates.jsonl"
DST  = Path(__file__).parent.parent / "data" / "eval" / "gold_qa.jsonl"
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL      = os.getenv("OLLAMA_MODEL",    "qwen2.5:7b-instruct")
MIN_WORDS  = 15   # target answer length after expansion

# ---------------------------------------------------------------------------
# Abstention questions
# ---------------------------------------------------------------------------
ABSTAIN_QUESTIONS = [
    "How does vLLM implement PagedAttention?",
    "What is the architecture of GPT-4?",
    "How does Gemini handle multimodal inputs?",
    "What BLEU scores did the original attention paper report?",
    "How does Claude 3 handle tool use?",
    "What is the Columbia GR5293 final exam format?",
    "How does Flash Attention reduce memory usage?",
    "What are the exact hyperparameters used in GPT-3 training?",
    "How does Mistral implement sliding window attention?",
    "What is the difference between PPO and GRPO?",
]

# ---------------------------------------------------------------------------
# Delete predicates  (any True → drop the row)
# ---------------------------------------------------------------------------
def _is_date(a: str) -> bool:
    return bool(re.search(
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d+,?\s*\d{4}\b", a
    ))

def _is_bare_number(a: str) -> bool:
    return bool(re.match(r"^\s*[\d./]+\s*$", a.strip()))

def _is_type_or_identifier(a: str) -> bool:
    # Optional[...], bare function call, bare snake_case identifier
    if re.search(r"Optional\[|^\s*\w+\s*\(\)\s*$", a):
        return True
    # two-word or fewer identifier-style tokens with no spaces between words
    stripped = a.strip()
    if re.match(r"^[a-z_][a-zA-Z0-9_]*$", stripped):   # single CamelCase / snake identifier
        return True
    return False

def _is_vague_here(a: str) -> bool:
    """Answers whose substance is just a spatial pointer."""
    low = a.strip().lower().rstrip(".")
    triggers = {
        "at the top here", "here", "in exactly these two places here",
        "at the top", "right here", "over here",
    }
    return low in triggers or (re.search(r"\bhere\b", low) and len(a.split()) < 7)

def _is_next_topic_question(q: str) -> bool:
    low = q.lower()
    patterns = [
        r"what will be (discussed|covered|shown|taught|done|presented) (next|in the next)",
        r"what (is|are) (suggested|mentioned|planned) to be discussed next",
        r"what will.+cover.+next",
        r"what activity.+discussed next",
        r"what does.+intend to do before diving",
    ]
    return any(re.search(p, low) for p in patterns)

def should_delete(q: str, a: str) -> tuple[bool, str]:
    word_count = len(a.split())
    if _is_bare_number(a):
        return True, "bare number"
    if _is_date(a):
        return True, "date"
    if word_count == 1:
        return True, "single word"
    if _is_type_or_identifier(a):
        return True, "type/identifier"
    if _is_vague_here(a):
        return True, "vague spatial reference"
    if _is_next_topic_question(q):
        return True, "next-topic question"
    if word_count < 10:
        return True, f"short answer ({word_count} words)"
    return False, ""

# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------
def _ollama(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()

def expand_answer(question: str, short_answer: str) -> str:
    """Ask Ollama to rewrite a short answer as a complete explanatory sentence."""
    prompt = (
        "Rewrite the answer below into a single complete sentence of at least 15 words. "
        "Keep exactly the same factual content — do not add information that is not in the short answer. "
        "Output only the rewritten answer, nothing else.\n\n"
        f"Question: {question}\n"
        f"Short answer: {short_answer}\n"
        "Rewritten answer:"
    )
    expanded = _ollama(prompt)
    # strip leading labels the model sometimes emits ("Answer: …")
    expanded = re.sub(r"^(answer\s*:?\s*)", "", expanded, flags=re.IGNORECASE).strip()
    return expanded if len(expanded.split()) >= MIN_WORDS else short_answer

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    candidates = [
        json.loads(line)
        for line in SRC.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(candidates)} candidates from {SRC.name}\n")

    kept:    list[dict] = []
    dropped: list[tuple[str, str]] = []   # (question_snippet, reason)

    for row in candidates:
        q = row["question"]
        a = row["answer"]
        delete, reason = should_delete(q, a)
        if delete:
            dropped.append((q[:70], reason))
            print(f"  DROP [{reason:30s}]  {q[:65]}…")
            continue

        # Expand short-but-passing answers
        if len(a.split()) < MIN_WORDS:
            print(f"  EXPAND ({len(a.split())}w → {MIN_WORDS}w+)  {q[:60]}…")
            a = expand_answer(q, a)

        kept.append({
            "question":         q,
            "expected_answer":  a,
            "source":           row["source"],
            "topic":            row["topic"],
            "difficulty":       row["difficulty"],
            "expected_action":  "answer",
        })

    print(f"\n  Kept {len(kept)} / {len(candidates)}  ({len(dropped)} dropped)\n")

    # Append abstention questions
    for q in ABSTAIN_QUESTIONS:
        kept.append({
            "question":        q,
            "expected_answer": "",
            "source":          "out-of-scope",
            "topic":           "abstain",
            "difficulty":      "n/a",
            "expected_action": "abstain",
        })

    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")

    print(f"Saved {len(kept)} entries → {DST}\n")

    # ── Summary tables ────────────────────────────────────────────────────
    def tally(key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in kept:
            v = r[key]
            counts[v] = counts.get(v, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    by_action     = tally("expected_action")
    by_difficulty = tally("difficulty")
    by_topic      = tally("topic")

    W = 32
    print(f"{'─'*45}")
    print(f"  {'EXPECTED ACTION':<{W}} {'N':>4}")
    print(f"{'─'*45}")
    for k, n in by_action.items():
        print(f"  {k:<{W}} {n:>4}")

    print(f"\n{'─'*45}")
    print(f"  {'DIFFICULTY':<{W}} {'N':>4}")
    print(f"{'─'*45}")
    for k, n in by_difficulty.items():
        print(f"  {k:<{W}} {n:>4}")

    print(f"\n{'─'*45}")
    print(f"  {'TOPIC':<{W}} {'N':>4}")
    print(f"{'─'*45}")
    for k, n in by_topic.items():
        print(f"  {k:<{W}} {n:>4}")

    print(f"{'─'*45}")
    print(f"\nDone. gold_qa.jsonl has {len(kept)} entries "
          f"({by_action.get('answer', 0)} answerable + "
          f"{by_action.get('abstain', 0)} abstain).")


if __name__ == "__main__":
    main()
