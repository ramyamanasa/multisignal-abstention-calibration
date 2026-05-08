"""Evaluate the full CALAS pipeline against data/eval/gold_qa.jsonl.

Prints per-question results and a summary table covering:
  - Accuracy on answerable questions (expected_action=answer)
  - Abstention rate on out-of-scope questions (expected_action=abstain)
  - Average corpus_confidence by expected_action
  - Wrong answers vs incorrect abstentions
"""

import json
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.index import build_index
from src.agents.graph import build_graph

TRANSCRIPTS  = Path("data/raw/transcripts")
GOLD_PATH    = Path("data/eval/gold_qa.jsonl")
COLLECTION   = "lectureOS_genai"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_chunks(directory: Path) -> list[dict]:
    chunks: list[dict] = []
    for p in sorted(directory.glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r.setdefault("page", r.get("start", r.get("chunk_index", 0)))
                chunks.append(r)
    return chunks


def load_gold(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def keyword_correctness(answer: str, reference: str) -> str:
    """Heuristic correctness label against a reference answer.

    Returns 'yes', 'partial', or 'no'.
    """
    if not answer or not reference:
        return "n/a"
    ref_words = {w.lower() for w in re.findall(r"[a-z]+", reference) if len(w) > 4}
    ans_words = {w.lower() for w in re.findall(r"[a-z]+", answer)}
    overlap = ref_words & ans_words
    ratio = len(overlap) / len(ref_words) if ref_words else 0.0
    if ratio >= 0.45:
        return "yes"
    if ratio >= 0.20:
        return "partial"
    return "no"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading corpus and building index...")
    all_chunks = load_chunks(TRANSCRIPTS)
    collection = build_index(all_chunks, COLLECTION)
    graph = build_graph(collection=collection, all_chunks=all_chunks)
    print(f"  {collection.count()} vectors ready\n")

    gold = load_gold(GOLD_PATH)
    print(f"Evaluating {len(gold)} gold questions from {GOLD_PATH}\n")

    records = []

    for i, item in enumerate(gold):
        qid       = item.get("id", f"q{i+1:02d}")
        question  = item["question"]
        expected  = item["expected_action"]
        reference = item.get("expected_answer") or ""

        print(f"  [{i+1:>2}/{len(gold)}] {question[:70]}…")

        result = graph.invoke({
            "query":      question,
            "chunks":     [],
            "answer":     "",
            "confidence": 0.0,
            "action":     "answer",
            "mode":       "beginner",
        })

        action     = result.get("action", "answer")
        confidence = float(result.get("confidence", 0.0))
        answer     = result.get("answer", "")

        action_match = action == expected

        if expected == "answer":
            correctness = keyword_correctness(answer, reference) if action == "answer" else "abstained"
        else:
            # Out-of-scope: correctly abstaining is good, answering is bad
            correctness = "correct_abstain" if action == "abstain" else "wrong_answer"

        records.append({
            "id":           qid,
            "question":     question,
            "expected":     expected,
            "action":       action,
            "confidence":   confidence,
            "correctness":  correctness,
            "action_match": action_match,
            "answer":       answer,
            "reference":    reference,
        })

    print(f"\nCollected {len(records)} records.\n")
    if not records:
        print("No records to display — check that the pipeline is running correctly.")
        return

    # ── Per-question table ────────────────────────────────────────────────────
    q_w   = 52
    exp_w = 8
    act_w = 8
    con_w = 10
    cor_w = 16
    mat_w = 7

    header = (
        f"{'ID':<5} {'Question':<{q_w}} {'Exp':<{exp_w}} {'Act':<{act_w}}"
        f" {'Confidence':>{con_w}} {'Correctness':<{cor_w}} {'Match':<{mat_w}}"
    )
    div = "-" * len(header)
    print(div)
    print(header)
    print(div)

    for r in records:
        q_short = textwrap.shorten(r["question"], width=q_w, placeholder="…")
        match_str = "OK" if r["action_match"] else "MISMATCH"
        print(
            f"{r['id']:<5} {q_short:<{q_w}} {r['expected']:<{exp_w}} {r['action']:<{act_w}}"
            f" {r['confidence']:>{con_w}.4f} {r['correctness']:<{cor_w}} {match_str:<{mat_w}}"
        )

    print(div)

    # ── Answers inline ────────────────────────────────────────────────────────
    print("\n\nANSWERS / ABSTENTIONS")
    for r in records:
        tag = "✓" if r["action_match"] else "✗"
        print(f"\n[{r['id']}] {tag} expected={r['expected']}  actual={r['action']}  conf={r['confidence']:.4f}")
        print(f"Q: {r['question']}")
        if r["answer"]:
            print(textwrap.fill(r["answer"], width=80, initial_indent="A: ", subsequent_indent="   "))
        else:
            print("A: [ABSTAINED]")

    # ── Summary ───────────────────────────────────────────────────────────────
    answerable  = [r for r in records if r["expected"] == "answer"]
    oos         = [r for r in records if r["expected"] == "abstain"]

    # Accuracy: answered (any correctness label that isn't 'abstained') / total answerable
    correctly_answered = [r for r in answerable if r["correctness"] in ("yes", "partial")]
    wrong_answer       = [r for r in answerable if r["correctness"] == "no"]
    incorrect_abstain  = [r for r in answerable if r["correctness"] == "abstained"]

    abstention_rate    = sum(1 for r in oos if r["action"] == "abstain") / len(oos) if oos else 0.0
    false_answers_oos  = [r for r in oos if r["action"] != "abstain"]

    avg_conf_answerable = sum(r["confidence"] for r in answerable) / len(answerable) if answerable else 0.0
    avg_conf_oos        = sum(r["confidence"] for r in oos)        / len(oos)        if oos        else 0.0

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Answerable questions (expected=answer):  {len(answerable)}")
    print(f"    Correct / partial answers:             {len(correctly_answered)}/{len(answerable)}")
    print(f"    Wrong answers (answered but off):      {len(wrong_answer)}")
    print(f"    Incorrectly abstained:                 {len(incorrect_abstain)}")
    print()
    print(f"  Out-of-scope questions (expected=abstain): {len(oos)}")
    print(f"    Abstention rate:                         {abstention_rate:.0%} ({sum(1 for r in oos if r['action']=='abstain')}/{len(oos)})")
    print(f"    False answers on OOS:                    {len(false_answers_oos)}")
    print()
    print(f"  Avg corpus_confidence — answerable: {avg_conf_answerable:.4f}")
    print(f"  Avg corpus_confidence — OOS:        {avg_conf_oos:.4f}")
    print(f"  Separation gap:                     {avg_conf_answerable - avg_conf_oos:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
