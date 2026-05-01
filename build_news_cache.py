#!/usr/bin/env python3
"""
Build news fact-checker demo cache.
Runs 45 hand-written questions through the full pipeline.
Output: data/processed/news_demo_cache.json
"""

import json
import math
import os
import sys
import time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

# fusion.py and generation.py use relative paths (e.g. ../models/)
# so CWD must be src/ at import time.
os.chdir(SRC_DIR)

import pipeline as _pipeline_mod
from pipeline import run_pipeline  # noqa: E402 — models load here

# sklearn 1.6→1.8 compat: SimpleImputer renamed _fit_dtype → _fill_dtype in transform().
# Patch in-memory so the loaded pkl works without overwriting meta_clf.pkl.
for _cal in _pipeline_mod.clf.calibrated_classifiers_:
    _imp = _cal.estimator.named_steps["imputer"]
    if not hasattr(_imp, "_fill_dtype") and hasattr(_imp, "_fit_dtype"):
        _imp._fill_dtype = _imp._fit_dtype

# ── Question bank ──────────────────────────────────────────────────────────────

QUESTIONS = [
    # ── Breaking News: ANSWER (5) ──────────────────────────────────────────────
    {
        "id": "news_001",
        "domain": "Breaking News",
        "question": "Who is the Secretary-General of the United Nations?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "news_002",
        "domain": "Breaking News",
        "question": "Which country won the 2022 FIFA World Cup?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "news_003",
        "domain": "Breaking News",
        "question": "Who is the President of France?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "news_004",
        "domain": "Breaking News",
        "question": "Which country invaded Ukraine in February 2022?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "news_005",
        "domain": "Breaking News",
        "question": "Who is the founder and CEO of Tesla?",
        "expected_decision": "ANSWER",
    },
    # ── Breaking News: ABSTAIN (5) ─────────────────────────────────────────────
    {
        "id": "news_006",
        "domain": "Breaking News",
        "question": "What was the exact unemployment rate in the United States in February 2024 according to the Bureau of Labor Statistics?",
        "expected_decision": "ABSTAIN",
    },
    {
        "id": "news_007",
        "domain": "Breaking News",
        "question": "What was China's official GDP growth rate in the third quarter of 2023?",
        "expected_decision": "ABSTAIN",
    },
    {
        "id": "news_008",
        "domain": "Breaking News",
        "question": "According to UNHCR, how many people were internally displaced in Sudan as of December 2023?",
        "expected_decision": "ABSTAIN",
    },
    {
        "id": "news_009",
        "domain": "Breaking News",
        "question": "What exact vote share percentage did Marine Le Pen receive in the first round of the 2022 French presidential election?",
        "expected_decision": "ABSTAIN",
    },
    {
        "id": "news_010",
        "domain": "Breaking News",
        "question": "What was Turkey's official annual inflation rate for January 2024 as reported by TurkStat?",
        "expected_decision": "ABSTAIN",
    },
    # ── Breaking News: BORDERLINE (5) ─────────────────────────────────────────
    {
        "id": "news_011",
        "domain": "Breaking News",
        "question": "Who replaced Boris Johnson as Prime Minister of the United Kingdom?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "news_012",
        "domain": "Breaking News",
        "question": "What was the primary outcome agreed upon at the COP28 climate summit in Dubai in 2023?",
        "expected_decision": "ABSTAIN",
    },
    {
        "id": "news_013",
        "domain": "Breaking News",
        "question": "Who became Prime Minister of Italy following the 2022 general election?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "news_014",
        "domain": "Breaking News",
        "question": "What specific AI safety commitments did major tech companies sign at the UK AI Safety Summit at Bletchley Park in November 2023?",
        "expected_decision": "ABSTAIN",
    },
    {
        "id": "news_015",
        "domain": "Breaking News",
        "question": "Which country hosted the 2023 ICC Men's Cricket World Cup?",
        "expected_decision": "ANSWER",
    },

    # ── Science: ANSWER (5) ───────────────────────────────────────────────────
    {
        "id": "sci_001",
        "domain": "Science",
        "question": "What is the speed of light in a vacuum in meters per second?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "sci_002",
        "domain": "Science",
        "question": "What is the chemical formula for water?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "sci_003",
        "domain": "Science",
        "question": "How many bones are in the adult human body?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "sci_004",
        "domain": "Science",
        "question": "What is the atomic number of carbon?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "sci_005",
        "domain": "Science",
        "question": "What is the molecular structure of DNA called?",
        "expected_decision": "ANSWER",
    },
    # ── Science: ABSTAIN (5) ─────────────────────────────────────────────────
    {
        "id": "sci_006",
        "domain": "Science",
        "question": "What was the exact p-value reported in the 2023 New England Journal of Medicine phase 3 trial of tirzepatide for obesity treatment?",
        "expected_decision": "ABSTAIN",
    },
    {
        "id": "sci_007",
        "domain": "Science",
        "question": "According to the 2023 IPCC synthesis report, what is the median projected global temperature rise by 2100 under the SSP2-4.5 scenario in degrees Celsius?",
        "expected_decision": "ABSTAIN",
    },
    {
        "id": "sci_008",
        "domain": "Science",
        "question": "What exact power conversion efficiency percentage did the world record perovskite-silicon tandem solar cell achieve in 2023?",
        "expected_decision": "ABSTAIN",
    },
    {
        "id": "sci_009",
        "domain": "Science",
        "question": "According to the 2022 Global Burden of Disease study, what was the exact number of deaths globally attributed to ambient air pollution?",
        "expected_decision": "ABSTAIN",
    },
    {
        "id": "sci_010",
        "domain": "Science",
        "question": "What is the exact mass of the Higgs boson in GeV per c squared as measured in the latest combined ATLAS and CMS analysis at CERN?",
        "expected_decision": "ABSTAIN",
    },
    # ── Science: BORDERLINE (5) ──────────────────────────────────────────────
    {
        "id": "sci_011",
        "domain": "Science",
        "question": "What is the approximate diameter of a hydrogen atom in angstroms?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "sci_012",
        "domain": "Science",
        "question": "In what year was the CRISPR-Cas9 gene editing technique first described in a landmark Science paper by Jennifer Doudna and Emmanuelle Charpentier?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "sci_013",
        "domain": "Science",
        "question": "Approximately how many neurons are estimated to be in the adult human brain?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "sci_014",
        "domain": "Science",
        "question": "What percentage of the human genome is currently considered to be protein-coding according to the scientific consensus?",
        "expected_decision": "ABSTAIN",
    },
    {
        "id": "sci_015",
        "domain": "Science",
        "question": "What is the half-life of Carbon-14 as used in radiocarbon dating?",
        "expected_decision": "ANSWER",
    },

    # ── History: ANSWER (5) ───────────────────────────────────────────────────
    {
        "id": "hist_001",
        "domain": "History",
        "question": "In what year did World War II end?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "hist_002",
        "domain": "History",
        "question": "Who was the first President of the United States?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "hist_003",
        "domain": "History",
        "question": "In what year did humans first land on the Moon?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "hist_004",
        "domain": "History",
        "question": "In what year did the Berlin Wall fall?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "hist_005",
        "domain": "History",
        "question": "Who was the ancient Greek philosopher and tutor of Alexander the Great?",
        "expected_decision": "ANSWER",
    },
    # ── History: ABSTAIN (5) ─────────────────────────────────────────────────
    {
        "id": "hist_006",
        "domain": "History",
        "question": "What was the exact number of Union Army soldiers killed in action at the Battle of Gettysburg in July 1863?",
        "expected_decision": "ABSTAIN",
    },
    {
        "id": "hist_007",
        "domain": "History",
        "question": "What was the exact total tonnage of bombs dropped on Dresden by Allied forces during the February 1945 raids?",
        "expected_decision": "ABSTAIN",
    },
    {
        "id": "hist_008",
        "domain": "History",
        "question": "According to the most widely cited academic estimate, how many people died in the Holodomor famine in Ukraine between 1932 and 1933?",
        "expected_decision": "ABSTAIN",
    },
    {
        "id": "hist_009",
        "domain": "History",
        "question": "What was the exact purchase price in US dollars that the United States paid for the Louisiana Purchase in 1803?",
        "expected_decision": "ABSTAIN",
    },
    {
        "id": "hist_010",
        "domain": "History",
        "question": "How many ships and soldiers exactly comprised the Spanish Armada that sailed against England in 1588?",
        "expected_decision": "ABSTAIN",
    },
    # ── History: BORDERLINE (5) ──────────────────────────────────────────────
    {
        "id": "hist_011",
        "domain": "History",
        "question": "In what year did the French Revolution begin?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "hist_012",
        "domain": "History",
        "question": "Who was the Supreme Allied Commander in Europe who planned and oversaw the D-Day landings at Normandy?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "hist_013",
        "domain": "History",
        "question": "In what year did the Western Roman Empire officially fall?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "hist_014",
        "domain": "History",
        "question": "What was the name of the ship that carried the Pilgrims to Plymouth, Massachusetts in 1620?",
        "expected_decision": "ANSWER",
    },
    {
        "id": "hist_015",
        "domain": "History",
        "question": "Who was the Byzantine Emperor when Constantinople fell to the Ottoman Turks in 1453?",
        "expected_decision": "ABSTAIN",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_round(val, decimals=4):
    """Round float; return None for nan/inf so JSON serializes cleanly."""
    if val is None:
        return None
    try:
        if math.isnan(val) or math.isinf(val):
            return None
    except (TypeError, ValueError):
        return None
    return round(float(val), decimals)


def _run_with_retry(question, threshold, max_retries=6):
    """Call run_pipeline with exponential back-off on Groq 429 rate limits."""
    delay = 15
    for attempt in range(1, max_retries + 1):
        try:
            return run_pipeline(question, threshold=threshold)
        except Exception as exc:
            msg = str(exc)
            is_rate_limit = "429" in msg or "rate_limit" in msg.lower() or "Rate limit" in msg
            if is_rate_limit and attempt < max_retries:
                # Parse suggested wait time from error if present (e.g. "try again in 30s")
                import re
                m = re.search(r"try again in (\d+(?:\.\d+)?)s", msg)
                wait = float(m.group(1)) + 2 if m else delay
                print(f"         rate-limited (attempt {attempt}/{max_retries}) — waiting {wait:.0f}s...")
                time.sleep(wait)
                delay = min(delay * 2, 120)
            else:
                raise


# ── Main ──────────────────────────────────────────────────────────────────────

# Each question makes ~6 Groq API calls (5 samples + 1 model2 answer).
# Groq free tier is 30 RPM, so safe pace is ≥12 s between questions.
# With ood_eval.py also running, use 20 s to avoid conflicts.
INTER_QUESTION_DELAY = 20  # seconds

def main():
    output_path = os.path.join(PROJECT_ROOT, "data", "processed", "news_demo_cache.json")
    threshold = 0.5
    results = []

    total = len(QUESTIONS)
    print(f"\nStarting pipeline for {total} questions  (threshold={threshold})")
    print(f"Inter-question delay: {INTER_QUESTION_DELAY}s  (estimated runtime: ~{total*INTER_QUESTION_DELAY//60} min)")
    print(f"Output → {output_path}\n")

    for i, q in enumerate(QUESTIONS, 1):
        print(f"[{i:02d}/{total}] {q['id']}  {q['question'][:70]}...")
        try:
            r = _run_with_retry(q["question"], threshold=threshold)

            entry = {
                "id": q["id"],
                "domain": q["domain"],
                "question": q["question"],
                "expected_decision": q["expected_decision"],
                "hallucination_prob": _safe_round(r["hallucination_probability"]),
                "decision": r["decision"].upper(),
                "signals": {
                    "mean_entropy":             _safe_round(r["entropy_signals"]["mean_entropy"]),
                    "max_entropy":              _safe_round(r["entropy_signals"]["max_entropy"]),
                    "entity_entropy":           _safe_round(r["entropy_signals"]["entity_entropy"]),
                    "semantic_inconsistency":   _safe_round(r["consistency_signals"]["semantic_inconsistency"]),
                    "cross_model_disagreement": _safe_round(r["disagreement_signals"]["cross_model_disagreement"]),
                },
                "groq_answer": r["answer"],
                "domain_tag": q["domain"],
            }
            results.append(entry)

            match = "✓" if entry["decision"] == entry["expected_decision"] else "✗"
            print(
                f"         P(halluc)={entry['hallucination_prob']:.3f}  "
                f"decision={entry['decision']}  "
                f"expected={entry['expected_decision']}  {match}"
            )
        except Exception as exc:
            print(f"         ERROR — skipping: {exc}")

        # Pace requests to stay within Groq RPM limit
        if i < total:
            time.sleep(INTER_QUESTION_DELAY)

    cache = {
        "metadata": {
            "created": datetime.now().strftime("%Y-%m-%d"),
            "n_questions": len(results),
            "model": "meta_clf.pkl",
            "threshold": threshold,
        },
        "questions": results,
    }

    with open(output_path, "w") as f:
        json.dump(cache, f, indent=2)

    answered  = sum(1 for e in results if e["decision"] == "ANSWER")
    abstained = sum(1 for e in results if e["decision"] == "ABSTAIN")
    correct   = sum(1 for e in results if e["decision"] == e["expected_decision"])

    print(f"\n{'='*60}")
    print(f"Saved {len(results)}/{total} questions to {output_path}")
    print(f"ANSWER: {answered}  ABSTAIN: {abstained}")
    print(f"Expected match: {correct}/{len(results)} ({100*correct/max(len(results),1):.1f}%)")


if __name__ == "__main__":
    main()
