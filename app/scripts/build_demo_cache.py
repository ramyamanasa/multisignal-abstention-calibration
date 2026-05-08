"""Build data/processed/lecture_demo_cache.json for the lectureOS demo."""

import sys, os, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "abstention"))

from pipeline import run_pipeline

THRESHOLD = 0.65

QUESTIONS = [
    ("demo_001",
     "Training which model on which dataset using which GPU took 6 days on a single GPU?",
     "ABSTAIN",
     "AlexNet on the Places dataset (2.5M images) using a single K40 GPU"),
    ("demo_002",
     "What were the exact training times in hours for ResNet-50 on ImageNet comparing 8x Tesla V100 vs 8x Tesla P100?",
     "ABSTAIN",
     "5.1 hours (V100) vs 15.5 hours (P100)"),
    ("demo_003",
     "What inference throughput multiplier does the Tesla V100 achieve over a CPU server for ResNet-50 inference?",
     "ABSTAIN",
     "47X higher throughput"),
    ("demo_004",
     "What is the NVLink interconnect bandwidth cited in the lecture in GB/s?",
     "ABSTAIN",
     "300 GB/s"),
    ("demo_005",
     "How many tokens were LLaMA-33B and LLaMA-65B trained on and what batch size was used for all LLaMA models?",
     "ABSTAIN",
     "1.4T tokens, batch size 4M tokens"),
    ("demo_006",
     "In SWE-bench, how many software engineering problems does it contain, across how many Python repositories, and what percentage could Claude 2 resolve?",
     "ABSTAIN",
     "2,294 problems, 12 repositories, 1.96% resolution rate"),
    ("demo_007",
     "What is Gemini 1.5 Pro's context window as stated in the lecture?",
     "ABSTAIN",
     "2 million tokens"),
    ("demo_008",
     "What is the attention mechanism and what problem does it solve in Seq2Seq models?",
     "ANSWER",
     None),
    ("demo_009",
     "What are query, key, and value vectors in the transformer attention mechanism?",
     "ANSWER",
     None),
    ("demo_010",
     "What is the difference between zero-shot and few-shot prompting?",
     "ANSWER",
     None),
    ("demo_011",
     "What are the two training objectives used in BERT pre-training?",
     "ANSWER",
     None),
    ("demo_012",
     "What is Chain-of-Thought prompting and why does it improve reasoning?",
     "ANSWER",
     None),
    ("demo_013",
     "What is the bias-variance tradeoff in machine learning?",
     "ANSWER",
     None),
    ("demo_014",
     "What is the Mixture of Experts architecture and what is its main advantage?",
     "ANSWER",
     None),
    ("demo_015",
     "Why is position encoding necessary in transformer architectures?",
     "ANSWER",
     None),
]


def safe_signal(v):
    if v is None:
        return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    except (TypeError, ValueError):
        return None


def run_with_retry(question, threshold, max_retries=4):
    for attempt in range(max_retries):
        try:
            return run_pipeline(question, threshold=threshold)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower():
                wait = 60 * (attempt + 1)
                print(f"    Rate limit — waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries")


def main():
    results = []
    print(f"\n{'ID':<10} {'Decision':<9} {'P(halluc)':<11} {'Match':<6}  Question")
    print("-" * 92)

    for qid, question, expected, correct in QUESTIONS:
        r = run_with_retry(question, THRESHOLD)

        all_sigs = {
            **r.get("entropy_signals", {}),
            **r.get("consistency_signals", {}),
            **r.get("disagreement_signals", {}),
        }
        signals = {
            k: safe_signal(all_sigs.get(k))
            for k in ["mean_entropy", "max_entropy", "entity_entropy",
                      "semantic_inconsistency", "cross_model_disagreement"]
        }

        decision     = r["decision"].upper()
        hal_prob     = r["hallucination_probability"]
        correct_flag = decision == expected

        results.append({
            "id":                        qid,
            "question":                  question,
            "expected_decision":         expected,
            "correct_answer_from_slide": correct,
            "decision":                  decision,
            "hallucination_prob":        hal_prob,
            "groq_answer":               r["answer"],
            "signals":                   signals,
            "correct":                   correct_flag,
        })

        sym = "✓" if correct_flag else "✗"
        print(f"{qid:<10} {decision:<9} {hal_prob:<11.4f} {sym:<6}  {question[:57]}")
        sys.stdout.flush()

        time.sleep(6)

    os.makedirs("data/processed", exist_ok=True)
    out = "data/processed/lecture_demo_cache.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    n_correct = sum(r["correct"] for r in results)
    print(f"\n{'='*92}")
    print(f"Saved {len(results)} entries → {out}    Score: {n_correct}/{len(results)} correct\n")

    print("ABSTAIN questions (expected to abstain):")
    for r in results:
        if r["expected_decision"] == "ABSTAIN":
            sym = "✓" if r["correct"] else "✗"
            print(f"  {sym} {r['id']}  {r['decision']:<7}  P={r['hallucination_prob']:.4f}  "
                  f"{r['question'][:58]}")
    print()
    print("ANSWER questions (expected to answer):")
    for r in results:
        if r["expected_decision"] == "ANSWER":
            sym = "✓" if r["correct"] else "✗"
            print(f"  {sym} {r['id']}  {r['decision']:<7}  P={r['hallucination_prob']:.4f}  "
                  f"{r['question'][:58]}")


if __name__ == "__main__":
    main()
