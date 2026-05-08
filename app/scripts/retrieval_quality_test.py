"""Retrieval quality test against the lectureOS_genai ChromaDB collection."""

import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.index import load_index
from src.retrieval.retrieve import hybrid_retrieve

COLLECTION = "lectureOS_genai"
TRANSCRIPTS_DIR = Path(__file__).parent.parent / "data" / "raw" / "transcripts"
TOP_K = 3

QUESTIONS_IN_SCOPE = [
    "What is the purpose of the query and key vectors in self-attention?",
    "Why do transformers use positional encoding?",
    "How does byte pair encoding split words into tokens?",
    "What is the difference between encoder-only and decoder-only models?",
    "How does LoRA reduce the number of trainable parameters?",
]

QUESTIONS_OUT_OF_SCOPE = [
    "What is the architecture of GPT-4?",
    "How does Gemini handle multimodal inputs?",
    "What did the original BERT paper report on SQuAD?",
    "How does vLLM implement PagedAttention?",
    "What is the Columbia GR5293 final exam format?",
]

# Rough relevance judgements keyed on question index (1-based).
# True  = top chunk is genuinely on-topic for the question.
# False = retrieval miss or off-topic result.
# These are filled in programmatically below after inspecting results.


def load_all_chunks(directory: Path) -> list[dict]:
    chunks: list[dict] = []
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                r.setdefault("page", r.get("start", r.get("chunk_index", 0)))
                chunks.append(r)
    return chunks


def short_source(source: str) -> str:
    mapping = {
        "youtube:kCc8FmEb1nY": "YT:nanoGPT",
        "youtube:zduSFxRajkE": "YT:tokenize",
        "youtube:LWMzyfvuehA": "YT:CS224N",
        "hf_peft_lora":        "HF:peft_lora",
    }
    return mapping.get(source, source[:20])


def judge(q_idx: int, top_chunk_text: str, top_source: str) -> str:
    """Heuristic relevance judgement based on keyword presence."""
    text = top_chunk_text.lower()
    src  = top_source.lower()

    rules = {
        0: (["query", "key", "attention", "vector"], []),
        1: (["position", "encoding", "order", "sequential"], []),
        2: (["byte pair", "bpe", "token", "split", "merge", "subword"], []),
        3: (["encoder", "decoder", "gpt", "bert", "causal", "autoregress"], []),
        4: (["lora", "low-rank", "trainable", "parameter", "adapt"], ["hf:peft_lora", "yt:nanogpt", "yt:tokenize", "yt:cs224n"]),
        5: (["gpt-4", "gpt4", "architecture", "openai"], []),
        6: (["gemini", "multimodal", "image", "google"], []),
        7: (["bert", "squad", "f1", "benchmark"], []),
        8: (["vllm", "paged", "attention", "kv cache"], []),
        9: (["exam", "gr5293", "columbia", "final"], []),
    }

    keywords, preferred_sources = rules.get(q_idx, ([], []))
    hit = any(kw in text for kw in keywords)

    # For out-of-scope (indices 5-9) a hit means bad retrieval (hallucination risk)
    if q_idx >= 5:
        return "No (OOS)" if not hit else "Weak hit"
    return "Yes" if hit else "No"


def main() -> None:
    print(f"Loading collection '{COLLECTION}' ...")
    collection = load_index(COLLECTION)
    print(f"Loading chunks for BM25 ({TRANSCRIPTS_DIR}) ...")
    all_chunks = load_all_chunks(TRANSCRIPTS_DIR)
    print(f"  {collection.count()} vectors | {len(all_chunks)} BM25 docs\n")

    all_questions = QUESTIONS_IN_SCOPE + QUESTIONS_OUT_OF_SCOPE
    rows = []

    for q_idx, question in enumerate(all_questions):
        results = hybrid_retrieve(question, collection, all_chunks, k=TOP_K)
        top = results[0] if results else {"text": "", "metadata": {"source": "none"}, "score": 0.0}
        src     = short_source(top["metadata"].get("source", "?"))
        conf    = getattr(results, "corpus_confidence", 0.0)
        label   = "IN-SCOPE " if q_idx < 5 else "OUT-SCOPE"
        verdict = judge(q_idx, top["text"], src)
        rows.append((label, q_idx + 1, question, src, conf, verdict, results))

    # ── Table ─────────────────────────────────────────────────────────────────
    q_col  = 58
    s_col  = 14
    c_col  = 12   # corpus_confidence column
    v_col  = 10

    header = (
        f"{'#':<3} {'Question':<{q_col}} {'Top source':<{s_col}}"
        f" {'Cos sim (top3)':>{c_col}} {'Relevant?':<{v_col}}"
    )
    div = "-" * len(header)

    print(div)
    print(header)
    print(div)

    in_scores, out_scores = [], []

    for label, num, question, src, score, verdict, results in rows:
        conf = getattr(results, "corpus_confidence", 0.0)
        q_short = textwrap.shorten(question, width=q_col, placeholder="…")
        print(
            f"{num:<3} {q_short:<{q_col}} {src:<{s_col}}"
            f" {conf:>{c_col}.4f} {verdict:<{v_col}}"
        )
        if num <= 5:
            in_scores.append(conf)
        else:
            out_scores.append(conf)

    print(div)

    # ── Per-question chunk detail ──────────────────────────────────────────────
    print("\n\nDETAILED CHUNKS PER QUESTION")
    for label, num, question, src, score, verdict, results in rows:
        tag = "✓ IN-SCOPE" if num <= 5 else "✗ OUT-SCOPE"
        conf = getattr(results, "corpus_confidence", 0.0)
        print(f"\n[{num}] [{tag}] corpus_confidence={conf:.4f}  {question}")
        for rank, chunk in enumerate(results, 1):
            csrc  = short_source(chunk["metadata"].get("source", "?"))
            crrf  = chunk.get("score", 0.0)
            csim  = chunk.get("max_similarity", -1.0)
            ctext = textwrap.shorten(chunk["text"], width=100, placeholder="…")
            print(f"  [{rank}] {csrc:<14} rrf={crrf:.5f}  cos_sim={csim:.4f}  {ctext}")

    # ── Score summary ──────────────────────────────────────────────────────────
    avg_in  = sum(in_scores)  / len(in_scores)
    avg_out = sum(out_scores) / len(out_scores)
    gap     = avg_in - avg_out
    print(f"\n{'='*57}")
    print(f"  Avg corpus_confidence — IN-SCOPE  (Q1–5):  {avg_in:.4f}")
    print(f"  Avg corpus_confidence — OUT-SCOPE (Q6–10): {avg_out:.4f}")
    print(f"  Separation gap:                             {gap:.4f}")
    print(f"{'='*57}")


if __name__ == "__main__":
    main()
