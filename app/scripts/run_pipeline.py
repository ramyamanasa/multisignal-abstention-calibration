"""Build ChromaDB index from a transcript JSONL and run a question through
the full CALAS pipeline (classify → retrieve → route → generate → verify).

Usage:
    python scripts/run_pipeline.py <video_id> "<question>" [--mode beginner|exam|deep]
"""

import json
import sys
import textwrap
from pathlib import Path

# Ensure project root is on sys.path when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.index import build_index, chunk_documents
from src.agents.graph import build_graph


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _normalize(records: list[dict]) -> list[dict]:
    """Map transcript JSONL records to the shape expected by chunk_documents."""
    return [
        {
            "text": r["text"],
            "source": r.get("source", "unknown"),
            "page": round(r.get("start", 0)),
        }
        for r in records
    ]


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python run_pipeline.py <video_id> '<question>' [--mode beginner|exam|deep]")
        sys.exit(1)

    video_id = sys.argv[1]
    question = sys.argv[2]
    mode = "beginner"
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        mode = sys.argv[idx + 1]

    jsonl_path = Path("data/raw/transcripts") / f"{video_id}.jsonl"
    if not jsonl_path.exists():
        print(f"Transcript not found: {jsonl_path}")
        sys.exit(1)

    collection_name = f"yt_{video_id}"

    # --- Build or reload index ---
    print(f"Loading transcript from {jsonl_path} ...")
    records = load_jsonl(jsonl_path)
    docs = _normalize(records)
    chunks = chunk_documents(docs)
    print(f"  {len(records)} JSONL records → {len(chunks)} text chunks")

    print("Building ChromaDB index ...")
    collection = build_index(chunks, collection_name)
    print(f"  Index '{collection_name}' ready ({collection.count()} vectors)\n")

    graph = build_graph(collection=collection, all_chunks=chunks)

    # --- Run pipeline ---
    print(f"Question : {question}")
    print(f"Mode     : {mode}\n")

    initial_state = {
        "query": question,
        "chunks": [],
        "answer": "",
        "confidence": 0.0,
        "action": "answer",
        "mode": mode,
    }

    result = graph.invoke(initial_state)

    # --- Print retrieved chunks ---
    retrieved = result.get("chunks", [])
    print("=" * 70)
    print(f"RETRIEVED CHUNKS  (action={result['action']}, confidence={result['confidence']})")
    print("=" * 70)
    for i, c in enumerate(retrieved, 1):
        src = c["metadata"].get("source", "?")
        page = c["metadata"].get("page", "?")
        score = c.get("score", 0.0)
        print(f"\n[{i}] source={src}  start≈{page}s  rrf_score={score:.5f}")
        print(textwrap.fill(c["text"], width=80, initial_indent="    ", subsequent_indent="    "))

    # --- Print answer ---
    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    answer = result.get("answer", "")
    if answer:
        print(textwrap.fill(answer, width=80))
    else:
        print(f"[No answer generated — action: {result['action']}]")


if __name__ == "__main__":
    main()
