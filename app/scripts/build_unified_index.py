"""Build a single ChromaDB collection from all JSONL files in data/raw/transcripts/."""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.index import build_index

COLLECTION_NAME = "lectureOS_genai"
TRANSCRIPTS_DIR = Path(__file__).parent.parent / "data" / "raw" / "transcripts"


def load_all_jsonl(directory: Path) -> tuple[list[dict], dict[str, int]]:
    """Load every JSONL file in directory into a flat chunk list.

    Args:
        directory: Path containing .jsonl files.

    Returns:
        Tuple of (all_chunks, counts_by_file) where counts_by_file maps
        filename to the number of records loaded from it.
    """
    all_chunks: list[dict] = []
    counts: dict[str, int] = {}
    for path in sorted(directory.glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        # Ensure every record has the keys build_index expects.
        for r in records:
            r.setdefault("page", r.get("start", r.get("chunk_index", 0)))
        all_chunks.extend(records)
        counts[path.name] = len(records)
    return all_chunks, counts


def main() -> None:
    print(f"Scanning {TRANSCRIPTS_DIR} ...")
    chunks, counts = load_all_jsonl(TRANSCRIPTS_DIR)

    if not chunks:
        print("No JSONL files found. Exiting.")
        sys.exit(1)

    print(f"\nBreakdown by source file:")
    for fname, n in counts.items():
        print(f"  {fname:<35} {n:>5} chunks")
    print(f"  {'TOTAL':<35} {len(chunks):>5} chunks\n")

    print(f"Building ChromaDB collection '{COLLECTION_NAME}' ...")
    collection = build_index(chunks, COLLECTION_NAME)

    print(f"\n{'='*50}")
    print(f"Collection : {collection.name}")
    print(f"Vectors    : {collection.count()}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
