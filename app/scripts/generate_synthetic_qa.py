"""Generate synthetic Q&A pairs from transcript chunks via Ollama.

Reads all JSONL files under data/raw/transcripts/, skips chunks under
100 characters, prompts the local LLM for 2 question-answer pairs per
chunk, and writes results to data/train/synthetic_qa.jsonl.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).parent.parent
TRANSCRIPTS_DIR = REPO_ROOT / "data" / "raw" / "transcripts"
OUTPUT_FILE = REPO_ROOT / "data" / "train" / "synthetic_qa.jsonl"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_MODEL = "qwen2.5:7b-instruct"
MIN_CHUNK_LEN = 100


def _ollama_chat(prompt: str) -> str:
    """Send a prompt to Ollama and return the response text.

    Args:
        prompt: Full prompt string.

    Returns:
        Model response as a plain string.

    Raises:
        requests.HTTPError: On non-2xx Ollama response.
    """
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": _MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def _extract_pairs(raw: str) -> list[dict]:
    """Parse Q&A pairs from a (possibly markdown-wrapped) JSON string.

    Extracts the first JSON array found in ``raw``, then filters for
    dicts that contain both ``question`` and ``answer`` keys.

    Args:
        raw: Raw LLM output, may include markdown fences or preamble.

    Returns:
        List of dicts with ``question`` and ``answer`` keys.
        Empty list if parsing fails or no valid pairs found.
    """
    # Strip markdown fences then grab first [...] block.
    cleaned = re.sub(r"```(?:json)?|```", "", raw)
    match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
    if not match:
        return []
    try:
        candidates = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    return [
        p for p in candidates
        if isinstance(p, dict) and "question" in p and "answer" in p
    ]


def _load_chunks(transcripts_dir: Path) -> list[dict]:
    """Load all chunks from JSONL files in transcripts_dir.

    Args:
        transcripts_dir: Directory containing ``*.jsonl`` transcript files.

    Returns:
        List of chunk dicts, each with at least a ``text`` key.
    """
    chunks: list[dict] = []
    for path in sorted(transcripts_dir.glob("*.jsonl")):
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    return chunks


def generate(transcripts_dir: Path, output_file: Path) -> int:
    """Run the full generation pipeline and return the total pairs written.

    Args:
        transcripts_dir: Path to the directory of transcript JSONL files.
        output_file: Destination JSONL file for synthetic Q&A pairs.

    Returns:
        Total number of Q&A pairs written.
    """
    chunks = _load_chunks(transcripts_dir)
    eligible = [c for c in chunks if len(c.get("text", "")) >= MIN_CHUNK_LEN]

    print(f"Loaded {len(chunks)} chunks, {len(eligible)} above {MIN_CHUNK_LEN} chars.")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with output_file.open("w") as out:
        for i, chunk in enumerate(eligible, start=1):
            text = chunk["text"]
            source = chunk.get("source", "unknown")
            start = chunk.get("start")

            prompt = (
                "Generate 2 question-answer pairs a student might ask about this "
                "content. Output as a JSON array with no extra text:\n"
                '[{"question": "...", "answer": "..."}]\n\n'
                f"Content:\n{text}"
            )

            try:
                raw = _ollama_chat(prompt)
                pairs = _extract_pairs(raw)
            except Exception as exc:
                print(f"  [{i}/{len(eligible)}] error — {exc}", file=sys.stderr)
                continue

            for pair in pairs:
                record = {
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "source": source,
                    "start": start,
                }
                out.write(json.dumps(record) + "\n")
                total += 1

            print(f"  [{i}/{len(eligible)}] {len(pairs)} pairs — {source}")

    return total


def main() -> None:
    total = generate(TRANSCRIPTS_DIR, OUTPUT_FILE)
    print(f"\nDone. {total} pairs written to {OUTPUT_FILE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
