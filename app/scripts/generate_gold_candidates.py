"""Generate gold evaluation candidates from lecture chunks via Ollama.

Strategy
--------
* 15 chunks from kCc8FmEb1nY  (nanoGPT lecture)
* 15 chunks from zduSFxRajkE  (tokenization lecture)
* 12 chunks from LWMzyfvuehA  (CS224N lecture)
*  8 chunks from HF/LoRA docs (hf_lora_config + hf_peft_lora + lora_paper)

YouTube chunks are spread across their full timestamp range (sort by start,
divide into n equal-count buckets, take the middle of each bucket).
Doc chunks have no timestamps so they are spread by index instead.

For each selected chunk the model is asked for 2 JSON question objects.
Near-duplicate questions (Jaccard word overlap ≥ 60 %) are dropped.

Output: data/eval/gold_candidates.jsonl
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRANSCRIPTS     = Path(__file__).parent.parent / "data" / "raw" / "transcripts"
OUT_PATH        = Path(__file__).parent.parent / "data" / "eval" / "gold_candidates.jsonl"
OLLAMA_URL      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL           = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
DEDUP_THRESHOLD = 0.60

# (file stem, n to select, human label)
# "hf_docs" is a virtual key that merges all three HF/LoRA doc files
ALLOCATION: list[tuple[str, int, str]] = [
    ("kCc8FmEb1nY", 15, "nanoGPT"),
    ("zduSFxRajkE", 15, "tokenization"),
    ("LWMzyfvuehA", 12, "CS224N"),
    ("hf_docs",      8, "HF/LoRA docs"),
]

HF_DOC_STEMS = {"hf_lora_config", "hf_peft_lora", "lora_paper"}

QUESTION_PROMPT = """\
Generate 2 specific factual questions that test whether a student understood this excerpt. \
Questions must be answerable ONLY from this exact text. Do not ask vague questions. \
Output JSON only:
[{{"question": "...", "answer": "...", "difficulty": "easy|medium|hard", \
"topic": "attention|tokenization|training|lora|architecture"}}]

Excerpt:
{text}"""

VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_TOPICS       = {"attention", "tokenization", "training", "lora", "architecture"}

# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_file(path: Path) -> list[dict]:
    chunks = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        c = json.loads(line)
        c.setdefault("chunk_id", f"{path.stem}_{i}")
        c.setdefault("start", 0.0)
        c.setdefault("source", path.stem)
        chunks.append(c)
    return chunks


def load_all() -> dict[str, list[dict]]:
    """Return {stem: [chunks]} for every file in TRANSCRIPTS."""
    bank: dict[str, list[dict]] = {}
    for p in sorted(TRANSCRIPTS.glob("*.jsonl")):
        bank[p.stem] = load_file(p)
        print(f"  loaded {len(bank[p.stem]):>4} chunks  ← {p.name}")
    return bank


# ---------------------------------------------------------------------------
# Chunk selection
# ---------------------------------------------------------------------------
def select_spread(chunks: list[dict], n: int) -> list[dict]:
    """Pick n chunks spread evenly across the list (by timestamp or by index)."""
    if len(chunks) <= n:
        return list(chunks)
    sorted_chunks = sorted(chunks, key=lambda c: c.get("start", 0.0))
    group_size    = len(sorted_chunks) / n
    return [sorted_chunks[int((i + 0.5) * group_size)] for i in range(n)]


def build_selection(bank: dict[str, list[dict]]) -> list[dict]:
    selected: list[dict] = []
    for stem, n, label in ALLOCATION:
        if stem == "hf_docs":
            pool = [c for s, chunks in bank.items() if s in HF_DOC_STEMS for c in chunks]
        else:
            pool = bank.get(stem, [])

        if not pool:
            print(f"  WARNING: no chunks found for '{stem}' ({label})")
            continue

        chosen = select_spread(pool, n)
        print(f"  selected {len(chosen):>2} / {len(pool):>4} chunks  [{label}]")
        selected.extend(chosen)
    return selected


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
def ollama_generate(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


# ---------------------------------------------------------------------------
# JSON extraction & validation
# ---------------------------------------------------------------------------
def extract_json_array(raw: str) -> list[dict] | None:
    """Pull the first [...] block out of a model response and parse it."""
    # strip markdown fences
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def validate_item(item: object) -> dict | None:
    if not isinstance(item, dict):
        return None
    q = str(item.get("question", "")).strip()
    a = str(item.get("answer",   "")).strip()
    if not q or not a:
        return None
    difficulty = str(item.get("difficulty", "medium")).lower()
    topic      = str(item.get("topic",      "architecture")).lower()
    if difficulty not in VALID_DIFFICULTIES:
        difficulty = "medium"
    if topic not in VALID_TOPICS:
        topic = "architecture"
    return {"question": q, "answer": a, "difficulty": difficulty, "topic": topic}


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def is_duplicate(question: str, seen: list[str]) -> bool:
    return any(jaccard(question, s) >= DEDUP_THRESHOLD for s in seen)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"\n{'='*55}")
    print("  generate_gold_candidates.py")
    print(f"  model : {MODEL} @ {OLLAMA_URL}")
    print(f"{'='*55}\n")

    # 1 · load
    print("Loading chunks…")
    bank = load_all()
    print()

    # 2 · select
    print("Selecting diverse chunks…")
    selection = build_selection(bank)
    print(f"\n  Total chunks selected: {len(selection)}\n")

    # 3 · generate
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    records:  list[dict] = []
    seen_qs:  list[str]  = []
    skipped_parse = 0
    skipped_dedup = 0

    for idx, chunk in enumerate(selection, 1):
        text     = chunk.get("text", "").strip()
        source   = chunk.get("source", "unknown")
        chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
        start    = chunk.get("start", 0.0)

        if len(text) < 40:          # too short to yield meaningful questions
            continue

        print(f"[{idx:>2}/{len(selection)}] {source}  t={start:.0f}s  …{text[:60].strip()}…")

        prompt = QUESTION_PROMPT.format(text=text)
        try:
            raw = ollama_generate(prompt)
        except requests.RequestException as exc:
            print(f"  ERROR calling Ollama: {exc}")
            continue

        items = extract_json_array(raw)
        if not items:
            print(f"  SKIP — could not parse JSON from response")
            skipped_parse += 1
            continue

        for item in items:
            validated = validate_item(item)
            if not validated:
                continue
            if is_duplicate(validated["question"], seen_qs):
                skipped_dedup += 1
                print(f"  DEDUP — '{validated['question'][:60]}…'")
                continue
            seen_qs.append(validated["question"])
            records.append({
                "source":     source,
                "chunk_id":   chunk_id,
                "start":      start,
                "question":   validated["question"],
                "answer":     validated["answer"],
                "difficulty": validated["difficulty"],
                "topic":      validated["topic"],
            })

    # 4 · save
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nSaved {len(records)} candidates → {OUT_PATH}")
    print(f"  Skipped (parse error) : {skipped_parse}")
    print(f"  Skipped (duplicate)   : {skipped_dedup}")

    # 5 · summary tables
    def tally(key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in records:
            v = r[key]
            counts[v] = counts.get(v, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    by_source     = tally("source")
    by_difficulty = tally("difficulty")
    by_topic      = tally("topic")

    print(f"\n{'─'*40}")
    print(f"  {'SOURCE':<30} {'N':>4}")
    print(f"{'─'*40}")
    for src, n in by_source.items():
        print(f"  {src:<30} {n:>4}")

    print(f"\n{'─'*40}")
    print(f"  {'DIFFICULTY':<30} {'N':>4}")
    print(f"{'─'*40}")
    for diff, n in by_difficulty.items():
        print(f"  {diff:<30} {n:>4}")

    print(f"\n{'─'*40}")
    print(f"  {'TOPIC':<30} {'N':>4}")
    print(f"{'─'*40}")
    for topic, n in by_topic.items():
        print(f"  {topic:<30} {n:>4}")

    print(f"{'─'*40}\n")


if __name__ == "__main__":
    main()
