"""Fetch a YouTube transcript and save overlapping chunks to JSONL."""

import json
import sys
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi


WINDOW = 5
STEP = 3
BASE_URL = "https://www.youtube.com/watch?v={video_id}&t={seconds}s"


def fetch_chunks(video_id: str) -> list[dict]:
    """Fetch transcript and return overlapping 5-segment windows.

    Args:
        video_id: YouTube video ID string.

    Returns:
        List of dicts with keys: text, start, end, source, timestamp_url.
    """
    api = YouTubeTranscriptApi()
    transcript = list(api.fetch(video_id))
    chunks = []
    for i in range(0, len(transcript), STEP):
        window = transcript[i : i + WINDOW]
        if not window:
            break
        text = " ".join(seg.text.strip() for seg in window)
        start = window[0].start
        end = window[-1].start + window[-1].duration
        chunks.append(
            {
                "text": text,
                "start": round(start, 3),
                "end": round(end, 3),
                "source": f"youtube:{video_id}",
                "timestamp_url": BASE_URL.format(
                    video_id=video_id, seconds=int(start)
                ),
            }
        )
    return chunks


def main(video_id: str) -> int:
    """Fetch and save transcript for one video. Returns chunk count, or -1 on error."""
    out_dir = Path(__file__).parent.parent / "data" / "raw" / "transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_id}.jsonl"

    print(f"Fetching transcript for {video_id}...")
    try:
        chunks = fetch_chunks(video_id)
    except Exception as exc:
        print(f"  ERROR [{video_id}]: {exc}")
        return -1

    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"  Saved {len(chunks)} chunks → {out_path}")
    return len(chunks)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_youtube.py <video_id> [video_id ...]")
        sys.exit(1)

    video_ids = sys.argv[1:]
    results: dict[str, int] = {}
    for vid in video_ids:
        results[vid] = main(vid)

    if len(video_ids) > 1:
        print("\n--- Summary ---")
        for vid, count in results.items():
            status = f"{count} chunks" if count >= 0 else "FAILED"
            print(f"  {vid}: {status}")
