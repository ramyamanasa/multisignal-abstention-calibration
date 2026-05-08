"""Fetch a web page, strip HTML, chunk text, and save to JSONL."""

import json
import re
import sys
from pathlib import Path

import httpx
from bs4 import BeautifulSoup


CHUNK_SIZE = 400
OVERLAP = 80


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def fetch_text(url: str) -> str:
    """Fetch a URL and return clean plain text with HTML stripped.

    Args:
        url: The page URL to fetch.

    Returns:
        Plain text with whitespace normalised.

    Raises:
        ValueError: If the response body is too short to contain real content.
    """
    resp = httpx.get(url, headers=_HEADERS, follow_redirects=True, timeout=30)
    resp.raise_for_status()
    if len(resp.text) < 1000:
        raise ValueError(
            f"Response too short ({len(resp.text)} chars) — page may require JS rendering. "
            f"Try the canonical URL directly. Body: {resp.text[:200]!r}"
        )
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Split text into overlapping character windows.

    Args:
        text: Plain text to chunk.
        chunk_size: Target character length of each chunk.
        overlap: Number of characters shared between consecutive chunks.

    Returns:
        List of text chunk strings.
    """
    step = chunk_size - overlap
    return [text[i : i + chunk_size] for i in range(0, len(text), step) if text[i : i + chunk_size].strip()]


def main(url: str, source: str, out_path: Path) -> int:
    """Fetch, chunk, and save a single URL. Returns number of chunks saved."""
    print(f"Fetching {url} ...")
    try:
        text = fetch_text(url)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return -1

    print(f"  Extracted {len(text):,} characters of clean text")
    chunks = chunk_text(text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            record = {
                "text": chunk,
                "chunk_index": i,
                "source": source,
                "url": url,
            }
            f.write(json.dumps(record) + "\n")

    print(f"  Saved {len(chunks)} chunks → {out_path}")
    return len(chunks)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest_url.py <url> [--source <name>] [--out <path>]")
        sys.exit(1)

    url = sys.argv[1]

    source = "web"
    if "--source" in sys.argv:
        source = sys.argv[sys.argv.index("--source") + 1]

    out_dir = Path(__file__).parent.parent / "data" / "raw" / "transcripts"
    out_path = out_dir / f"{source}.jsonl"
    if "--out" in sys.argv:
        out_path = Path(sys.argv[sys.argv.index("--out") + 1])

    result = main(url, source, out_path)
    if result < 0:
        sys.exit(1)
