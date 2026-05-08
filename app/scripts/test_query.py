"""Run two queries through the full CALAS graph against the lectureOS_genai collection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb

from src.retrieval.index import _chroma_path
from src.agents.graph import _chunks_from_collection, build_graph

COLLECTION_NAME = "lectureOS_genai"

QUERIES = [
    "What is the purpose of query and key vectors in self-attention?",
    "How does vLLM implement PagedAttention?",
]


def main() -> None:
    client = chromadb.PersistentClient(path=_chroma_path())
    collection = client.get_collection(COLLECTION_NAME)

    all_chunks = _chunks_from_collection(collection)
    print(f"Loaded {len(all_chunks)} chunks from '{COLLECTION_NAME}'")

    graph = build_graph(collection=collection, all_chunks=all_chunks)

    for query in QUERIES:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        result = graph.invoke(
            {
                "query": query,
                "chunks": [],
                "answer": "",
                "confidence": 0.0,
                "action": "answer",
                "mode": "beginner",
            }
        )
        print(f"  action:     {result['action']}")
        print(f"  confidence: {result['confidence']}")
        answer_preview = result.get("answer", "")[:200]
        print(f"  answer:     {answer_preview!r}")


if __name__ == "__main__":
    main()
