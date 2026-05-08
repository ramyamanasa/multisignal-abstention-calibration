from rank_bm25 import BM25Okapi

from src.retrieval.index import _get_encoder


def hybrid_retrieve(
    query: str,
    collection,
    all_chunks: list[dict],
    k: int = 5,
) -> list[dict]:
    """Retrieve relevant chunks using Reciprocal Rank Fusion over dense + sparse results.

    Combines ChromaDB dense retrieval (BAAI/bge-small-en-v1.5 embeddings) with
    BM25Okapi sparse retrieval, fusing rankings via RRF so neither signal dominates.

    Each returned chunk includes a ``max_similarity`` field (cosine similarity
    of the dense retrieval hit) and the result list carries a
    ``corpus_confidence`` attribute (mean cosine similarity of the top-k chunks)
    that callers can use for abstention decisions.

    Args:
        query: The user's natural-language question.
        collection: A ChromaDB ``Collection`` instance to query for dense results.
        all_chunks: Full list of chunk dicts (each with a ``text`` key) used to
            build the BM25 index.  Must match the documents stored in ``collection``.
        k: Number of final results to return.

    Returns:
        List of up to ``k`` dicts, each with keys:
            - ``text``            (str)   — chunk text.
            - ``metadata``        (dict)  — ``source`` and page metadata.
            - ``score``           (float) — fused RRF score (higher is more relevant).
            - ``max_similarity``  (float) — cosine similarity from dense retrieval
                                            (1 − cosine_distance); -1.0 if not available.
        The list object also exposes a ``corpus_confidence`` attribute equal to
        the mean ``max_similarity`` of the returned chunks.
        Sorted descending by ``score``.
    """
    fetch = k * 2

    # --- Dense retrieval with cosine distances ---
    encoder = _get_encoder()
    query_embedding: list[float] = encoder.encode(
        [query], normalize_embeddings=True
    ).tolist()[0]

    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(fetch, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    dense_texts: list[str] = dense_results["documents"][0]
    dense_metas: list[dict] = dense_results["metadatas"][0]
    # ChromaDB cosine collection stores 1 - cosine_similarity as distance.
    dense_distances: list[float] = dense_results["distances"][0]
    dense_similarities: list[float] = [1.0 - d for d in dense_distances]

    # Map text → cosine similarity for fast lookup during fusion.
    text_to_similarity: dict[str, float] = {}
    for text, sim in zip(dense_texts, dense_similarities):
        text_to_similarity.setdefault(text, sim)

    # --- Sparse BM25 retrieval ---
    tokenized_corpus = [chunk["text"].lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())

    top_sparse_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:fetch]
    sparse_texts: list[str] = [all_chunks[i]["text"] for i in top_sparse_indices]
    sparse_metas: list[dict] = [
        {"source": all_chunks[i].get("source", ""), "page": all_chunks[i].get("page", 0)}
        for i in top_sparse_indices
    ]

    # --- Reciprocal Rank Fusion ---
    RRF_K = 60
    fused: dict[str, dict] = {}

    for rank, (text, meta) in enumerate(zip(dense_texts, dense_metas)):
        entry = fused.setdefault(
            text,
            {
                "text": text,
                "metadata": meta,
                "score": 0.0,
                "max_similarity": text_to_similarity.get(text, -1.0),
            },
        )
        entry["score"] += 1.0 / (rank + RRF_K)

    for rank, (text, meta) in enumerate(zip(sparse_texts, sparse_metas)):
        entry = fused.setdefault(
            text,
            {
                "text": text,
                "metadata": meta,
                "score": 0.0,
                "max_similarity": text_to_similarity.get(text, -1.0),
            },
        )
        entry["score"] += 1.0 / (rank + RRF_K)

    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:k]

    # Attach corpus_confidence as a list subclass attribute so callers can read
    # it without changing the return type.
    top3_sims = [c["max_similarity"] for c in ranked[:3] if c["max_similarity"] >= 0]
    corpus_confidence = sum(top3_sims) / len(top3_sims) if top3_sims else 0.0

    class RankedList(list):
        pass

    result = RankedList(ranked)
    result.corpus_confidence = round(corpus_confidence, 4)
    return result
