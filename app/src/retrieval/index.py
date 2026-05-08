import os
from typing import Any

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_encoder: SentenceTransformer | None = None


def _get_encoder() -> SentenceTransformer:
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(_EMBED_MODEL)
    return _encoder


def _chroma_path() -> str:
    data_dir = os.getenv("DATA_DIR", "data")
    return os.path.join(data_dir, "chroma_db")


def chunk_documents(
    docs: list[dict],
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[dict]:
    """Split documents into overlapping text chunks while preserving metadata.

    Args:
        docs: List of document dicts, each with at least a ``text`` key and
            optional ``source`` and ``page`` metadata keys.
        chunk_size: Maximum character length of each chunk.
        overlap: Number of characters shared between consecutive chunks.

    Returns:
        List of chunk dicts with keys ``text``, ``source``, and ``page``.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks: list[dict] = []
    for doc in docs:
        text = doc.get("text", "")
        source = doc.get("source", "")
        page = doc.get("page", 0)
        for piece in splitter.split_text(text):
            chunks.append({"text": piece, "source": source, "page": page})
    return chunks


def build_index(
    chunks: list[dict],
    collection_name: str,
) -> chromadb.Collection:
    """Embed chunks and store them in a persistent ChromaDB collection.

    Creates or overwrites the named collection at the path defined by
    ``DATA_DIR`` (defaults to ``data/chroma_db/``).

    Args:
        chunks: List of chunk dicts produced by :func:`chunk_documents`.
            Each dict must have ``text``, ``source``, and ``page`` keys.
        collection_name: Name of the ChromaDB collection to create.

    Returns:
        The populated :class:`chromadb.Collection` instance.
    """
    client = chromadb.PersistentClient(path=_chroma_path())

    if collection_name in [c.name for c in client.list_collections()]:
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    encoder = _get_encoder()
    texts = [c["text"] for c in chunks]
    embeddings: list[list[float]] = encoder.encode(
        texts, show_progress_bar=True, normalize_embeddings=True
    ).tolist()

    ids = [f"{collection_name}_{i}" for i in range(len(chunks))]
    # Preserve all non-text keys as ChromaDB metadata (values must be scalar).
    metadatas: list[dict[str, Any]] = [
        {k: v for k, v in c.items() if k != "text"} for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return collection


def load_index(collection_name: str) -> chromadb.Collection:
    """Load an existing ChromaDB collection without rebuilding it.

    Args:
        collection_name: Name of the collection to load.

    Returns:
        The existing :class:`chromadb.Collection` instance.

    Raises:
        ValueError: If the collection does not exist.
    """
    client = chromadb.PersistentClient(path=_chroma_path())
    existing = [c.name for c in client.list_collections()]
    if collection_name not in existing:
        raise ValueError(
            f"Collection '{collection_name}' not found in {_chroma_path()}. "
            "Run build_index first."
        )
    return client.get_collection(collection_name)
