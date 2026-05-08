"""Tests for src/retrieval/index.py and src/retrieval/retrieve.py."""

import importlib
import numpy as np
import pytest

import src.retrieval.index as index_mod


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

LONG_TEXT = " ".join(["word"] * 300)  # 1 500+ chars — forces at least two chunks

SAMPLE_DOCS = [
    {"text": LONG_TEXT, "source": "lecture1.pdf", "page": 1},
    {"text": "Short sentence about attention mechanisms.", "source": "lecture2.pdf", "page": 3},
]

SAMPLE_CHUNKS = [
    {"text": "Attention is all you need.", "source": "lecture1.pdf", "page": 1},
    {"text": "Gradient descent minimizes the loss.", "source": "lecture1.pdf", "page": 2},
    {"text": "Transformers replaced RNNs for sequence tasks.", "source": "lecture2.pdf", "page": 1},
    {"text": "Backpropagation computes gradients efficiently.", "source": "lecture2.pdf", "page": 2},
    {"text": "Softmax normalizes logit vectors into probabilities.", "source": "lecture3.pdf", "page": 1},
    {"text": "Layer normalization stabilizes deep network training.", "source": "lecture3.pdf", "page": 2},
]


@pytest.fixture(autouse=True)
def mock_encoder(monkeypatch):
    """Replace SentenceTransformer with a fast deterministic stub."""

    class _FakeEncoder:
        def encode(self, texts, show_progress_bar=False, normalize_embeddings=False):
            rng = np.random.default_rng(seed=42)
            return rng.random((len(texts), 16)).astype(np.float32)

    fake = _FakeEncoder()

    # Patch the module-level singleton so both index and retrieve share it
    monkeypatch.setattr(index_mod, "_encoder", fake)
    # Also intercept fresh SentenceTransformer() calls inside _get_encoder
    monkeypatch.setattr(index_mod, "SentenceTransformer", lambda *a, **kw: fake)

    # retrieve.py imports _get_encoder from index; patch the same object there
    import src.retrieval.retrieve as retrieve_mod
    monkeypatch.setattr(retrieve_mod, "_get_encoder", lambda: fake)

    yield fake


@pytest.fixture()
def chroma_tmp(tmp_path, monkeypatch):
    """Point DATA_DIR at a temp directory so ChromaDB never touches real data."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # Force _chroma_path() to re-evaluate with the new env var
    importlib.reload(index_mod)
    # Re-apply the encoder patch after reload
    class _FakeEncoder:
        def encode(self, texts, show_progress_bar=False, normalize_embeddings=False):
            rng = np.random.default_rng(seed=42)
            return rng.random((len(texts), 16)).astype(np.float32)

    monkeypatch.setattr(index_mod, "_encoder", _FakeEncoder())
    yield tmp_path


# ---------------------------------------------------------------------------
# chunk_documents
# ---------------------------------------------------------------------------

class TestChunkDocuments:
    def test_returns_list_of_dicts(self):
        chunks = index_mod.chunk_documents(SAMPLE_DOCS)
        assert isinstance(chunks, list)
        assert all(isinstance(c, dict) for c in chunks)

    def test_each_chunk_has_required_keys(self):
        chunks = index_mod.chunk_documents(SAMPLE_DOCS)
        for chunk in chunks:
            assert "text" in chunk
            assert "source" in chunk
            assert "page" in chunk

    def test_long_doc_produces_multiple_chunks(self):
        docs = [{"text": LONG_TEXT, "source": "a.pdf", "page": 1}]
        chunks = index_mod.chunk_documents(docs, chunk_size=200, overlap=20)
        assert len(chunks) > 1

    def test_source_metadata_preserved(self):
        chunks = index_mod.chunk_documents(SAMPLE_DOCS)
        sources = {c["source"] for c in chunks}
        assert "lecture1.pdf" in sources
        assert "lecture2.pdf" in sources

    def test_page_metadata_preserved(self):
        docs = [{"text": "Hello world.", "source": "x.pdf", "page": 7}]
        chunks = index_mod.chunk_documents(docs)
        assert all(c["page"] == 7 for c in chunks)

    def test_short_doc_produces_single_chunk(self):
        docs = [{"text": "Short text.", "source": "b.pdf", "page": 0}]
        chunks = index_mod.chunk_documents(docs, chunk_size=400)
        assert len(chunks) == 1

    def test_empty_docs_list_returns_empty(self):
        assert index_mod.chunk_documents([]) == []


# ---------------------------------------------------------------------------
# build_index
# ---------------------------------------------------------------------------

class TestBuildIndex:
    def test_returns_collection(self, chroma_tmp):
        import chromadb
        col = index_mod.build_index(SAMPLE_CHUNKS, "test_col")
        assert isinstance(col, chromadb.Collection)

    def test_collection_count_matches_chunks(self, chroma_tmp):
        col = index_mod.build_index(SAMPLE_CHUNKS, "test_col")
        assert col.count() == len(SAMPLE_CHUNKS)

    def test_collection_name_matches_argument(self, chroma_tmp):
        col = index_mod.build_index(SAMPLE_CHUNKS, "my_lecture")
        assert col.name == "my_lecture"

    def test_overwrite_existing_collection(self, chroma_tmp):
        index_mod.build_index(SAMPLE_CHUNKS, "overwrite_col")
        col = index_mod.build_index(SAMPLE_CHUNKS[:2], "overwrite_col")
        assert col.count() == 2

    def test_persisted_to_disk(self, chroma_tmp):
        index_mod.build_index(SAMPLE_CHUNKS, "persist_col")
        chroma_dir = chroma_tmp / "chroma_db"
        assert chroma_dir.exists()


# ---------------------------------------------------------------------------
# load_index
# ---------------------------------------------------------------------------

class TestLoadIndex:
    def test_raises_value_error_for_missing_collection(self, chroma_tmp):
        with pytest.raises(ValueError, match="not found"):
            index_mod.load_index("does_not_exist")

    def test_loads_existing_collection(self, chroma_tmp):
        index_mod.build_index(SAMPLE_CHUNKS, "load_col")
        col = index_mod.load_index("load_col")
        assert col.count() == len(SAMPLE_CHUNKS)

    def test_error_message_contains_collection_name(self, chroma_tmp):
        with pytest.raises(ValueError, match="ghost_collection"):
            index_mod.load_index("ghost_collection")


# ---------------------------------------------------------------------------
# hybrid_retrieve
# ---------------------------------------------------------------------------

class TestHybridRetrieve:
    @pytest.fixture()
    def collection(self, chroma_tmp):
        return index_mod.build_index(SAMPLE_CHUNKS, "retrieve_col")

    def test_returns_list(self, collection):
        from src.retrieval.retrieve import hybrid_retrieve
        results = hybrid_retrieve("attention mechanism", collection, SAMPLE_CHUNKS, k=3)
        assert isinstance(results, list)

    def test_returns_k_results(self, collection):
        from src.retrieval.retrieve import hybrid_retrieve
        results = hybrid_retrieve("gradient descent", collection, SAMPLE_CHUNKS, k=3)
        assert len(results) == 3

    def test_returns_fewer_than_k_when_corpus_small(self, chroma_tmp):
        from src.retrieval.retrieve import hybrid_retrieve
        tiny = SAMPLE_CHUNKS[:2]
        col = index_mod.build_index(tiny, "tiny_col")
        results = hybrid_retrieve("loss", col, tiny, k=5)
        assert len(results) <= 5

    def test_each_result_has_text_key(self, collection):
        from src.retrieval.retrieve import hybrid_retrieve
        results = hybrid_retrieve("backpropagation", collection, SAMPLE_CHUNKS, k=2)
        assert all("text" in r for r in results)

    def test_each_result_has_metadata_key(self, collection):
        from src.retrieval.retrieve import hybrid_retrieve
        results = hybrid_retrieve("transformers", collection, SAMPLE_CHUNKS, k=2)
        assert all("metadata" in r for r in results)

    def test_each_result_has_score_key(self, collection):
        from src.retrieval.retrieve import hybrid_retrieve
        results = hybrid_retrieve("softmax", collection, SAMPLE_CHUNKS, k=2)
        assert all("score" in r for r in results)

    def test_scores_are_positive_floats(self, collection):
        from src.retrieval.retrieve import hybrid_retrieve
        results = hybrid_retrieve("normalization", collection, SAMPLE_CHUNKS, k=3)
        assert all(isinstance(r["score"], float) and r["score"] > 0 for r in results)

    def test_results_sorted_descending_by_score(self, collection):
        from src.retrieval.retrieve import hybrid_retrieve
        results = hybrid_retrieve("layer normalization", collection, SAMPLE_CHUNKS, k=4)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_duplicate_texts(self, collection):
        from src.retrieval.retrieve import hybrid_retrieve
        results = hybrid_retrieve("attention", collection, SAMPLE_CHUNKS, k=5)
        texts = [r["text"] for r in results]
        assert len(texts) == len(set(texts))
