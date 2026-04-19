"""
Module: signals.py
Owner: Person A (Signals)
Responsibility: All three uncertainty signal computations.
"""

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, util

# Load these once at import time
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Signal 1: Token Entropy (entity-focused)
# ---------------------------------------------------------------------------

def compute_entropy_signal(token_logprobs: list, tokens: list) -> dict:
    """
    Args:
        token_logprobs: list of log probabilities per generated token
        tokens:         list of token strings (same length as token_logprobs)

    Returns dict with keys:
        mean_entropy        float  average entropy across all tokens
        max_entropy         float  highest entropy token
        entity_entropy      float  average entropy at named entity positions
                                   (nan if no entities found)
    """
    if len(token_logprobs) == 0:
        return {"mean_entropy": float("nan"),
                "max_entropy": float("nan"),
                "entity_entropy": float("nan")}

    # Convert log probs to entropy values: H(token) = -p * log(p) = -exp(lp) * lp
    log_probs = np.array(token_logprobs)
    probs = np.exp(log_probs)
    entropies = -probs * log_probs          # one entropy value per token

    mean_entropy = float(np.mean(entropies))
    max_entropy  = float(np.max(entropies))

    # Find named entity positions using spaCy
    # We reconstruct the answer text from tokens and run NER on it
    answer_text = "".join(tokens).strip()
    doc = nlp(answer_text)

    # Map character positions of entities back to token indices
    entity_indices = _find_entity_token_indices(tokens, doc)

    if len(entity_indices) == 0:
        entity_entropy = mean_entropy  # fallback to mean instead of nan
    else:
        entity_entropy = float(np.mean(entropies[entity_indices]))

    return {
        "mean_entropy":   mean_entropy,
        "max_entropy":    max_entropy,
        "entity_entropy": entity_entropy,
    }


def _find_entity_token_indices(tokens: list, doc) -> np.ndarray:
    """
    Finds which token indices correspond to named entities.
    Matches by rebuilding character offsets from the token list.
    """
    # Build character offset map: for each token, what char range does it cover
    char_offsets = []
    cursor = 0
    for tok in tokens:
        start = cursor
        end = cursor + len(tok)
        char_offsets.append((start, end))
        cursor = end

    # Collect character spans of all named entities
    entity_char_spans = [(ent.start_char, ent.end_char) for ent in doc.ents]

    # Find token indices that overlap with any entity span
    entity_indices = []
    for i, (tok_start, tok_end) in enumerate(char_offsets):
        for ent_start, ent_end in entity_char_spans:
            if tok_start < ent_end and tok_end > ent_start:  # overlap
                entity_indices.append(i)
                break

    return np.array(entity_indices, dtype=int)


# ---------------------------------------------------------------------------
# Signal 2: Semantic Consistency
# ---------------------------------------------------------------------------

def compute_consistency_signal(samples: list) -> dict:
    """
    Args:
        samples: list of N answer strings (stochastic samples)

    Returns dict with key:
        semantic_inconsistency  float  0 = all samples say same thing,
                                       1 = samples completely disagree
    """
    if len(samples) < 2:
        return {"semantic_inconsistency": float("nan")}

    embeddings = embedder.encode(samples, convert_to_tensor=True)

    # Compute all pairwise cosine similarities
    sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()

    # Take upper triangle only (avoid self-similarity and duplicates)
    n = len(samples)
    upper_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_indices]

    mean_similarity = float(np.mean(pairwise_sims))
    inconsistency = 1.0 - mean_similarity

    return {"semantic_inconsistency": inconsistency}


# ---------------------------------------------------------------------------
# Signal 3: Cross-Model Disagreement
# ---------------------------------------------------------------------------

def compute_disagreement_signal(answer_model1: str, answer_model2: str) -> dict:
    """
    Args:
        answer_model1: answer string from primary model
        answer_model2: answer string from secondary model

    Returns dict with key:
        cross_model_disagreement  float  0 = models agree, 1 = completely disagree
    """
    if not answer_model1 or not answer_model2:
        return {"cross_model_disagreement": float("nan")}

    emb1 = embedder.encode(answer_model1, convert_to_tensor=True)
    emb2 = embedder.encode(answer_model2, convert_to_tensor=True)

    similarity = float(util.cos_sim(emb1, emb2).item())
    disagreement = 1.0 - similarity

    return {"cross_model_disagreement": disagreement}


# ---------------------------------------------------------------------------
# Feature vector assembly
# ---------------------------------------------------------------------------

def build_feature_vector(
    entropy_signals: dict,
    consistency_signals: dict,
    disagreement_signals: dict,
) -> np.ndarray:
    """
    Assembles the 5D feature vector in fixed order:
    [mean_entropy, max_entropy, entity_entropy,
     semantic_inconsistency, cross_model_disagreement]

    nan values are kept as nan here.
    The fusion model will handle imputation.
    """
    return np.array([
        entropy_signals.get("mean_entropy",            float("nan")),
        entropy_signals.get("max_entropy",             float("nan")),
        entropy_signals.get("entity_entropy",          float("nan")),
        consistency_signals.get("semantic_inconsistency",   float("nan")),
        disagreement_signals.get("cross_model_disagreement", float("nan")),
    ])


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from generation import load_model, generate_with_logprobs, generate_samples

    TEST_QUESTIONS = [
        "Who wrote the play Hamlet?",
        "What is the capital of France?",
        "What year did the first moon landing occur?",
    ]

    tokenizer, model = load_model("facebook/opt-125m")

    for q in TEST_QUESTIONS:
        print(f"\n{'='*60}")
        print(f"Question: {q}")

        # Signal 1
        result = generate_with_logprobs(q, tokenizer, model)
        print(f"Answer:   {result['answer_text']}")
        entropy_signals = compute_entropy_signal(
            result["token_logprobs"], result["tokens"]
        )
        print(f"Signal 1 (entropy):")
        print(f"  mean_entropy:   {entropy_signals['mean_entropy']:.4f}")
        print(f"  max_entropy:    {entropy_signals['max_entropy']:.4f}")
        print(f"  entity_entropy: {entropy_signals['entity_entropy']}")

        # Signal 2
        samples = generate_samples(q, tokenizer, model, n=5)
        print(f"Samples: {samples}")
        consistency_signals = compute_consistency_signal(samples)
        print(f"Signal 2 (consistency):")
        print(f"  semantic_inconsistency: {consistency_signals['semantic_inconsistency']:.4f}")

        # Signal 3 (using two different answers as a proxy for now)
        answer_a = result["answer_text"]
        answer_b = samples[0] if samples else ""
        disagreement_signals = compute_disagreement_signal(answer_a, answer_b)
        print(f"Signal 3 (disagreement):")
        print(f"  cross_model_disagreement: {disagreement_signals['cross_model_disagreement']:.4f}")

        # Feature vector
        fv = build_feature_vector(entropy_signals, consistency_signals, disagreement_signals)
        print(f"Feature vector: {np.round(fv, 4)}")