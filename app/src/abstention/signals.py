"""
Module: signals.py (lectureOS abstention)
Verbatim copy from multisignal-abstention-calibration/src/signals.py
"""

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def compute_entropy_signal(token_logprobs: list, tokens: list) -> dict:
    if len(token_logprobs) == 0:
        return {"mean_entropy": float("nan"),
                "max_entropy": float("nan"),
                "entity_entropy": float("nan")}

    log_probs = np.array(token_logprobs)
    probs = np.exp(log_probs)
    entropies = -probs * log_probs

    mean_entropy = float(np.mean(entropies))
    max_entropy  = float(np.max(entropies))

    answer_text = "".join(tokens).strip()
    doc = nlp(answer_text)
    entity_indices = _find_entity_token_indices(tokens, doc)

    if len(entity_indices) == 0:
        entity_entropy = mean_entropy
    else:
        entity_entropy = float(np.mean(entropies[entity_indices]))

    return {
        "mean_entropy":   mean_entropy,
        "max_entropy":    max_entropy,
        "entity_entropy": entity_entropy,
    }


def _find_entity_token_indices(tokens: list, doc) -> np.ndarray:
    char_offsets = []
    cursor = 0
    for tok in tokens:
        start = cursor
        end = cursor + len(tok)
        char_offsets.append((start, end))
        cursor = end

    entity_char_spans = [(ent.start_char, ent.end_char) for ent in doc.ents]

    entity_indices = []
    for i, (tok_start, tok_end) in enumerate(char_offsets):
        for ent_start, ent_end in entity_char_spans:
            if tok_start < ent_end and tok_end > ent_start:
                entity_indices.append(i)
                break

    return np.array(entity_indices, dtype=int)


def compute_consistency_signal(samples: list) -> dict:
    if len(samples) < 2:
        return {"semantic_inconsistency": float("nan")}

    embeddings = embedder.encode(samples, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()

    n = len(samples)
    upper_indices = np.triu_indices(n, k=1)
    pairwise_sims = sim_matrix[upper_indices]

    mean_similarity = float(np.mean(pairwise_sims))
    inconsistency = 1.0 - mean_similarity

    return {"semantic_inconsistency": inconsistency}


def compute_disagreement_signal(answer_model1: str, answer_model2: str) -> dict:
    if not answer_model1 or not answer_model2:
        return {"cross_model_disagreement": float("nan")}

    emb1 = embedder.encode(answer_model1, convert_to_tensor=True)
    emb2 = embedder.encode(answer_model2, convert_to_tensor=True)

    similarity = float(util.cos_sim(emb1, emb2).item())
    disagreement = 1.0 - similarity

    return {"cross_model_disagreement": disagreement}


def build_feature_vector(
    entropy_signals: dict,
    consistency_signals: dict,
    disagreement_signals: dict,
) -> np.ndarray:
    return np.array([
        entropy_signals.get("mean_entropy",            float("nan")),
        entropy_signals.get("max_entropy",             float("nan")),
        entropy_signals.get("entity_entropy",          float("nan")),
        consistency_signals.get("semantic_inconsistency",   float("nan")),
        disagreement_signals.get("cross_model_disagreement", float("nan")),
    ])
