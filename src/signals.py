"""
Module: signals.py
Owner: Person A (Signals)
Responsibility: All three uncertainty signal computations.
"""
import numpy as np

def compute_entropy_signal(token_logprobs: list, entity_indices: list) -> dict:
    """
    Args:
        token_logprobs: list of log probabilities per token
        entity_indices: list of int positions that are named entities
    Returns:
        dict with keys: mean_entropy, max_entropy, entity_entropy
    """
    raise NotImplementedError

def compute_consistency_signal(samples: list) -> dict:
    """
    Args:
        samples: list of N answer strings
    Returns:
        dict with key: semantic_inconsistency (float, 0=consistent, 1=inconsistent)
    """
    raise NotImplementedError

def compute_disagreement_signal(answer_model1: str, answer_model2: str) -> dict:
    """
    Args:
        answer_model1: answer string from primary model
        answer_model2: answer string from secondary model
    Returns:
        dict with key: cross_model_disagreement (float)
    """
    raise NotImplementedError

def build_feature_vector(entropy_signals: dict, consistency_signals: dict, disagreement_signals: dict) -> np.ndarray:
    """
    Assembles the 5D feature vector in fixed order:
    [mean_entropy, max_entropy, entity_entropy, semantic_inconsistency, cross_model_disagreement]
    """
    raise NotImplementedError
