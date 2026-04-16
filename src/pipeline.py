"""
Module: pipeline.py
Owner: Person C (Integration + Demo)
Responsibility: End-to-end function for a single query. Uses stubs until real modules are ready.
"""

def run_pipeline(question: str, threshold: float = 0.5) -> dict:
    """
    Full pipeline for a single input question.

    Returns dict with keys:
        answer (str)
        token_logprobs (list)
        entropy_signals (dict)
        consistency_signals (dict)
        disagreement_signals (dict)
        feature_vector (list of 5 floats)
        hallucination_probability (float)
        decision (str): 'answer' or 'abstain'
    """
    raise NotImplementedError
