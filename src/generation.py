"""
Module: generation.py
Owner: Person A (Signals)
Responsibility: LLM calls, log prob extraction, stochastic sampling, second model call.
"""

def generate_with_logprobs(question: str, model, tokenizer, device="cuda"):
    """
    Returns:
        answer_text (str)
        token_logprobs (list of float)
    """
    raise NotImplementedError

def generate_samples(question: str, model, tokenizer, n=10, temperature=0.7, device="cuda"):
    """
    Returns:
        samples (list of str) - n stochastic answers
    """
    raise NotImplementedError

def generate_model2_answer(question: str):
    """
    Calls second model (e.g., Phi-3-mini via API).
    Returns:
        answer_text (str)
    """
    raise NotImplementedError
