"""
Module: generation.py
Owner: Person A (Signals)
Responsibility: LLM calls, log prob extraction, stochastic sampling, second model call.

Architecture:
- Signal 1 (entropy):      opt-125m generates + scores its own tokens locally
- Signal 2 (consistency):  Groq llama-3.1-8b-instant, N stochastic samples
- Signal 3 (disagreement): Groq answer vs opt-125m answer, embedding distance
"""

import os
import torch
import numpy as np
from groq import Groq
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

PRIMARY_MODEL   = "llama-3.1-8b-instant"
SECONDARY_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = (
    "You are a factual question answering assistant. "
    "Answer each question in one short sentence. "
    "Do not explain. Do not hedge. Just answer."
)

SYSTEM_PROMPT_2 = (
    "Answer the following question briefly and directly. "
    "Give only the answer, nothing else."
)

# Local model (opt-125m) for Signal 1
_local_tokenizer = None
_local_model     = None


# ---------------------------------------------------------------------------
# Local model loader
# ---------------------------------------------------------------------------

def _load_local_model():
    global _local_tokenizer, _local_model
    if _local_model is None:
        print("Loading local model (facebook/opt-125m) for Signal 1...")
        _local_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        _local_model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            dtype=torch.float32,
        )
        _local_model.eval()
        print("Local model loaded.")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str = None):
    """
    Loads local opt-125m for Signal 1.
    Returns (None, None) to keep interface compatible with pipeline.py.
    """
    _load_local_model()
    print(f"Using Groq API. Primary: {PRIMARY_MODEL}, Secondary: {SECONDARY_MODEL}")
    return None, None


# ---------------------------------------------------------------------------
# Signal 1: opt-125m generates + scores its own answer
# ---------------------------------------------------------------------------

def generate_with_logprobs(
    question: str,
    tokenizer=None,
    model=None,
    max_new_tokens: int = 50,
) -> dict:
    """
    opt-125m generates its own answer and returns its own token log probs.
    This gives real uncertainty signal from a weak model.

    Returns dict with keys:
        answer_text     (str)
        token_logprobs  (list of float)
        tokens          (list of str)
    """
    _load_local_model()

    prompt = f"Question: {question}\nAnswer:"
    inputs = _local_tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]
    newline_id = _local_tokenizer.encode("\n", add_special_tokens=False)[0]

    with torch.no_grad():
        outputs = _local_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=newline_id,
        )

    generated_ids = outputs.sequences[0][input_len:]

    log_probs = []
    for i, score in enumerate(outputs.scores):
        token_id = generated_ids[i].item()
        lp = torch.log_softmax(score[0], dim=-1)[token_id].item()
        log_probs.append(lp)

    tokens = [_local_tokenizer.decode([tid]) for tid in generated_ids]
    answer_text = _local_tokenizer.decode(
        generated_ids, skip_special_tokens=True
    ).strip()

    return {
        "answer_text":    answer_text,
        "token_logprobs": log_probs,
        "tokens":         tokens,
    }


# ---------------------------------------------------------------------------
# Signal 2: Groq stochastic sampling for consistency
# ---------------------------------------------------------------------------

def generate_samples(
    question: str,
    tokenizer=None,
    model=None,
    n: int = 10,
    temperature: float = 0.7,
    max_new_tokens: int = 50,
) -> list:
    """
    Generates N stochastic answers via Groq for consistency signal.

    Returns:
        samples (list of str)
    """
    samples = []
    for idx in range(n):
        response = groq_client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        samples.append(response.choices[0].message.content.strip())
        if idx < n - 1:
            time.sleep(2)  # avoid RPM burst across consecutive sample calls
    return samples


# ---------------------------------------------------------------------------
# Signal 3: two Groq models for cross-model disagreement
# ---------------------------------------------------------------------------

def generate_primary_answer(
    question: str,
    max_new_tokens: int = 50,
) -> str:
    """Deterministic answer from PRIMARY_MODEL (llama-3.1-8b-instant) for Signal 3.

    Called once per question at temperature=0 so the comparison against
    the secondary model is stable across runs.
    """
    response = groq_client.chat.completions.create(
        model=PRIMARY_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_2},
            {"role": "user",   "content": question},
        ],
        max_tokens=max_new_tokens,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def generate_model2_answer(
    question: str,
    max_new_tokens: int = 50,
) -> str:
    """
    Gets a Groq answer to compare against opt-125m answer for Signal 3.
    The embedding distance between these two answers is the disagreement signal.

    Returns:
        answer_text (str)
    """
    response = groq_client.chat.completions.create(
        model=SECONDARY_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_2},
            {"role": "user",   "content": question},
        ],
        max_tokens=max_new_tokens,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_QUESTIONS = [
        "Who wrote the play Hamlet?",
        "What is the capital of France?",
        "What year did the first moon landing occur?",
    ]

    load_model()

    for q in TEST_QUESTIONS:
        print(f"\n{'='*60}")
        print(f"Question: {q}")

        result = generate_with_logprobs(q)
        print(f"opt-125m Answer: {result['answer_text']}")
        print(f"Tokens:          {result['tokens']}")
        print(f"Log probs:       {[round(lp, 3) for lp in result['token_logprobs']]}")
        print(f"Mean logp:       {round(np.mean(result['token_logprobs']), 3)}")

        model2 = generate_model2_answer(q)
        print(f"Groq Answer:     {model2}")

        samples = generate_samples(q, n=3)
        print(f"Groq Samples:    {samples}")