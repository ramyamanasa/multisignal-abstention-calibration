"""
Module: generation.py
Owner: Person A (Signals)
Responsibility: LLM calls, log prob extraction, stochastic sampling, second model call.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Model loading (call once, reuse across questions)
# ---------------------------------------------------------------------------

def load_model(model_name: str = "facebook/opt-125m"):
    """
    Loads tokenizer and model onto CPU.
    Returns (tokenizer, model).
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
    )
    model.eval()
    print("Model loaded.")
    return tokenizer, model


# ---------------------------------------------------------------------------
# Core generation: answer + token log probabilities
# ---------------------------------------------------------------------------

def generate_with_logprobs(
    question: str,
    tokenizer,
    model,
    max_new_tokens: int = 50,
) -> dict:
    """
    Generates a greedy answer for the question and returns token-level log probs.

    Returns dict with keys:
        answer_text     (str)        the generated answer
        token_logprobs  (list of float)  log prob of each generated token
        tokens          (list of str)    the actual token strings
    """
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            eos_token_id=newline_id,
        )

    # Exclude prompt tokens
    generated_ids = outputs.sequences[0][input_len:]

    # Compute log probs from scores
    log_probs = []
    for i, score in enumerate(outputs.scores):
        token_id = generated_ids[i].item()
        lp = torch.log_softmax(score[0], dim=-1)[token_id].item()
        log_probs.append(lp)

    tokens = [tokenizer.decode([tid]) for tid in generated_ids]
    answer_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return {
        "answer_text": answer_text,
        "token_logprobs": log_probs,
        "tokens": tokens,
    }


# ---------------------------------------------------------------------------
# Stochastic sampling: N answers for consistency signal
# ---------------------------------------------------------------------------

def generate_samples(
    question: str,
    tokenizer,
    model,
    n: int = 10,
    temperature: float = 0.7,
    max_new_tokens: int = 50,
) -> list:
    """
    Generates N stochastic answers for the same question.

    Returns:
        samples (list of str) - n answer strings
    """
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]

    samples = []
    with torch.no_grad():
        for _ in range(n):
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                eos_token_id=newline_id,
            )
            generated_ids = output[0][input_len:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            samples.append(text)

    return samples


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_QUESTIONS = [
        "Who wrote the play Hamlet?",
        "What is the capital of France?",
        "What year did the first moon landing occur?",
    ]

    tokenizer, model = load_model("facebook/opt-125m")

    for q in TEST_QUESTIONS:
        print(f"\nQuestion: {q}")
        result = generate_with_logprobs(q, tokenizer, model)
        print(f"Answer:     {result['answer_text']}")
        print(f"Tokens:     {result['tokens']}")
        print(f"Log probs:  {[round(lp, 3) for lp in result['token_logprobs']]}")
        print(f"Mean logp:  {round(np.mean(result['token_logprobs']), 3)}")
        print(f"Num tokens: {len(result['tokens'])}")