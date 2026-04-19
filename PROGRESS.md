# Project Progress Log

## Team
- Person A (Signals): Ramya
- Person B (Fusion + Evaluation): Nitanshi
- Person C (Integration + Demo): Adityaram Komaraneni 

---

## Week 1

### Day 1 - April 16, 2026
**Status:** Generation + all 3 signals working on test questions
**What works:** generate_with_logprobs, generate_samples, compute_entropy_signal,
compute_consistency_signal, compute_disagreement_signal, build_feature_vector
**Blockers:** cross_model_disagreement currently uses first stochastic sample
as proxy for second model. Real second model not wired yet.
**Next:** Load TriviaQA dataset, run signals on 100 questions, get first AUROC number