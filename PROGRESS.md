# Project Progress Log

## Team
- Person A (Signals): Ramya
- Person B (Fusion + Evaluation): Nitanshi
- Person C (Integration + Demo): Aditya

---

## Week 1

### Day 1
**Status:** Generation + all 3 signals working on test questions
**What works:** generate_with_logprobs, generate_samples, compute_entropy_signal,
compute_consistency_signal, compute_disagreement_signal, build_feature_vector
**Blockers:** cross_model_disagreement was using first stochastic sample as proxy
**Next:** Load real dataset, run signals at scale

### Day 2
**Status:** Full feature dataset built on HaluEval, first real AUROC numbers
**What works:** Complete pipeline end to end, 200 questions, perfectly balanced 100/100
**Key results:**
- max_entropy AUROC: 0.9168
- mean_entropy AUROC: 0.8319
- entity_entropy AUROC: 0.6030
- semantic_inconsistency AUROC: 0.5314
- cross_model_disagreement AUROC: 0.5180
**Finding:** Entropy signals are dominant. Signals 2 and 3 measure Groq uncertainty,
not answer quality. Need redesign for fusion to add value over entropy alone.
**Next:** Build fusion classifier, run ablation

---

## Experiment Index
| Exp ID | Date | Description | best AUROC | Notes |
|--------|------|-------------|------------|-------|
| exp001 | 2026-04-20 | Single signals on HaluEval 200q | 0.9168 (max_entropy) | Entropy dominates | 
