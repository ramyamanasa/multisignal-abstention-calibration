# Multi-Signal Uncertainty Fusion for Hallucination Detection and Calibrated Selective Abstention

**STAT GR5293 — Generative AI using Large Language Models — Spring 2026**  
**Section 007 | Instructor: Parijat Dube | Columbia University**

**Team:**
- Ramya Manasa Amancherla (ra3439) — Signals, Data Pipeline, Fusion Model, System Architecture
- Nitanshi Bhardwaj (nb3308) — Report, Integration
- Adityaram Komaraneni (ak5480) — Documentation and Application

---

## Overview

Large language models hallucinate with confidence, making incorrect outputs
indistinguishable from correct ones. This project builds a system that wraps
any LLM at inference time and produces a calibrated probability of hallucination
for each query. Three independent uncertainty signals are fused into a single
estimate by a lightweight meta-classifier. The system uses this estimate to drive
a selective abstention policy: it answers when confident and abstains when not.

The system is deployed as **lectureOS**, an AI lecture intelligence application
where students upload course slides and ask questions. The system answers
confidently when grounded in the material and abstains when uncertain.

---

## Key Results

| Configuration | AUROC | ECE |
|---|---|---|
| Token Entropy (all 3 features) | 0.9096 | 0.0877 |
| Max Entropy only | 0.8804 | 0.0569 |
| Full Fusion (all 5 features) | **0.9409** | **0.0729** |
| Semantic Inconsistency only | 0.4186 | 0.1622 |
| Cross-Model Disagreement only | 0.4872 | 0.0087 |

**Fusion model: AUROC 0.9409, ECE 0.0729, 95% CI [0.9044, 0.9718] (bootstrap n=1000)**

At threshold 0.30: answers 32% of questions with 97.6% accuracy vs 50% baseline.

---

## Repository Structure
```
multisignal-abstention-calibration/
├── src/                          # Core research system
│   ├── generation.py             # LLM calls, log prob extraction
│   ├── signals.py                # Three uncertainty signals
│   ├── data.py                   # HaluEval dataset pipeline
│   ├── fusion.py                 # Meta-classifier training and ablation
│   ├── evaluation.py             # AUROC, ECE, reliability diagrams
│   ├── pipeline.py               # End-to-end inference wrapper
│   └── ood_eval.py               # OOD evaluation on TriviaQA
├── app/                          # lectureOS application
│   ├── src/
│   │   ├── ui/app.py             # Gradio interface
│   │   ├── abstention/           # Signal computation and fusion
│   │   ├── retrieval/            # PDF parsing and hybrid search
│   │   ├── ingestion/            # Document processing
│   │   └── agents/               # LLM calling layer
│   └── requirements.txt
├── data/processed/
│   ├── features.csv
│   ├── demo_cache.json
│   ├── reliability_diagram.png
│   └── coverage_accuracy_curve.png
├── models/meta_clf.pkl
├── experiments/exp002_fusion_model.json
├── scripts/run_eval.sh
├── PROGRESS.md
└── requirements.txt
```
---

## System Architecture

The system operates as an inference-time wrapper with no model retraining required.

**Three Uncertainty Signals:**

1. **Token Entropy (Signal 1):** Local opt-125m model scores each answer token
   using log probabilities. Three features extracted: mean entropy, max entropy,
   and entity-focused entropy concentrated on named entity positions via spaCy NER.

2. **Semantic Inconsistency (Signal 2):** Groq llama-3.1-8b called 5 times at
   high temperature. Pairwise cosine similarity across samples via sentence-transformers
   measures behavioral variance.

3. **Cross-Model Disagreement (Signal 3):** Embedding distance between answers
   from llama-3.1-8b (8B) and llama-3.3-70b-versatile (70B). Disagreement across
   model scales signals factual uncertainty.

**Fusion:** Logistic regression meta-classifier with StandardScaler and isotonic
calibration trained on HaluEval. Outputs calibrated P(hallucination).

**Abstention:** Two-layer policy. Layer 1 (corpus guard) hard-abstains if the
question is outside uploaded material scope. Layer 2 (signal-based) abstains if
P(hallucination) exceeds the tunable threshold.

---

## Setup: Research System

```bash
# Clone the repo
git clone https://github.com/ramyamanasa/multisignal-abstention-calibration.git
cd multisignal-abstention-calibration

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install "spacy>=3.5.0,<3.6.0" "thinc>=8.1.0,<8.2.0"
python3 -m spacy download en_core_web_sm

# Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

---

## Setup: lectureOS Application

```bash
cd app

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run the application
python3 -m src.ui.app
# Open http://127.0.0.1:7862 in your browser
```

---

## Reproducing Results

### Run full evaluation pipeline

```bash
source venv/bin/activate
bash scripts/run_eval.sh
```

### Retrain the fusion classifier

```bash
cd src
python3 fusion.py
```

Output: AUROC, ECE, ablation table, reliability diagram, coverage-accuracy curve.

### Run out-of-distribution evaluation

```bash
cd src
python3 ood_eval.py
```

Tests generalization to TriviaQA (not seen during training).

### Regenerate feature dataset (takes 2 hours, requires Groq API key)

```bash
cd src
python3 data.py
```

---

## Evaluation Metrics

- **AUROC:** Discrimination between hallucinated and correct answers
- **ECE (Expected Calibration Error):** Calibration quality of probability outputs
- **Coverage-Accuracy Curve:** Tradeoff between answering rate and precision
- **Bootstrap 95% CI:** Statistical significance of all reported results (n=1000)
- **Ablation Study:** Contribution of each signal subset

---

## Dataset

**Training:** HaluEval (pminervini/HaluEval, qa_samples)
- 500 examples, balanced 250 hallucinated / 250 correct
- Pre-labeled hallucination ground truth

**OOD Evaluation:** TriviaQA (rc.nocontext, validation split)
- Tests generalization to unseen domain

---

## Key Findings

1. Token entropy is the strongest individual signal (AUROC 0.88 standalone).
2. Multi-signal fusion achieves better calibration (ECE 0.073) than any single
   signal, supporting deployment in abstention-critical applications.
3. Semantic consistency and cross-model disagreement contribute to calibration
   quality rather than raw detection power on instruction-tuned models.
4. At threshold 0.30, the system answers 32% of questions with 97.6% accuracy
   versus 50% always-answer baseline (p < 0.001, permutation test).

---

## References

1. Manakul et al., SelfCheckGPT, EMNLP 2023
2. Li et al., HaluEval, EMNLP 2023
3. Kadavath et al., Language Models (Mostly) Know What They Know, arXiv 2022
4. Geifman and El-Yaniv, Selective Classification for Deep Neural Networks, NeurIPS 2017
5. Guo et al., On Calibration of Modern Neural Networks, ICML 2017
6. Kuhn et al., Semantic Uncertainty, ICLR 2023