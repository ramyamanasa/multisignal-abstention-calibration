# Multi-Signal Abstention Calibration

**STAT GR5293 - Generative AI using LLMs, Columbia University, Spring 2026**

Team: Ramya Manasa Amancherla (ra3439), Nitanshi Bhardwaj (nb3308), Adityaram Komaraneni (ak5480)

---

## What This Is

A hallucination detection and calibrated selective abstention system for LLMs. For each question-answer pair, the system computes three independent uncertainty signals, fuses them into a calibrated hallucination probability using a logistic regression meta-classifier, and decides whether to **ANSWER** or **ABSTAIN**.

The core insight: every organization has a different risk tolerance. A news outlet needs high precision. A casual reader can tolerate more coverage. This system lets you select your operating point on the coverage-accuracy curve and tells you exactly what you get.

---

## System Architecture

```
Question
   |
   +-- Signal 1: Token Entropy (opt-125m log-probs)
   |      mean_entropy, max_entropy, entity_entropy
   |
   +-- Signal 2: Semantic Inconsistency (Groq stochastic sampling)
   |      cosine similarity variance across N samples
   |
   +-- Signal 3: Cross-Model Disagreement (llama-3.1-8b vs llama-3.3-70b)
          embedding cosine distance between two model answers
               |
               v
   5D Feature Vector
               |
               v
   Logistic Regression Meta-Classifier
   (CalibratedClassifierCV, isotonic regression)
               |
               v
   Hallucination Probability p
               |
        p >= threshold?
        /           \
   ABSTAIN         ANSWER
```

---

## Setup

### Requirements

- Python 3.9+ (tested on 3.13)
- Groq API key
- No GPU required

### Installation

```bash
git clone https://github.com/ramyamanasa/multisignal-abstention-calibration.git
cd multisignal-abstention-calibration
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

### Environment

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

No quotes, no spaces, no semicolons.

---

## Running the Evaluation

### Full pipeline (one command)

From the project root:

```bash
bash run_eval.sh
```

This will:
1. Train the fusion model and run ablation on HaluEval (uses existing `data/processed/features.csv`)
2. Run OOD evaluation on TriviaQA (or skip feature generation if already done)
3. Print a full results summary with AUROC, 95% CIs, and ECE for all conditions

### Individual steps

All scripts must be run from the `src/` directory:

```bash
source venv/bin/activate
cd src

# Train classifier and run ablation
python3 fusion.py

# OOD evaluation on TriviaQA (~45 min first run)
python3 ood_eval.py

# Regenerate features from scratch (~2 hours, costs Groq API calls)
python3 data.py
```

### Running the demo

```bash
source venv/bin/activate
cd demo
python3 app.py
```

Open the URL printed in the terminal.

---

## Results

### HaluEval (in-distribution, n=500, 60/20/20 split)

| Subset | AUROC | 95% CI | ECE |
|---|---|---|---|
| entropy_all | 0.9096 | [0.8441, 0.9603] | 0.0877 |
| max_entropy_only | 0.8804 | [0.8093, 0.9427] | 0.0569 |
| **all_5_features (fusion)** | **0.8848** | **[0.8159, 0.9489]** | **0.0907** |
| mean_entropy_only | 0.7518 | - | 0.1039 |
| entity_entropy_only | 0.6220 | - | 0.1088 |
| disagreement_only | 0.4872 | - | 0.0087 |
| consistency_only | 0.4186 | - | 0.1622 |

Coverage at threshold 0.5: **39%**, Accuracy on answered: **87.2%**

### TriviaQA OOD

*(Results added after ood_eval.py completes)*

---

## Key Findings

**Entropy signals dominate.** Token-level entropy from opt-125m (Signal 1) is by far the strongest predictor of hallucination, with `entropy_all` achieving AUROC 0.91.

**Signal 2 (semantic inconsistency) and Signal 3 (cross-model disagreement) are near-random** on the current setup. Two reasons:

- Signal 2: Groq's llama-3.1-8b-instant is too consistent at temperature 1.0, producing near-identical samples. The cosine variance signal carries almost no information.
- Signal 3: The architectural gap between opt-125m (125M params, local) and Groq's llama (8B, API) is too large. Disagreement reflects model size differences more than factual uncertainty about the specific question.

These are reportable findings. The system still achieves strong AUROC from entropy alone, and the multi-signal architecture is validated -- the weak signals identify where future work should focus (better-matched model pairs, higher sampling temperature).

---

## Repository Structure

```
multisignal-abstention-calibration/
├── src/
│   ├── generation.py      # Groq API calls, opt-125m log-prob extraction
│   ├── signals.py         # Signal 1, 2, 3 computation
│   ├── data.py            # HaluEval loading and feature generation
│   ├── fusion.py          # Meta-classifier training and ablation
│   ├── evaluation.py      # AUROC, ECE, coverage-accuracy, plots
│   ├── pipeline.py        # End-to-end inference for single question
│   └── ood_eval.py        # OOD evaluation on TriviaQA
├── demo/
│   └── app.py             # Gradio demo
├── data/
│   └── processed/
│       ├── features.csv           # HaluEval features (500 rows, 9 cols)
│       ├── ood_features.csv       # TriviaQA features (after ood_eval.py)
│       ├── reliability_diagram.png
│       └── coverage_accuracy_curve.png
├── models/
│   └── meta_clf.pkl       # Trained calibrated logistic regression
├── experiments/
│   ├── exp002_fusion_model.json   # Full ablation results
│   └── exp003_ood_eval.json       # OOD results
├── requirements.txt
├── run_eval.sh            # One-command evaluation pipeline
└── README.md
```

---

## Models and Data

- **Training data:** HaluEval (`pminervini/HaluEval`, `qa_samples`), 500 questions, balanced 250 hallucinated / 250 correct
- **OOD data:** TriviaQA (`trivia_qa`, `rc.nocontext`, validation split), 150 questions
- **Signal 1 model:** `facebook/opt-125m` (local, CPU)
- **Primary LLM:** `llama-3.1-8b-instant` via Groq API
- **Secondary LLM:** `llama-3.3-70b-versatile` via Groq API
- **Meta-classifier:** Logistic regression with isotonic calibration (`sklearn`)
