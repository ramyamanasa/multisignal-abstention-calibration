# lectureOS — AI Lecture Intelligence System

An AI research system that processes lecture recordings and slides, answers student questions with grounded citations, calibrated confidence, and selective abstention.

## What it does

- Transcribes lecture recordings with faster-whisper
- Parses PDF and PPTX slides
- Answers questions using hybrid BM25 + dense retrieval over lecture content
- Routes each query through a LangGraph pipeline — answer, abstain, clarify, or escalate
- Adapts responses to beginner, exam, or deep understanding modes
- Evaluates using selective accuracy, ECE calibration, and AUROC

## Stack

| Layer | Tools |
|---|---|
| Transcription | faster-whisper |
| Parsing | pymupdf, python-pptx |
| Retrieval | ChromaDB, FAISS, BM25, sentence-transformers |
| Orchestration | LangGraph |
| Local model | Qwen2.5-7B-Instruct via Ollama |
| UI | Gradio |
| Evaluation | scikit-learn, custom ECE and selective accuracy |

## Setup

```bash
conda create -n lectureOS python=3.11 -y
conda activate lectureOS
pip install -r requirements.txt
cp .env.example .env  # add your keys
brew services start ollama
ollama pull qwen2.5:7b-instruct
```

## Run

```bash
python src/ui/app.py
```

Open `http://localhost:7860`

## Test

```bash
pytest tests/ -v
```

87 tests across ingestion, retrieval, agents, alignment, and evaluation.

## Project structure
src/
ingestion/    — Whisper transcription, PDF/PPTX parsing
retrieval/    — Hybrid BM25 + dense search, ChromaDB indexing
agents/       — LangGraph orchestrator with 4-action routing
alignment/    — Pedagogical modes, QLoRA fine-tuning scripts
eval/         — ECE, selective accuracy, AUROC metrics
ui/           — Gradio interface
data/
eval/         — Gold QA pairs for evaluation
notebooks/      — Ablation studies and experiments

## Course

Columbia University — STAT GR5293 Generative AI using LLMs, Spring 2026
