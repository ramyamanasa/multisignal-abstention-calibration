# lectureOS — AI Lecture Intelligence System

## What this project does
Processes lecture recordings and slides, answers student questions
with grounded citations, calibrated confidence, and selective abstention.

## Environment
- Conda env: lectureOS (Python 3.11)
- Activate: conda activate lectureOS
- Install deps: pip install -r requirements.txt

## Local model
- Ollama running qwen2.5:7b-instruct on localhost:11434
- Start with: brew services start ollama

## Run the app
- UI: python src/ui/app.py
- Tests: pytest tests/ -v

## Folder structure
- src/ingestion/    — Whisper transcription, PDF/PPTX parsing
- src/retrieval/    — ChromaDB, FAISS, hybrid BM25+dense search
- src/agents/       — LangGraph orchestrator, abstention logic
- src/alignment/    — Pedagogical modes, QLoRA fine-tuning
- src/eval/         — Metrics: ECE, selective accuracy, AUROC
- src/ui/           — Gradio interface
- data/raw/         — Raw uploads (gitignored)
- data/eval/        — Gold QA pairs for evaluation
- notebooks/        — Experiments and ablation studies

## Code style
- Type hints on all functions
- Google-style docstrings
- Functions under 40 lines
- Never hardcode API keys — always use os.getenv()
