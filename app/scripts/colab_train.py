"""
lectureOS QLoRA fine-tuning script.

In Colab:
  1. Upload synthetic_qa.jsonl via the Files panel (it lands at /content/).
  2. Upload this script the same way.
  3. Run:  !python colab_train.py

The trained adapter is saved to /content/lectureOS_sft_adapter/.
No Google Drive mounting is required.
"""

# ---------------------------------------------------------------------------
# 0 · Bootstrap: install deps before any heavy imports
# ---------------------------------------------------------------------------
import subprocess
import sys

_PACKAGES = [
    "transformers==4.47.1",
    "peft==0.14.0",
    "bitsandbytes==0.45.0",
    "accelerate==1.2.1",
    "trl==0.13.0",
    "datasets==3.2.0",
    "matplotlib",
]

print("Installing dependencies…")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *_PACKAGES]
)
print("Dependencies ready.\n")

# ---------------------------------------------------------------------------
# 1 · Standard imports (after install)
# ---------------------------------------------------------------------------
import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves PNG without a display
import matplotlib.pyplot as plt
import torch

from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ---------------------------------------------------------------------------
# 2 · Paths
# ---------------------------------------------------------------------------
# Colab uploads land in /content/; locally fall back to the repo layout.
_IN_COLAB = os.path.isdir("/content")

DATA_PATH   = "/content/synthetic_qa.jsonl" if _IN_COLAB else os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "train", "synthetic_qa.jsonl")
)
ADAPTER_DIR = "/content/lectureOS_sft_adapter" if _IN_COLAB else os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "lectureOS_sft_adapter")
)

os.makedirs(ADAPTER_DIR, exist_ok=True)
print(f"Training data: {DATA_PATH}")
print(f"Adapter out  : {ADAPTER_DIR}\n")

# ---------------------------------------------------------------------------
# 3 · Config constants
# ---------------------------------------------------------------------------
MODEL_ID      = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM_PROMPT = (
    "You are lectureOS, an AI teaching assistant. "
    "Answer student questions accurately and concisely based on lecture material."
)
TEST_QUESTION = "What is the role of rank r in LoRA?"

# ---------------------------------------------------------------------------
# 4 · GPU check
# ---------------------------------------------------------------------------
def check_gpu() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU found. "
            "Switch to a GPU runtime: Runtime → Change runtime type → T4/A100."
        )
    print(f"CUDA device : {torch.cuda.get_device_name(0)}")
    print(f"VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

# ---------------------------------------------------------------------------
# 5 · Data
# ---------------------------------------------------------------------------
def load_dataset() -> tuple[Dataset, Dataset]:
    with open(DATA_PATH) as f:
        raw = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(raw):,} examples")

    records = [
        {
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": ex["question"]},
                {"role": "assistant", "content": ex["answer"]},
            ]
        }
        for ex in raw
    ]

    split    = Dataset.from_list(records).train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds  = split["test"]
    print(f"Train: {len(train_ds):,}  |  Eval: {len(eval_ds):,}\n")
    return train_ds, eval_ds

# ---------------------------------------------------------------------------
# 6 · Model + tokenizer
# ---------------------------------------------------------------------------
def load_model() -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache      = False
    model.config.pretraining_tp = 1

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model loaded — {n_params:.2f}B parameters\n")
    return model, tokenizer

# ---------------------------------------------------------------------------
# 7 · LoRA
# ---------------------------------------------------------------------------
def apply_lora(model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()
    print()
    return model

# ---------------------------------------------------------------------------
# 8 · Train
# ---------------------------------------------------------------------------
def train(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    train_ds: Dataset,
    eval_ds: Dataset,
) -> SFTTrainer:

    def formatting_func(examples: dict) -> list[str]:
        return [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in examples["messages"]
        ]

    sft_config = SFTConfig(
        output_dir=ADAPTER_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_seq_length=1024,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        dataset_text_field=None,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        formatting_func=formatting_func,
        tokenizer=tokenizer,
    )

    print("Starting training…")
    train_result = trainer.train()

    trainer.save_model(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"\nAdapter saved to: {ADAPTER_DIR}")

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    return trainer

# ---------------------------------------------------------------------------
# 9 · Loss curve
# ---------------------------------------------------------------------------
def plot_loss(trainer: SFTTrainer) -> None:
    history      = trainer.state.log_history
    train_steps  = [e["step"] for e in history if "loss" in e]
    train_losses = [e["loss"] for e in history if "loss" in e]
    eval_steps   = [e["step"] for e in history if "eval_loss" in e]
    eval_losses  = [e["eval_loss"] for e in history if "eval_loss" in e]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(train_steps, train_losses, label="Train loss", color="steelblue")
    if eval_losses:
        ax.plot(eval_steps, eval_losses, label="Eval loss", color="tomato", marker="o")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("lectureOS QLoRA — SFT Loss Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = os.path.join(ADAPTER_DIR, "loss_curve.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved to: {out}")
    print(f"Final train loss : {train_losses[-1]:.4f}")
    if eval_losses:
        print(f"Final eval loss  : {eval_losses[-1]:.4f}")

# ---------------------------------------------------------------------------
# 10 · Inference test
# ---------------------------------------------------------------------------
def run_inference_test(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> None:
    inf_model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    inf_model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": TEST_QUESTION},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = inf_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    response_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    response     = tokenizer.decode(response_ids, skip_special_tokens=True)

    print("\n" + "=" * 60)
    print(f"Q: {TEST_QUESTION}")
    print("-" * 60)
    print(f"A: {response}")
    print("=" * 60)

# ---------------------------------------------------------------------------
# 11 · Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    check_gpu()
    train_ds, eval_ds   = load_dataset()
    model, tokenizer    = load_model()
    model               = apply_lora(model)
    trainer             = train(model, tokenizer, train_ds, eval_ds)
    plot_loss(trainer)
    run_inference_test(model, tokenizer)
    print("\nDone.")


if __name__ == "__main__":
    main()
