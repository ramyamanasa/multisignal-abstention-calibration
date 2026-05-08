## Final V1 Baseline (after corpus fixes)

Sources: 6 files, 2964 chunks

| File | Description | Chunks |
|---|---|---|
| kCc8FmEb1nY | Karpathy nanoGPT | 985 |
| zduSFxRajkE | Karpathy tokenization | 1141 |
| LWMzyfvuehA | CS224N transformers | 674 |
| hf_peft_lora | HF PEFT LoRA docs | 28 |
| lora_paper | LoRA paper abstract | 11 |
| hf_lora_config | HF LoRA API ref | 125 |

### Eval results (12 gold questions)

- Answerable (7): 6/7 correct/partial, 1 escalated (q06)
- Out-of-scope (5): 100% abstention rate
- Avg corpus_confidence — answerable: 0.7911
- Avg corpus_confidence — OOS: 0.6608
- Separation gap: 0.1292
- Abstention threshold: 0.74

### q06 notes

- Question: "What rank r controls in a LoRA adapter?"
- corpus_confidence: 0.7406 (passes 0.74 threshold ✓)
- fix: hf_lora_config source (125 chunks of LoRA API reference) pushed q06 from 0.7343 → 0.7406
- verify_node escalation on q06: expected — thin but genuine coverage; answer elaborates beyond chunk text
- Threshold remains: 0.74

## Gold QA Baseline (23 questions)
Date: 2026-04-22
Answerable accuracy: 11/13 (85%)
Wrong answers: 0
Incorrect abstentions: 2
OOS abstention rate: 10/10 (100%)
False answers on OOS: 0
Avg confidence answerable: 0.731
Avg confidence OOS: 0.662
Separation gap: 0.069
