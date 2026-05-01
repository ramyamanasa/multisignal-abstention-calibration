#!/usr/bin/env bash
# run_eval.sh
# Full evaluation pipeline for multisignal-abstention-calibration.
# Run from the project root: bash run_eval.sh
# Outputs all metrics to stdout and saves plots + JSON to data/processed/ and experiments/.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=============================================="
echo " Multi-Signal Abstention Calibration: Eval"
echo "=============================================="

# Activate venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Activated venv: $(python3 --version)"
else
    echo "ERROR: venv not found. Run: python3 -m venv venv && pip install -r requirements.txt"
    exit 1
fi

# Verify features.csv exists
if [ ! -f "data/processed/features.csv" ]; then
    echo "ERROR: data/processed/features.csv not found."
    echo "Run src/data.py first to generate features (takes ~2 hours)."
    exit 1
fi

echo ""
echo "----------------------------------------------"
echo "Step 1: Train fusion model + ablation (HaluEval)"
echo "----------------------------------------------"
cd src
python3 fusion.py
cd "$PROJECT_ROOT"

echo ""
echo "----------------------------------------------"
echo "Step 2: OOD evaluation (TriviaQA)"
echo "----------------------------------------------"
if [ -f "data/processed/ood_features.csv" ]; then
    echo "OOD features already exist. Running evaluation only..."
    cd src
    python3 - <<'PYEOF'
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(".")))
from ood_eval import evaluate_ood
evaluate_ood()
PYEOF
    cd "$PROJECT_ROOT"
else
    echo "OOD features not found. Running full OOD pipeline (~45 min)..."
    cd src
    python3 ood_eval.py
    cd "$PROJECT_ROOT"
fi

echo ""
echo "----------------------------------------------"
echo "Step 3: Print results summary"
echo "----------------------------------------------"
python3 - <<'PYEOF'
import json
from pathlib import Path

def load(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())

halueval = load("experiments/exp002_fusion_model.json")
ood      = load("experiments/exp003_ood_eval.json")

print("\n========== RESULTS SUMMARY ==========")

if halueval:
    fm = halueval["metrics"]["fusion_model"]
    print(f"\nHaluEval (in-distribution, n=100 test):")
    print(f"  AUROC:    {fm['auroc']:.4f}  95% CI: {fm['auroc_ci']}")
    print(f"  ECE:      {fm['ece']:.4f}")
    print(f"  Coverage: {fm['coverage']:.3f}  Accuracy: {fm['accuracy']:.3f}")

    print(f"\nAblation table:")
    print(f"  {'Subset':<30} {'AUROC':>8}  {'95% CI':>20}  {'ECE':>8}")
    print(f"  {'-'*30} {'-'*8}  {'-'*20}  {'-'*8}")
    for name, row in halueval["metrics"]["ablation"].items():
        ci = row.get("auroc_ci", ["?", "?"])
        ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
        print(f"  {name:<30} {row['auroc']:>8.4f}  {ci_str:>20}  {row['ece']:>8.4f}")

if ood:
    print(f"\nTriviaQA OOD (n={ood['n']}):")
    print(f"  AUROC:  {ood['auroc']:.4f}  95% CI: {ood['auroc_ci']}")
    print(f"  ECE:    {ood['ece']:.4f}")

print("\nPlots saved to data/processed/")
print("=====================================\n")
PYEOF

echo ""
echo "Evaluation complete."
