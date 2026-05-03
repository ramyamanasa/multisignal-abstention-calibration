"""
Module: evaluation.py
Owner: Person B (Fusion + Evaluation)
Responsibility: All metrics and plots.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score


def compute_auroc(y_true, y_prob) -> float:
    return float(roc_auc_score(y_true, y_prob))


def compute_ece(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    bins   = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    n      = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece     += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_coverage_accuracy(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_true   = np.array(y_true)
    y_prob   = np.array(y_prob)
    answered = y_prob < threshold
    coverage = answered.mean()
    if answered.sum() == 0:
        return {"coverage": 0.0, "accuracy": 0.0}
    accuracy = (y_true[answered] == 0).mean()
    return {"coverage": float(coverage), "accuracy": float(accuracy)}


def compute_operating_points(
    y_true, y_prob, thresholds=None
) -> list:
    """
    Report coverage and accuracy at a fixed set of operating-point thresholds.
    Prints a formatted table and returns a list of dicts.
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    print(f"\n{'Threshold':>10}  {'Coverage':>10}  {'Acc@Answered':>14}")
    print("-" * 40)
    rows = []
    for t in thresholds:
        r = compute_coverage_accuracy(y_true, y_prob, threshold=t)
        rows.append({
            "threshold":            t,
            "coverage":             round(r["coverage"], 4),
            "accuracy_on_answered": round(r["accuracy"], 4),
        })
        print(f"{t:>10.2f}  {r['coverage']:>10.4f}  {r['accuracy']:>14.4f}")
    return rows


def plot_reliability_diagram(
    y_true, y_prob, n_bins: int = 10, save_path: str = None
):
    y_true      = np.array(y_true)
    y_prob      = np.array(y_prob)
    bins        = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accs    = []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_centers.append(y_prob[mask].mean())
        bin_accs.append(y_true[mask].mean())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(bin_centers, bin_accs, width=1/n_bins, align="center",
           alpha=0.6, color="steelblue", label="Model")
    ax.plot(bin_centers, bin_accs, "o-", color="steelblue")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of hallucinations")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Reliability diagram saved to {save_path}")
    plt.close()


def plot_coverage_accuracy_curve(
    y_true, y_prob, save_path: str = None
):
    y_true     = np.array(y_true)
    y_prob     = np.array(y_prob)
    thresholds = np.linspace(0.1, 0.9, 50)
    coverages  = []
    accuracies = []
    for t in thresholds:
        result = compute_coverage_accuracy(y_true, y_prob, threshold=t)
        coverages.append(result["coverage"])
        accuracies.append(result["accuracy"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(coverages, accuracies, "o-", color="steelblue", markersize=4)
    ax.set_xlabel("Coverage (fraction of questions answered)")
    ax.set_ylabel("Accuracy on answered questions")
    ax.set_title("Coverage vs Accuracy Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axhline(
        y=(y_true == 0).mean(),
        color="red", linestyle="--", label="Baseline (always answer)"
    )
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Coverage accuracy curve saved to {save_path}")
    plt.close()


def log_experiment(
    exp_id: str,
    description: str,
    config: dict,
    metrics: dict,
    notes: str = "",
    output_path: str = "../experiments/latest.json",
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "exp_id":      exp_id,
        "description": description,
        "config":      config,
        "metrics":     metrics,
        "notes":       notes,
    }
    with open(output_path, "w") as f:
        json.dump(entry, f, indent=2)
    print(f"Experiment logged to {output_path}")