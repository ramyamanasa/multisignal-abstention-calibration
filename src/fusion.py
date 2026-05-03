"""
Module: fusion.py
Owner: Person B (Fusion + Evaluation)
Responsibility: Meta-classifier training, calibration, abstention policy.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import pickle


FEATURE_COLS = [
    "mean_entropy",
    "max_entropy",
    "entity_entropy",
    "semantic_inconsistency",
    "cross_model_disagreement",
]


def load_features(path: str = "../data/processed/features.csv"):
    df = pd.read_csv(path)
    df["cross_model_disagreement"] = df["cross_model_disagreement"].clip(0, 1)
    X = df[FEATURE_COLS].values
    y = df["is_hallucination"].values
    print(f"Loaded {len(df)} examples. Labels: {dict(pd.Series(y).value_counts())}")
    return X, y, df


def bootstrap_auroc_ci(y_true, y_prob, n_bootstrap: int = 1000, seed: int = 42):
    """Return (auroc, ci_low, ci_high) with 95% bootstrap CI."""
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, yp))
    aucs = np.array(aucs)
    auroc = float(roc_auc_score(y_true, y_prob))
    return auroc, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def select_threshold(y_val, probs_val, target_coverage: float = 0.5) -> dict:
    """
    Find the threshold on the validation set whose coverage is closest to
    target_coverage; break ties by maximising accuracy on answered questions.
    """
    from evaluation import compute_coverage_accuracy
    y_val     = np.array(y_val)
    probs_val = np.array(probs_val)
    best      = None
    best_dist = float("inf")
    for t in np.linspace(0.01, 0.99, 990):
        r    = compute_coverage_accuracy(y_val, probs_val, threshold=float(t))
        dist = abs(r["coverage"] - target_coverage)
        if best is None or dist < best_dist or (
            abs(dist - best_dist) < 1e-6 and r["accuracy"] > best["accuracy_on_answered"]
        ):
            best_dist = dist
            best = {
                "target_coverage":    round(target_coverage, 2),
                "threshold":          round(float(t), 4),
                "coverage":           round(r["coverage"], 4),
                "accuracy_on_answered": round(r["accuracy"], 4),
            }
    return best


def permutation_test_auroc(
    y_true, probs_fusion, probs_single, n_permutations: int = 1000, seed: int = 42
) -> float:
    """
    One-sided paired permutation test.
    H0: AUROC(single) >= AUROC(fusion).
    Returns p-value; low value means fusion is significantly better than the
    single-signal baseline.
    """
    rng          = np.random.default_rng(seed)
    y_true       = np.array(y_true)
    probs_fusion = np.array(probs_fusion)
    probs_single = np.array(probs_single)
    observed_delta = (
        roc_auc_score(y_true, probs_fusion) - roc_auc_score(y_true, probs_single)
    )
    n = len(y_true)
    extreme = 0
    for _ in range(n_permutations):
        swap        = rng.random(n) < 0.5
        perm_fusion = np.where(swap, probs_single, probs_fusion)
        perm_single = np.where(swap, probs_fusion, probs_single)
        perm_delta  = (
            roc_auc_score(y_true, perm_fusion) - roc_auc_score(y_true, perm_single)
        )
        if perm_delta >= observed_delta:
            extreme += 1
    return extreme / n_permutations


def train_classifier(X_train, y_train):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("clf",     LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    calibrated = CalibratedClassifierCV(pipeline, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    print("Classifier trained and calibrated.")
    return calibrated


def predict_with_abstention(clf, X, threshold: float = 0.5):
    probs     = clf.predict_proba(X)[:, 1]
    decisions = ["abstain" if p >= threshold else "answer" for p in probs]
    return decisions, probs.tolist()


def save_classifier(clf, path: str = "../models/meta_clf.pkl"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Classifier saved to {path}")


def load_classifier(path: str = "../models/meta_clf.pkl"):
    with open(path, "rb") as f:
        clf = pickle.load(f)
    print(f"Classifier loaded from {path}")
    return clf


def run_experiment(
    features_path: str = "../data/processed/features.csv",
    output_path:   str = "../experiments/exp002_fusion_model.json",
    test_size:     float = 0.2,
    val_size:      float = 0.2,
    threshold:     float = 0.5,
    n_bootstrap:   int   = 1000,
):
    from evaluation import (
        compute_auroc,
        compute_ece,
        compute_coverage_accuracy,
        compute_operating_points,
        plot_reliability_diagram,
        plot_coverage_accuracy_curve,
        log_experiment,
    )

    X, y, df = load_features(features_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size / (1 - test_size),
        random_state=42, stratify=y_train
    )

    print(f"\nSplit: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    print("\nTraining fusion model (all 5 features)...")
    clf = train_classifier(X_train, y_train)
    save_classifier(clf)

    _, probs_test = predict_with_abstention(clf, X_test, threshold=threshold)
    probs_test    = np.array(probs_test)
    _, probs_val_list = predict_with_abstention(clf, X_val, threshold=threshold)
    probs_val         = np.array(probs_val_list)

    auroc, ci_lo, ci_hi = bootstrap_auroc_ci(y_test, probs_test, n_bootstrap)
    ece     = compute_ece(y_test, probs_test)
    cov_acc = compute_coverage_accuracy(y_test, probs_test, threshold=threshold)

    print(f"\nFusion model results:")
    print(f"  AUROC:    {auroc:.4f}  95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  ECE:      {ece:.4f}")
    print(f"  Coverage: {cov_acc['coverage']:.3f}")
    print(f"  Accuracy: {cov_acc['accuracy']:.3f}")

    plot_reliability_diagram(
        y_test, probs_test,
        save_path="../data/processed/reliability_diagram.png"
    )
    plot_coverage_accuracy_curve(
        y_test, probs_test,
        save_path="../data/processed/coverage_accuracy_curve.png"
    )

    print("\nOperating points across fixed thresholds (test set):")
    operating_points = compute_operating_points(y_test, probs_test)

    print("\nCoverage-aware threshold selection (validation set):")
    threshold_selection = []
    for tc in [0.3, 0.5, 0.7]:
        sel = select_threshold(y_val, probs_val, target_coverage=tc)
        threshold_selection.append(sel)
        print(
            f"  target={tc:.1f}  →  threshold={sel['threshold']:.4f}  "
            f"coverage={sel['coverage']:.4f}  "
            f"accuracy={sel['accuracy_on_answered']:.4f}"
        )

    print("\nRunning ablation (with 95% bootstrap CIs and permutation tests)...")
    ablation_results = {}

    subsets = {
        "mean_entropy_only":   [0],
        "max_entropy_only":    [1],
        "entity_entropy_only": [2],
        "consistency_only":    [3],
        "disagreement_only":   [4],
        "entropy_all":         [0, 1, 2],
        "signals_2_and_3":     [3, 4],
        "all_5_features":      [0, 1, 2, 3, 4],
    }

    for name, indices in subsets.items():
        X_tr     = X_train[:, indices]
        X_te     = X_test[:,  indices]
        clf_sub  = train_classifier(X_tr, y_train)
        _, probs = predict_with_abstention(clf_sub, X_te)
        probs_arr = np.array(probs)
        auc, ci_lo_sub, ci_hi_sub = bootstrap_auroc_ci(
            y_test, probs_arr, n_bootstrap
        )
        ece_sub = compute_ece(y_test, probs_arr)

        if name == "all_5_features":
            p_val = None  # same model as fusion — skip test
        else:
            p_val = permutation_test_auroc(
                y_test, probs_test, probs_arr, n_permutations=n_bootstrap
            )

        ablation_results[name] = {
            "auroc":             round(auc, 4),
            "auroc_ci":          [round(ci_lo_sub, 4), round(ci_hi_sub, 4)],
            "ece":               round(ece_sub, 4),
            "p_value_vs_fusion": round(p_val, 4) if p_val is not None else None,
        }
        pval_str = f"  p={p_val:.4f}" if p_val is not None else "  p=N/A (same model)"
        print(
            f"  {name:<30} AUROC: {auc:.4f} "
            f"[{ci_lo_sub:.4f}, {ci_hi_sub:.4f}]  ECE: {ece_sub:.4f}{pval_str}"
        )

    results = {
        "fusion_model": {
            "auroc":    round(auroc, 4),
            "auroc_ci": [round(ci_lo, 4), round(ci_hi, 4)],
            "ece":      round(ece, 4),
            "coverage": round(cov_acc["coverage"], 3),
            "accuracy": round(cov_acc["accuracy"], 3),
        },
        "operating_points":    operating_points,
        "threshold_selection": threshold_selection,
        "ablation":            ablation_results,
    }

    log_experiment(
        exp_id="exp002",
        description="Fusion model with all 5 features, HaluEval",
        config={"dataset": "HaluEval", "threshold": threshold, "n_bootstrap": n_bootstrap},
        metrics=results,
        output_path=output_path,
    )

    return clf, results


if __name__ == "__main__":
    clf, results = run_experiment()
