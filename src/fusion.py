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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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


def train_classifier(X_train, y_train):
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
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
):
    from evaluation import (
        compute_auroc,
        compute_ece,
        compute_coverage_accuracy,
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

    auroc   = compute_auroc(y_test, probs_test)
    ece     = compute_ece(y_test, probs_test)
    cov_acc = compute_coverage_accuracy(y_test, probs_test, threshold=threshold)

    print(f"\nFusion model results:")
    print(f"  AUROC:    {auroc:.4f}")
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

    print("\nRunning ablation...")
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
        auc      = compute_auroc(y_test, np.array(probs))
        ece_sub  = compute_ece(y_test,   np.array(probs))
        ablation_results[name] = {
            "auroc": round(auc, 4),
            "ece":   round(ece_sub, 4)
        }
        print(f"  {name:<30} AUROC: {auc:.4f}  ECE: {ece_sub:.4f}")

    results = {
        "fusion_model": {
            "auroc":    round(auroc, 4),
            "ece":      round(ece, 4),
            "coverage": round(cov_acc["coverage"], 3),
            "accuracy": round(cov_acc["accuracy"], 3),
        },
        "ablation": ablation_results,
    }

    log_experiment(
        exp_id="exp002",
        description="Fusion model with all 5 features, HaluEval 200q",
        config={"dataset": "HaluEval", "n_questions": 200, "threshold": threshold},
        metrics=results,
        output_path=output_path,
    )

    return clf, results


if __name__ == "__main__":
    clf, results = run_experiment()