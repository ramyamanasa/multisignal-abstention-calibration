"""
Module: fusion.py (lectureOS abstention)
Verbatim copy from multisignal-abstention-calibration/src/fusion.py
Only load_classifier and predict_with_abstention are used by pipeline.py.
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
