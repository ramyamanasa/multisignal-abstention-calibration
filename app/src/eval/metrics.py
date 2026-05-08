"""Evaluation metrics for the CALAS selective-prediction pipeline."""

from __future__ import annotations

import math

from sklearn.metrics import roc_auc_score


def selective_accuracy(predictions: list[dict]) -> dict:
    """Compute accuracy, coverage, and abstention rate over a prediction set.

    Answered predictions are those where ``abstained`` is ``False``.
    Accuracy is measured by case-insensitive exact match between ``answer``
    and ``gold``.

    Args:
        predictions: List of dicts, each with keys:
            - ``answer`` (str): The model's answer string.
            - ``gold`` (str): The ground-truth answer string.
            - ``confidence`` (float): Confidence score in [0, 1].
            - ``abstained`` (bool): Whether the model chose to abstain.

    Returns:
        A dict with:
            - ``accuracy`` (float): Fraction of answered questions that are
              correct. ``0.0`` when all questions were abstained.
            - ``coverage`` (float): Fraction of questions answered (not
              abstained). ``0.0`` when ``predictions`` is empty.
            - ``abstention_rate`` (float): Fraction of questions abstained.
              ``0.0`` when ``predictions`` is empty.

    Raises:
        ValueError: If ``predictions`` contains a dict missing required keys.
    """
    if not predictions:
        return {"accuracy": 0.0, "coverage": 0.0, "abstention_rate": 0.0}

    n = len(predictions)
    answered = [p for p in predictions if not p["abstained"]]
    correct = sum(
        1 for p in answered
        if p["answer"].strip().lower() == p["gold"].strip().lower()
    )

    coverage = len(answered) / n
    abstention_rate = 1.0 - coverage
    accuracy = correct / len(answered) if answered else 0.0

    return {
        "accuracy": round(accuracy, 6),
        "coverage": round(coverage, 6),
        "abstention_rate": round(abstention_rate, 6),
    }


def expected_calibration_error(
    predictions: list[dict],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) over confidence bins.

    Partitions predictions into ``n_bins`` equal-width bins over [0, 1] by
    confidence, then returns the weighted average of
    ``|mean_accuracy - mean_confidence|`` per bin.

    Only non-abstained predictions contribute to the ECE calculation.

    Args:
        predictions: List of dicts with keys ``answer``, ``gold``,
            ``confidence``, and ``abstained`` (same schema as
            :func:`selective_accuracy`).
        n_bins: Number of equal-width confidence bins. Defaults to ``10``.

    Returns:
        ECE as a float in [0, 1]. Returns ``0.0`` when there are no
        answered predictions.
    """
    answered = [p for p in predictions if not p["abstained"]]
    if not answered:
        return 0.0

    bin_width = 1.0 / n_bins
    bins: list[list[dict]] = [[] for _ in range(n_bins)]

    for p in answered:
        idx = min(int(p["confidence"] / bin_width), n_bins - 1)
        bins[idx].append(p)

    n = len(answered)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        bin_acc = sum(
            1 for p in bucket
            if p["answer"].strip().lower() == p["gold"].strip().lower()
        ) / len(bucket)
        bin_conf = sum(p["confidence"] for p in bucket) / len(bucket)
        ece += (len(bucket) / n) * abs(bin_acc - bin_conf)

    return round(ece, 6)


def auroc_abstention(predictions: list[dict]) -> float:
    """Compute AUROC treating confidence as a score for answer correctness.

    Considers only non-abstained predictions. Correctness is a binary label
    derived from case-insensitive exact match. Uses
    ``sklearn.metrics.roc_auc_score`` for the computation.

    Args:
        predictions: List of dicts with keys ``answer``, ``gold``,
            ``confidence``, and ``abstained`` (same schema as
            :func:`selective_accuracy`).

    Returns:
        AUROC as a float in [0, 1]. Returns ``0.5`` (random baseline) when
        there are fewer than two distinct correctness labels.
    """
    answered = [p for p in predictions if not p["abstained"]]
    labels = [
        int(p["answer"].strip().lower() == p["gold"].strip().lower())
        for p in answered
    ]
    scores = [p["confidence"] for p in answered]

    if len(set(labels)) < 2:
        return 0.5

    return float(round(roc_auc_score(labels, scores), 6))


def run_eval(predictions: list[dict]) -> dict:
    """Run all evaluation metrics and return a consolidated results dict.

    Args:
        predictions: List of dicts with keys ``answer``, ``gold``,
            ``confidence``, and ``abstained`` (same schema as
            :func:`selective_accuracy`).

    Returns:
        A dict with keys:
            - ``selective_accuracy`` (float)
            - ``coverage`` (float)
            - ``abstention_rate`` (float)
            - ``ece`` (float)
            - ``auroc`` (float)
    """
    sa = selective_accuracy(predictions)
    return {
        "selective_accuracy": sa["accuracy"],
        "coverage": sa["coverage"],
        "abstention_rate": sa["abstention_rate"],
        "ece": expected_calibration_error(predictions),
        "auroc": auroc_abstention(predictions),
    }
