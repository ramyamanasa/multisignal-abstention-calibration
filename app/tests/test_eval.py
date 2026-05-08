"""Tests for src/eval/metrics.py — evaluation metrics for the CALAS pipeline.

Run with:
    pytest tests/test_eval.py -v
"""

from __future__ import annotations

import pytest

from src.eval.metrics import (
    auroc_abstention,
    expected_calibration_error,
    run_eval,
    selective_accuracy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pred(
    answer: str,
    gold: str,
    confidence: float,
    abstained: bool = False,
) -> dict:
    """Construct a prediction dict for use in tests."""
    return {"answer": answer, "gold": gold, "confidence": confidence, "abstained": abstained}


# ---------------------------------------------------------------------------
# selective_accuracy
# ---------------------------------------------------------------------------

class TestSelectiveAccuracy:
    """Unit tests for selective_accuracy."""

    def test_all_correct_answered(self):
        """Perfect answers give accuracy=1.0, coverage=1.0, abstention_rate=0.0."""
        preds = [
            _pred("paris", "paris", 0.9),
            _pred("berlin", "berlin", 0.8),
        ]
        result = selective_accuracy(preds)
        assert result["accuracy"] == 1.0
        assert result["coverage"] == 1.0
        assert result["abstention_rate"] == 0.0

    def test_half_correct(self):
        """Two answered, one correct → accuracy=0.5."""
        preds = [
            _pred("paris", "paris", 0.9),
            _pred("wrong", "berlin", 0.7),
        ]
        result = selective_accuracy(preds)
        assert result["accuracy"] == pytest.approx(0.5)

    def test_coverage_and_abstention_rate_sum_to_one(self):
        """coverage + abstention_rate must always equal 1.0."""
        preds = [
            _pred("paris", "paris", 0.9),
            _pred("", "berlin", 0.2, abstained=True),
            _pred("rome", "rome", 0.85),
            _pred("", "tokyo", 0.1, abstained=True),
        ]
        result = selective_accuracy(preds)
        assert result["coverage"] + result["abstention_rate"] == pytest.approx(1.0)

    def test_all_abstained_returns_zero_accuracy_zero_coverage(self):
        """All abstentions → accuracy=0.0, coverage=0.0, abstention_rate=1.0."""
        preds = [
            _pred("", "paris", 0.1, abstained=True),
            _pred("", "berlin", 0.05, abstained=True),
        ]
        result = selective_accuracy(preds)
        assert result["accuracy"] == 0.0
        assert result["coverage"] == 0.0
        assert result["abstention_rate"] == 1.0

    def test_empty_predictions(self):
        """Empty list returns all zeros."""
        result = selective_accuracy([])
        assert result == {"accuracy": 0.0, "coverage": 0.0, "abstention_rate": 0.0}

    def test_case_insensitive_match(self):
        """Answer matching is case-insensitive."""
        preds = [_pred("PARIS", "paris", 0.9)]
        result = selective_accuracy(preds)
        assert result["accuracy"] == 1.0

    def test_abstention_rate_value(self):
        """2 of 4 abstained → abstention_rate=0.5."""
        preds = [
            _pred("paris", "paris", 0.9),
            _pred("rome", "rome", 0.8),
            _pred("", "berlin", 0.1, abstained=True),
            _pred("", "tokyo", 0.1, abstained=True),
        ]
        result = selective_accuracy(preds)
        assert result["abstention_rate"] == pytest.approx(0.5)
        assert result["coverage"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------

class TestExpectedCalibrationError:
    """Unit tests for expected_calibration_error."""

    def test_perfect_calibration_returns_zero(self):
        """ECE must be 0.0 when accuracy equals confidence in every bin.

        One correct prediction at confidence=1.0 and one incorrect at
        confidence=0.0 — each sits in its own bin with zero gap.
        """
        preds = [
            _pred("paris", "paris", 1.0),   # bin 9: acc=1.0, conf=1.0 → gap=0
            _pred("wrong", "berlin", 0.0),  # bin 0: acc=0.0, conf=0.0 → gap=0
        ]
        assert expected_calibration_error(preds) == pytest.approx(0.0)

    def test_worst_case_calibration(self):
        """ECE > 0 when confidence and accuracy are maximally misaligned."""
        preds = [
            _pred("wrong", "correct", 1.0),  # high confidence, wrong
            _pred("right", "right", 0.0),    # zero confidence, right
        ]
        ece = expected_calibration_error(preds)
        assert ece > 0.0

    def test_abstained_predictions_excluded(self):
        """Abstained predictions must not affect ECE."""
        answered = [_pred("paris", "paris", 1.0)]
        with_abstained = answered + [_pred("", "x", 0.5, abstained=True)]
        assert expected_calibration_error(answered) == expected_calibration_error(with_abstained)

    def test_empty_predictions_returns_zero(self):
        """ECE must be 0.0 for an empty prediction list."""
        assert expected_calibration_error([]) == 0.0

    def test_all_abstained_returns_zero(self):
        """ECE must be 0.0 when every prediction abstained."""
        preds = [_pred("", "gold", 0.5, abstained=True)]
        assert expected_calibration_error(preds) == 0.0

    def test_returns_float(self):
        """ECE must be a float value."""
        preds = [_pred("a", "a", 0.8)]
        assert isinstance(expected_calibration_error(preds), float)


# ---------------------------------------------------------------------------
# auroc_abstention
# ---------------------------------------------------------------------------

class TestAurocAbstention:
    """Unit tests for auroc_abstention."""

    def test_perfect_ranking_returns_one(self):
        """AUROC must be 1.0 when correct answers always have higher confidence."""
        preds = [
            _pred("right", "right", 0.9),
            _pred("right", "right", 0.8),
            _pred("wrong", "gold", 0.3),
            _pred("wrong", "gold", 0.2),
        ]
        assert auroc_abstention(preds) == pytest.approx(1.0)

    def test_inverted_ranking_returns_zero(self):
        """AUROC must be 0.0 when incorrect answers always have higher confidence."""
        preds = [
            _pred("wrong", "gold", 0.9),
            _pred("wrong", "gold", 0.8),
            _pred("right", "right", 0.3),
            _pred("right", "right", 0.2),
        ]
        assert auroc_abstention(preds) == pytest.approx(0.0)

    def test_all_correct_returns_half(self):
        """Only one label class present → degenerate case returns 0.5."""
        preds = [
            _pred("paris", "paris", 0.9),
            _pred("berlin", "berlin", 0.7),
        ]
        assert auroc_abstention(preds) == pytest.approx(0.5)

    def test_abstained_predictions_excluded(self):
        """Abstained predictions must not influence AUROC."""
        preds = [
            _pred("right", "right", 0.9),
            _pred("wrong", "gold", 0.2),
            _pred("", "x", 0.99, abstained=True),
        ]
        preds_no_abstain = [p for p in preds if not p["abstained"]]
        assert auroc_abstention(preds) == auroc_abstention(preds_no_abstain)

    def test_returns_float_in_unit_interval(self):
        """AUROC must be a float in [0, 1]."""
        preds = [
            _pred("right", "right", 0.8),
            _pred("wrong", "gold", 0.4),
        ]
        result = auroc_abstention(preds)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# run_eval
# ---------------------------------------------------------------------------

class TestRunEval:
    """Unit tests for run_eval."""

    def test_returns_all_five_keys(self):
        """run_eval must return a dict containing all five expected keys."""
        preds = [
            _pred("right", "right", 0.9),
            _pred("wrong", "gold", 0.3),
            _pred("", "x", 0.1, abstained=True),
        ]
        result = run_eval(preds)
        assert set(result.keys()) == {
            "selective_accuracy",
            "coverage",
            "abstention_rate",
            "ece",
            "auroc",
        }

    def test_values_match_individual_functions(self):
        """run_eval results must equal individually called metrics."""
        preds = [
            _pred("right", "right", 0.9),
            _pred("wrong", "gold", 0.2),
            _pred("", "x", 0.05, abstained=True),
        ]
        result = run_eval(preds)
        sa = selective_accuracy(preds)

        assert result["selective_accuracy"] == sa["accuracy"]
        assert result["coverage"] == sa["coverage"]
        assert result["abstention_rate"] == sa["abstention_rate"]
        assert result["ece"] == expected_calibration_error(preds)
        assert result["auroc"] == auroc_abstention(preds)

    def test_all_values_are_floats(self):
        """Every value in the run_eval output must be a float."""
        preds = [_pred("a", "a", 0.8), _pred("b", "c", 0.4)]
        result = run_eval(preds)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} is not a float"
