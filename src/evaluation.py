"""
Module: evaluation.py
Owner: Person B (Fusion + Evaluation)
Responsibility: All metrics and plots.
"""

def compute_auroc(y_true, y_prob) -> float:
    raise NotImplementedError

def compute_ece(y_true, y_prob, n_bins: int = 10) -> float:
    raise NotImplementedError

def plot_reliability_diagram(y_true, y_prob, n_bins: int = 10, save_path: str = None):
    raise NotImplementedError

def plot_coverage_accuracy_curve(y_true, y_prob, save_path: str = None):
    raise NotImplementedError

def log_experiment(exp_id: str, description: str, config: dict, metrics: dict, notes: str = ""):
    raise NotImplementedError
