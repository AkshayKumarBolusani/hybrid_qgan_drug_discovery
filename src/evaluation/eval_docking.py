"""Docking evaluation."""
from src.utils.metrics_utils import compute_docking_statistics

def evaluate_docking(docking_scores):
    """Evaluate docking results."""
    stats = compute_docking_statistics(docking_scores)
    print("\n=== Docking Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")
    return stats
