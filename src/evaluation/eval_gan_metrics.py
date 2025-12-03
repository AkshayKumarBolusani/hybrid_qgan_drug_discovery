"""GAN evaluation metrics."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.metrics_utils import (
    compute_validity,
    compute_uniqueness,
    compute_novelty,
    compute_diversity,
    compute_all_generation_metrics
)

def evaluate_gan_generation(generated_smiles, reference_smiles=None):
    """Evaluate GAN-generated molecules."""
    metrics = compute_all_generation_metrics(generated_smiles, reference_smiles)
    print("\n=== GAN Generation Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    return metrics

if __name__ == '__main__':
    # Example
    test_smiles = ['C', 'CC', 'CCC', 'CO', 'CCO']
    evaluate_gan_generation(test_smiles)
