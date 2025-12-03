"""Toxicity evaluation."""
from src.utils.metrics_utils import compute_classification_metrics
import numpy as np

def evaluate_toxicity(model, X_test, y_test):
    """Evaluate toxicity model."""
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    results = {}
    for task_idx in range(y_test.shape[1]):
        metrics = compute_classification_metrics(
            y_test[:, task_idx],
            y_pred_binary[:, task_idx],
            y_pred[:, task_idx]
        )
        results[f'task_{task_idx}'] = metrics
    
    return results
