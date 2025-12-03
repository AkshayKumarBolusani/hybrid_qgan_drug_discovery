"""QSAR evaluation."""
from src.utils.metrics_utils import compute_regression_metrics

def evaluate_qsar(model, X_test, y_test):
    """Evaluate QSAR model."""
    y_pred = model.predict(X_test)
    metrics = compute_regression_metrics(y_test, y_pred)
    print("\n=== QSAR Evaluation ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    return metrics
