"""Training script for QSAR models."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from src.models.qsar.qsar_sklearn import QSARSklearnModel
from src.data import load_molecular_dataset
from src.features import batch_smiles_to_fingerprints, calculate_descriptors_batch
from src.utils import get_logger, compute_regression_metrics

logger = get_logger(__name__)

def train_qsar(model_type='rf', dataset='qm9'):
    """Train QSAR model."""
    logger.info(f"Training {model_type} QSAR model on {dataset}...")
    
    # Load data
    smiles, properties = load_molecular_dataset(dataset, max_samples=1000)
    
    # Prepare features
    logger.info("Extracting features...")
    fps = batch_smiles_to_fingerprints(smiles, fp_type='morgan', n_bits=2048)
    descriptors = calculate_descriptors_batch(smiles)
    desc_array = np.array([[d.get('qed', 0), d.get('logp', 0), d.get('mol_weight', 0)] for d in descriptors])
    X = np.hstack([fps, desc_array])
    
    # Use first property as target
    if len(properties) > 0 and len(properties.columns) > 0:
        y = properties.iloc[:, 0].values
    else:
        y = np.random.randn(len(smiles))  # Mock target
    
    # Train/test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    model = QSARSklearnModel(model_type=model_type)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = compute_regression_metrics(y_test, y_pred)
    logger.info(f"Test metrics: {metrics}")
    
    # Save model
    model_path = Path('experiments/checkpoints') / f'qsar_{model_type}.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    return model, metrics

if __name__ == '__main__':
    train_qsar(model_type='rf')
