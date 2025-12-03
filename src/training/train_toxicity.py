"""Training script for toxicity models."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from src.models.toxicity.tox_classifiers import SKLearnToxicityModel
from src.models.toxicity.tox_preprocessing import prepare_toxicity_features
from src.data import load_molecular_dataset
from src.utils import get_logger

logger = get_logger(__name__)

def train_toxicity():
    """Train toxicity model."""
    logger.info("Training toxicity model...")
    
    # Load data
    smiles, _ = load_molecular_dataset('qm9', max_samples=500)
    
    # Prepare features
    X = prepare_toxicity_features(smiles[:500])
    
    # Mock labels (12 tasks)
    y = np.random.randint(0, 2, size=(len(X), 12))
    
    # Split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train
    model = SKLearnToxicityModel(num_tasks=12)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    logger.info("Training complete!")
    
    return model

if __name__ == '__main__':
    train_toxicity()
