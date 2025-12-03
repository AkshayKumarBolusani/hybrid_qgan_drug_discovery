#!/usr/bin/env python3
"""Preprocess all datasets for training."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data import (
    load_molecular_dataset,
    create_tokenizer_from_smiles,
    GraphFeaturizer,
    split_data
)
from src.utils import get_logger

logger = get_logger(__name__)

def preprocess_all():
    """Preprocess all datasets."""
    logger.info("=" * 60)
    logger.info("PREPROCESSING ALL DATASETS")
    logger.info("=" * 60)
    
    # Load QM9
    logger.info("\n[1/4] Loading QM9 dataset...")
    smiles_qm9, props_qm9 = load_molecular_dataset('qm9', max_samples=1000)
    logger.info(f"✓ Loaded {len(smiles_qm9)} QM9 molecules")
    
    # Create tokenizer
    logger.info("\n[2/4] Building SMILES tokenizer...")
    tokenizer = create_tokenizer_from_smiles(
        smiles_qm9,
        vocab_file='data/processed/vocab.json',
        max_length=100
    )
    logger.info(f"✓ Vocabulary size: {tokenizer.vocab_size}")
    
    # Create graph featurizer
    logger.info("\n[3/4] Preparing graph representations...")
    featurizer = GraphFeaturizer(max_atoms=9)
    
    valid_count = 0
    for smiles in smiles_qm9[:100]:  # Test on subset
        graph = featurizer.smiles_to_graph(smiles)
        if graph:
            valid_count += 1
    
    logger.info(f"✓ Graph conversion success rate: {valid_count/100:.1%}")
    
    # Split data
    logger.info("\n[4/4] Creating train/val/test splits...")
    train, val, test = split_data(smiles_qm9, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    logger.info(f"✓ Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    
    # Save splits
    import pickle
    splits_file = Path('data/processed/splits.pkl')
    splits_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(splits_file, 'wb') as f:
        pickle.dump({'train': train, 'val': val, 'test': test}, f)
    
    logger.info(f"✓ Saved splits to {splits_file}")
    
    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING COMPLETE!")
    logger.info("=" * 60)

if __name__ == '__main__':
    preprocess_all()
