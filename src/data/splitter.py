"""
Data splitter for train/val/test splits.
"""

import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def split_data(
    data: List,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
    shuffle: bool = True,
) -> Tuple[List, List, List]:
    """
    Split data into train/val/test sets.
    
    Args:
        data: List of data samples
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed
        shuffle: Whether to shuffle
        
    Returns:
        Tuple of (train, val, test)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # First split: train vs (val + test)
    train_data, temp_data = train_test_split(
        data,
        train_size=train_ratio,
        random_state=random_state,
        shuffle=shuffle,
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(
        temp_data,
        train_size=val_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    
    logger.info(f"Split data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    return train_data, val_data, test_data


def stratified_split(
    data: List,
    labels: List,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> Tuple[Tuple[List, List], Tuple[List, List], Tuple[List, List]]:
    """
    Stratified split maintaining label distribution.
    
    Args:
        data: List of data samples
        labels: List of labels
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed
        
    Returns:
        Tuple of ((train_data, train_labels), (val_data, val_labels), (test_data, test_labels))
    """
    # First split
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data, labels,
        train_size=train_ratio,
        stratify=labels,
        random_state=random_state,
    )
    
    # Second split
    val_size = val_ratio / (val_ratio + test_ratio)
    val_data, test_data, val_labels, test_labels = train_test_split(
        temp_data, temp_labels,
        train_size=val_size,
        stratify=temp_labels,
        random_state=random_state,
    )
    
    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)
