"""
Metrics utilities for evaluating models and molecules.
"""

from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================
# Regression Metrics
# ============================================

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {'r2': 0.0, 'rmse': float('inf'), 'mae': float('inf'), 'pearson': 0.0}
    
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
    }
    
    # Pearson correlation
    if len(y_true) > 1:
        pearson_corr, _ = stats.pearsonr(y_true, y_pred)
        metrics['pearson'] = pearson_corr
    else:
        metrics['pearson'] = 0.0
    
    return metrics


# ============================================
# Classification Metrics
# ============================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = 'binary'
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for AUC)
        average: Averaging method for multi-class
        
    Returns:
        Dictionary of metrics
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    if y_prob is not None:
        y_prob = np.array(y_prob)
        try:
            if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                metrics['auroc'] = roc_auc_score(y_true, y_prob[:, 1])
                metrics['auprc'] = average_precision_score(y_true, y_prob[:, 1])
            else:
                y_prob = y_prob.flatten()
                metrics['auroc'] = roc_auc_score(y_true, y_prob)
                metrics['auprc'] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics['auroc'] = 0.0
            metrics['auprc'] = 0.0
    
    return metrics


# ============================================
# Molecular Generation Metrics
# ============================================

def compute_validity(smiles_list: List[str]) -> float:
    """
    Compute validity of generated SMILES.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Validity ratio (0-1)
    """
    from rdkit import Chem
    
    valid_count = 0
    for smiles in smiles_list:
        if smiles and Chem.MolFromSmiles(smiles) is not None:
            valid_count += 1
    
    return valid_count / len(smiles_list) if smiles_list else 0.0


def compute_uniqueness(smiles_list: List[str]) -> float:
    """
    Compute uniqueness of generated SMILES.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Uniqueness ratio (0-1)
    """
    from rdkit import Chem
    
    # Canonicalize valid SMILES
    canonical_smiles = set()
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is not None:
            canonical = Chem.MolToSmiles(mol, canonical=True)
            canonical_smiles.add(canonical)
    
    valid_count = sum(1 for s in smiles_list if s and Chem.MolFromSmiles(s) is not None)
    
    return len(canonical_smiles) / valid_count if valid_count > 0 else 0.0


def compute_novelty(
    generated_smiles: List[str],
    reference_smiles: Set[str]
) -> float:
    """
    Compute novelty of generated SMILES against reference set.
    
    Args:
        generated_smiles: List of generated SMILES
        reference_smiles: Set of reference SMILES (training data)
        
    Returns:
        Novelty ratio (0-1)
    """
    from rdkit import Chem
    
    # Canonicalize reference SMILES
    canonical_reference = set()
    for smiles in reference_smiles:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is not None:
            canonical = Chem.MolToSmiles(mol, canonical=True)
            canonical_reference.add(canonical)
    
    novel_count = 0
    valid_count = 0
    
    for smiles in generated_smiles:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is not None:
            valid_count += 1
            canonical = Chem.MolToSmiles(mol, canonical=True)
            if canonical not in canonical_reference:
                novel_count += 1
    
    return novel_count / valid_count if valid_count > 0 else 0.0


def compute_diversity(smiles_list: List[str]) -> float:
    """
    Compute internal diversity using Tanimoto similarity.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Average pairwise Tanimoto distance (0-1)
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    
    # Generate fingerprints for valid molecules
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
    
    if len(fps) < 2:
        return 0.0
    
    # Compute pairwise Tanimoto distances
    distances = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            distances.append(1 - similarity)
    
    return np.mean(distances) if distances else 0.0


def compute_all_generation_metrics(
    generated_smiles: List[str],
    reference_smiles: Optional[Set[str]] = None
) -> Dict[str, float]:
    """
    Compute all molecular generation metrics.
    
    Args:
        generated_smiles: List of generated SMILES
        reference_smiles: Optional reference set for novelty
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {
        'validity': compute_validity(generated_smiles),
        'uniqueness': compute_uniqueness(generated_smiles),
        'diversity': compute_diversity(generated_smiles),
    }
    
    if reference_smiles:
        metrics['novelty'] = compute_novelty(generated_smiles, reference_smiles)
    
    return metrics


# ============================================
# Property Metrics
# ============================================

def compute_property_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute statistics for a property distribution.
    
    Args:
        values: List of property values
        
    Returns:
        Dictionary of statistics
    """
    values = [v for v in values if v is not None and not np.isnan(v)]
    
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
    
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values),
    }


def compute_property_coverage(
    generated_values: List[float],
    reference_values: List[float],
    n_bins: int = 20
) -> float:
    """
    Compute how well generated values cover the reference distribution.
    
    Args:
        generated_values: Generated property values
        reference_values: Reference property values
        n_bins: Number of bins for histogram
        
    Returns:
        Coverage ratio (0-1)
    """
    generated_values = [v for v in generated_values if v is not None and not np.isnan(v)]
    reference_values = [v for v in reference_values if v is not None and not np.isnan(v)]
    
    if not generated_values or not reference_values:
        return 0.0
    
    # Create bins from reference
    min_val = min(reference_values)
    max_val = max(reference_values)
    bins = np.linspace(min_val, max_val, n_bins + 1)
    
    # Count reference bins
    ref_hist, _ = np.histogram(reference_values, bins=bins)
    ref_occupied = ref_hist > 0
    
    # Count generated bins
    gen_hist, _ = np.histogram(generated_values, bins=bins)
    gen_occupied = gen_hist > 0
    
    # Compute coverage
    covered = np.sum(ref_occupied & gen_occupied)
    total = np.sum(ref_occupied)
    
    return covered / total if total > 0 else 0.0


# ============================================
# Docking Metrics
# ============================================

def compute_docking_statistics(scores: List[float]) -> Dict[str, float]:
    """
    Compute statistics for docking scores.
    
    Args:
        scores: List of docking scores (more negative = better)
        
    Returns:
        Dictionary of statistics
    """
    scores = [s for s in scores if s is not None and not np.isnan(s)]
    
    if not scores:
        return {'mean': 0.0, 'best': 0.0, 'worst': 0.0, 'hit_rate': 0.0}
    
    threshold = -6.0  # Typical hit threshold in kcal/mol
    
    return {
        'mean': np.mean(scores),
        'best': np.min(scores),
        'worst': np.max(scores),
        'std': np.std(scores),
        'hit_rate': np.mean([s < threshold for s in scores]),
    }
