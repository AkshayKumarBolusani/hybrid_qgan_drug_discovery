"""
Molecular fingerprint utilities.
"""

import numpy as np
from typing import List, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import rdFingerprintGenerator as rdFP
from rdkit import DataStructs

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def smiles_to_morgan_fingerprint(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[np.ndarray]:
    """
    Convert SMILES to Morgan fingerprint.
    
    Args:
        smiles: SMILES string
        radius: Morgan fingerprint radius
        n_bits: Number of bits
        
    Returns:
        Fingerprint as numpy array or None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        # Prefer new RDKit fingerprint generator to avoid deprecation
        generator = rdFP.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return None


def smiles_to_rdkit_fingerprint(
    smiles: str,
    n_bits: int = 2048,
) -> Optional[np.ndarray]:
    """
    Convert SMILES to RDKit fingerprint.
    
    Args:
        smiles: SMILES string
        n_bits: Number of bits
        
    Returns:
        Fingerprint as numpy array or None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        generator = rdFP.GetRDKitFPGenerator(fpSize=n_bits)
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return None


def smiles_to_maccs_fingerprint(smiles: str) -> Optional[np.ndarray]:
    """
    Convert SMILES to MACCS keys fingerprint.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Fingerprint as numpy array (167 bits) or None
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    try:
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros((167,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return None


def calculate_tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Calculate Tanimoto similarity between two fingerprints.
    
    Args:
        fp1: First fingerprint
        fp2: Second fingerprint
        
    Returns:
        Tanimoto similarity (0-1)
    """
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_dice_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """
    Calculate Dice similarity between two fingerprints.
    
    Args:
        fp1: First fingerprint
        fp2: Second fingerprint
        
    Returns:
        Dice similarity (0-1)
    """
    intersection = np.sum(fp1 & fp2)
    sum_both = np.sum(fp1) + np.sum(fp2)
    
    if sum_both == 0:
        return 0.0
    
    return 2 * intersection / sum_both


def batch_smiles_to_fingerprints(
    smiles_list: List[str],
    fp_type: str = 'morgan',
    **kwargs
) -> np.ndarray:
    """
    Convert batch of SMILES to fingerprints.
    
    Args:
        smiles_list: List of SMILES strings
        fp_type: Type of fingerprint ('morgan', 'rdkit', 'maccs')
        **kwargs: Additional arguments for fingerprint generation
        
    Returns:
        Array of fingerprints
    """
    fingerprints = []
    
    for smiles in smiles_list:
        if fp_type == 'morgan':
            fp = smiles_to_morgan_fingerprint(smiles, **kwargs)
        elif fp_type == 'rdkit':
            fp = smiles_to_rdkit_fingerprint(smiles, **kwargs)
        elif fp_type == 'maccs':
            fp = smiles_to_maccs_fingerprint(smiles)
        else:
            raise ValueError(f"Unknown fingerprint type: {fp_type}")
        
        if fp is not None:
            fingerprints.append(fp)
        else:
            # Add zero vector for invalid SMILES
            if fp_type == 'maccs':
                fingerprints.append(np.zeros(167, dtype=np.int8))
            else:
                n_bits = kwargs.get('n_bits', 2048)
                fingerprints.append(np.zeros(n_bits, dtype=np.int8))
    
    return np.array(fingerprints)


def get_fingerprint(
    smiles: str,
    fp_type: str = 'morgan',
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[np.ndarray]:
    """
    Unified interface for fingerprint generation.
    
    Args:
        smiles: SMILES string
        fp_type: Type of fingerprint
        radius: Radius for Morgan fingerprint
        n_bits: Number of bits
        
    Returns:
        Fingerprint array or None
    """
    if fp_type == 'morgan':
        return smiles_to_morgan_fingerprint(smiles, radius=radius, n_bits=n_bits)
    elif fp_type == 'rdkit':
        return smiles_to_rdkit_fingerprint(smiles, n_bits=n_bits)
    elif fp_type == 'maccs':
        return smiles_to_maccs_fingerprint(smiles)
    else:
        raise ValueError(f"Unknown fingerprint type: {fp_type}")
