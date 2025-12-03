"""
RDKit molecular descriptors calculation.
"""

import numpy as np
from typing import Dict, List, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def calculate_qed(smiles: str) -> float:
    """Calculate Quantitative Estimate of Drug-likeness (QED)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    try:
        return QED.qed(mol)
    except:
        return 0.0


def calculate_logp(smiles: str) -> float:
    """Calculate LogP (partition coefficient)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    try:
        return Crippen.MolLogP(mol)
    except:
        return 0.0


def calculate_sa_score(smiles: str) -> float:
    """
    Calculate Synthetic Accessibility score.
    Simplified version (1-10, lower is easier).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 10.0
    
    try:
        # Simplified SA score based on molecular complexity
        num_atoms = mol.GetNumAtoms()
        num_rings = Chem.GetSSSR(mol)
        num_rotatable = Lipinski.NumRotatableBonds(mol)
        
        # Simple heuristic
        sa = 1.0 + (num_atoms / 50.0) + (num_rings / 5.0) + (num_rotatable / 10.0)
        return min(10.0, max(1.0, sa))
    except:
        return 10.0


def calculate_molecular_weight(smiles: str) -> float:
    """Calculate molecular weight."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    try:
        return Descriptors.MolWt(mol)
    except:
        return 0.0


def calculate_tpsa(smiles: str) -> float:
    """Calculate Topological Polar Surface Area."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    try:
        return Descriptors.TPSA(mol)
    except:
        return 0.0


def calculate_num_h_donors(smiles: str) -> int:
    """Calculate number of H-bond donors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    try:
        return Lipinski.NumHDonors(mol)
    except:
        return 0


def calculate_num_h_acceptors(smiles: str) -> int:
    """Calculate number of H-bond acceptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    try:
        return Lipinski.NumHAcceptors(mol)
    except:
        return 0


def calculate_num_rotatable_bonds(smiles: str) -> int:
    """Calculate number of rotatable bonds."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    try:
        return Lipinski.NumRotatableBonds(mol)
    except:
        return 0


def calculate_num_aromatic_rings(smiles: str) -> int:
    """Calculate number of aromatic rings."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    try:
        return Lipinski.NumAromaticRings(mol)
    except:
        return 0


def calculate_all_descriptors(smiles: str) -> Dict[str, float]:
    """
    Calculate all common molecular descriptors.
    
    Args:
        smiles: SMILES string
        
    Returns:
        Dictionary of descriptors
    """
    return {
        'qed': calculate_qed(smiles),
        'logp': calculate_logp(smiles),
        'sa_score': calculate_sa_score(smiles),
        'mol_weight': calculate_molecular_weight(smiles),
        'tpsa': calculate_tpsa(smiles),
        'num_h_donors': calculate_num_h_donors(smiles),
        'num_h_acceptors': calculate_num_h_acceptors(smiles),
        'num_rotatable_bonds': calculate_num_rotatable_bonds(smiles),
        'num_aromatic_rings': calculate_num_aromatic_rings(smiles),
    }


def calculate_descriptors_batch(smiles_list: List[str]) -> List[Dict[str, float]]:
    """Calculate descriptors for a batch of SMILES."""
    return [calculate_all_descriptors(smi) for smi in smiles_list]


def get_descriptor_vector(smiles: str, descriptor_names: Optional[List[str]] = None) -> np.ndarray:
    """
    Get descriptor vector for a molecule.
    
    Args:
        smiles: SMILES string
        descriptor_names: List of descriptor names to include
        
    Returns:
        Numpy array of descriptor values
    """
    descriptors = calculate_all_descriptors(smiles)
    
    if descriptor_names is None:
        descriptor_names = list(descriptors.keys())
    
    vector = np.array([descriptors.get(name, 0.0) for name in descriptor_names])
    return vector


def lipinski_rule_of_five(smiles: str) -> Dict[str, bool]:
    """
    Check Lipinski's Rule of Five.
    
    Returns:
        Dictionary with pass/fail for each criterion
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            'mw': False,
            'logp': False,
            'h_donors': False,
            'h_acceptors': False,
            'passes': False,
        }
    
    mw = calculate_molecular_weight(smiles)
    logp = calculate_logp(smiles)
    h_donors = calculate_num_h_donors(smiles)
    h_acceptors = calculate_num_h_acceptors(smiles)
    
    results = {
        'mw': mw <= 500,
        'logp': logp <= 5,
        'h_donors': h_donors <= 5,
        'h_acceptors': h_acceptors <= 10,
    }
    
    results['passes'] = all(results.values())
    
    return results
