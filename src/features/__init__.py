"""
Feature extraction modules.
"""

from .rdkit_descriptors import (
    calculate_qed,
    calculate_logp,
    calculate_sa_score,
    calculate_molecular_weight,
    calculate_tpsa,
    calculate_num_h_donors,
    calculate_num_h_acceptors,
    calculate_num_rotatable_bonds,
    calculate_num_aromatic_rings,
    calculate_all_descriptors,
    calculate_descriptors_batch,
    get_descriptor_vector,
    lipinski_rule_of_five,
)
from .fingerprint_utils import (
    smiles_to_morgan_fingerprint,
    smiles_to_rdkit_fingerprint,
    smiles_to_maccs_fingerprint,
    calculate_tanimoto_similarity,
    calculate_dice_similarity,
    batch_smiles_to_fingerprints,
    get_fingerprint,
)

__all__ = [
    'calculate_qed',
    'calculate_logp',
    'calculate_sa_score',
    'calculate_molecular_weight',
    'calculate_tpsa',
    'calculate_num_h_donors',
    'calculate_num_h_acceptors',
    'calculate_num_rotatable_bonds',
    'calculate_num_aromatic_rings',
    'calculate_all_descriptors',
    'calculate_descriptors_batch',
    'get_descriptor_vector',
    'lipinski_rule_of_five',
    'smiles_to_morgan_fingerprint',
    'smiles_to_rdkit_fingerprint',
    'smiles_to_maccs_fingerprint',
    'calculate_tanimoto_similarity',
    'calculate_dice_similarity',
    'batch_smiles_to_fingerprints',
    'get_fingerprint',
]
