"""Toxicity preprocessing."""
import numpy as np
from ...features import batch_smiles_to_fingerprints, calculate_descriptors_batch

def prepare_toxicity_features(smiles_list, fp_type='morgan', n_bits=2048):
    fps = batch_smiles_to_fingerprints(smiles_list, fp_type=fp_type, n_bits=n_bits)
    descriptors = calculate_descriptors_batch(smiles_list)
    desc_array = np.array([[d[k] for k in ['qed', 'logp', 'mol_weight', 'tpsa']] for d in descriptors])
    return np.hstack([fps, desc_array])
