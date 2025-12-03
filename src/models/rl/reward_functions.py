"""Reward functions for RL optimization."""
import numpy as np
from ...features import calculate_qed, calculate_logp, calculate_sa_score

class RewardFunction:
    def __init__(self, weights=None):
        self.weights = weights or {
            'qed': 1.0,
            'logp': 0.5,
            'sa': 0.5,
            'qsar_score': 1.0,
            'novelty': 0.3,
            'diversity': 0.3,
            'docking_score': 1.5,
            'toxicity_penalty': 2.0
        }
    
    def calculate_reward(self, smiles, qsar_pred=None, tox_pred=None, docking_score=None, reference_set=None):
        qed = calculate_qed(smiles)
        logp = calculate_logp(smiles)
        sa = 10.0 - calculate_sa_score(smiles)  # Invert so higher is better
        sa = max(0, sa) / 10.0
        
        reward = (
            self.weights['qed'] * qed +
            self.weights['logp'] * (logp / 5.0) +  # Normalize
            self.weights['sa'] * sa
        )
        
        if qsar_pred is not None:
            reward += self.weights['qsar_score'] * qsar_pred
        
        if tox_pred is not None:
            reward -= self.weights['toxicity_penalty'] * tox_pred
        
        if docking_score is not None:
            norm_docking = min(1.0, max(0, (-docking_score - 6) / 4.0))
            reward += self.weights['docking_score'] * norm_docking
        
        return reward
