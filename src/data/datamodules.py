"""
PyTorch DataLoader wrappers for molecular data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple

from .graph_featurizer import GraphFeaturizer
from .smiles_tokenizer import SMILESTokenizer


class MolecularGraphDataset(Dataset):
    """Dataset for molecular graphs."""
    
    def __init__(
        self,
        smiles_list: List[str],
        labels: Optional[List] = None,
        featurizer: Optional[GraphFeaturizer] = None,
        max_atoms: int = 9,
    ):
        self.smiles_list = smiles_list
        self.labels = labels
        
        if featurizer is None:
            self.featurizer = GraphFeaturizer(max_atoms=max_atoms)
        else:
            self.featurizer = featurizer
        
        # Precompute graphs
        self.graphs = []
        self.valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            graph = self.featurizer.smiles_to_graph(smiles)
            if graph is not None:
                self.graphs.append(graph)
                self.valid_indices.append(i)
    
    def __len__(self) -> int:
        return len(self.graphs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        graph = self.graphs[idx]
        
        item = {
            'adj': torch.FloatTensor(graph['adj']),
            'nodes': torch.FloatTensor(graph['nodes']),
            'num_atoms': torch.LongTensor([graph['num_atoms']]),
        }
        
        if self.labels is not None:
            original_idx = self.valid_indices[idx]
            item['label'] = torch.FloatTensor([self.labels[original_idx]])
        
        return item


class SMILESDataset(Dataset):
    """Dataset for SMILES sequences."""
    
    def __init__(
        self,
        smiles_list: List[str],
        tokenizer: SMILESTokenizer,
        labels: Optional[List] = None,
    ):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.smiles_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        smiles = self.smiles_list[idx]
        tokens = self.tokenizer.encode(smiles)
        
        item = {
            'tokens': torch.LongTensor(tokens),
            'smiles': smiles,
        }
        
        if self.labels is not None:
            item['label'] = torch.FloatTensor([self.labels[idx]])
        
        return item


def create_dataloader(
    smiles_list: List[str],
    labels: Optional[List] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    mode: str = 'graph',
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for molecular data.
    
    Args:
        smiles_list: List of SMILES strings
        labels: Optional labels
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers
        mode: 'graph' or 'sequence'
        **kwargs: Additional arguments for dataset
        
    Returns:
        DataLoader
    """
    if mode == 'graph':
        dataset = MolecularGraphDataset(smiles_list, labels, **kwargs)
    elif mode == 'sequence':
        dataset = SMILESDataset(smiles_list, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
