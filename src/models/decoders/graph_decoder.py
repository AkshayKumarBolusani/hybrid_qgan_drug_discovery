'''Graph to SMILES decoder.'''
import torch
import torch.nn as nn
from ...data.graph_featurizer import GraphFeaturizer

class GraphDecoder:
    def __init__(self, max_atoms=9):
        self.featurizer = GraphFeaturizer(max_atoms=max_atoms)
    
    def decode(self, adj, nodes):
        adj_np = adj.detach().cpu().numpy()
        nodes_np = nodes.detach().cpu().numpy()
        return self.featurizer.graph_to_smiles(adj_np, nodes_np)
    
    def decode_batch(self, adj_batch, nodes_batch):
        smiles_list = []
        for i in range(len(adj_batch)):
            smiles = self.decode(adj_batch[i], nodes_batch[i])
            smiles_list.append(smiles if smiles else '')
        return smiles_list
