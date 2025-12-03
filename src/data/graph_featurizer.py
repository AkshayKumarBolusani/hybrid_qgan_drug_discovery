"""
Graph featurizer for converting SMILES to graph representations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class GraphFeaturizer:
    """Convert molecules to graph representations (adjacency matrix + node features)."""
    
    def __init__(
        self,
        max_atoms: int = 9,
        atom_types: Optional[List[str]] = None,
        bond_types: Optional[List[str]] = None,
    ):
        self.max_atoms = max_atoms
        
        if atom_types is None:
            self.atom_types = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'P']
        else:
            self.atom_types = atom_types
        
        if bond_types is None:
            self.bond_types = [
                Chem.BondType.SINGLE,
                Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE,
                Chem.BondType.AROMATIC,
            ]
        else:
            self.bond_types = bond_types
        
        self.num_atom_features = len(self.atom_types)
        self.num_bond_features = len(self.bond_types)
    
    def smiles_to_graph(
        self,
        smiles: str,
        add_self_loops: bool = False,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Convert SMILES to graph representation.
        
        Args:
            smiles: SMILES string
            add_self_loops: Whether to add self-loops
            
        Returns:
            Dictionary with 'adj' (adjacency), 'nodes' (node features), 'num_atoms'
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        num_atoms = mol.GetNumAtoms()
        if num_atoms > self.max_atoms or num_atoms == 0:
            return None
        
        # Node features (one-hot encoded atom types)
        node_features = np.zeros((self.max_atoms, self.num_atom_features), dtype=np.float32)
        for i, atom in enumerate(mol.GetAtoms()):
            symbol = atom.GetSymbol()
            if symbol in self.atom_types:
                idx = self.atom_types.index(symbol)
                node_features[i, idx] = 1.0
            # If atom type not in list, leave as zeros (unknown atom)
        
        # Adjacency matrix (with bond types as edge features)
        adj_matrix = np.zeros((self.max_atoms, self.max_atoms, self.num_bond_features), dtype=np.float32)
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            
            if bond_type in self.bond_types:
                bond_idx = self.bond_types.index(bond_type)
                adj_matrix[i, j, bond_idx] = 1.0
                adj_matrix[j, i, bond_idx] = 1.0  # Symmetric
        
        # Add self-loops if requested
        if add_self_loops:
            for i in range(num_atoms):
                adj_matrix[i, i, 0] = 1.0  # Use first bond type for self-loop
        
        # Binary adjacency (any bond type)
        adj_binary = np.sum(adj_matrix, axis=-1)
        adj_binary = (adj_binary > 0).astype(np.float32)
        
        return {
            'adj': adj_binary,
            'adj_full': adj_matrix,
            'nodes': node_features,
            'num_atoms': num_atoms,
        }
    
    def batch_smiles_to_graphs(
        self,
        smiles_list: List[str],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert batch of SMILES to graphs.
        
        Args:
            smiles_list: List of SMILES strings
            
        Returns:
            Tuple of (adjacency_batch, node_features_batch, num_atoms_batch)
        """
        batch_size = len(smiles_list)
        
        adj_batch = np.zeros((batch_size, self.max_atoms, self.max_atoms), dtype=np.float32)
        nodes_batch = np.zeros((batch_size, self.max_atoms, self.num_atom_features), dtype=np.float32)
        num_atoms_batch = np.zeros(batch_size, dtype=np.int32)
        
        for i, smiles in enumerate(smiles_list):
            graph = self.smiles_to_graph(smiles)
            if graph:
                adj_batch[i] = graph['adj']
                nodes_batch[i] = graph['nodes']
                num_atoms_batch[i] = graph['num_atoms']
        
        return adj_batch, nodes_batch, num_atoms_batch
    
    def graph_to_smiles(
        self,
        adj: np.ndarray,
        nodes: np.ndarray,
        sanitize: bool = True,
    ) -> Optional[str]:
        """
        Convert graph back to SMILES (approximate).
        
        Args:
            adj: Adjacency matrix
            nodes: Node features
            sanitize: Whether to sanitize molecule
            
        Returns:
            SMILES string or None
        """
        mol = Chem.RWMol()
        
        # Add atoms
        num_atoms = 0
        atom_idx_map = {}
        
        for i in range(len(nodes)):
            if np.sum(nodes[i]) == 0:
                continue
            
            atom_type_idx = np.argmax(nodes[i])
            if atom_type_idx < len(self.atom_types):
                atom_symbol = self.atom_types[atom_type_idx]
                atom = Chem.Atom(atom_symbol)
                mol_idx = mol.AddAtom(atom)
                atom_idx_map[i] = mol_idx
                num_atoms += 1
        
        if num_atoms == 0:
            return None
        
        # Add bonds
        for i in range(len(adj)):
            if i not in atom_idx_map:
                continue
            for j in range(i + 1, len(adj)):
                if j not in atom_idx_map:
                    continue
                
                if adj[i, j] > 0.5:  # Bond exists
                    mol.AddBond(atom_idx_map[i], atom_idx_map[j], Chem.BondType.SINGLE)
        
        # Convert to molecule
        mol = mol.GetMol()
        
        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except:
                return None
        
        try:
            smiles = Chem.MolToSmiles(mol, canonical=True)
            return smiles
        except:
            return None
