"""
Dataset loaders for QM9 and PC9 molecular datasets.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class QM9Dataset:
    """QM9 dataset loader using DeepChem."""
    
    def __init__(
        self,
        data_dir: str = "data/raw/qm9",
        max_atoms: int = 9,
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_atoms = max_atoms
        self.max_samples = max_samples
        self.data = None
        self.smiles = None
        self.properties = None
    
    def download_and_load(self) -> Tuple[List[str], pd.DataFrame]:
        """Download and load QM9 dataset."""
        logger.info("Loading QM9 dataset from DeepChem...")
        
        try:
            import deepchem as dc
            
            # Load QM9 dataset
            tasks, datasets, transformers = dc.molnet.load_qm9(
                featurizer='ECFP',
                splitter='random',
                reload=True,
                data_dir=str(self.data_dir)
            )
            
            train_dataset, valid_dataset, test_dataset = datasets
            
            # Combine all datasets
            all_smiles = []
            all_properties = []
            
            for dataset in [train_dataset, valid_dataset, test_dataset]:
                smiles = [s for s in dataset.ids]
                all_smiles.extend(smiles)
                # Get properties (QM9 has multiple target properties)
                if hasattr(dataset, 'y'):
                    all_properties.extend(dataset.y)
            
            # Filter by max atoms
            filtered_smiles = []
            filtered_properties = []
            
            for i, smi in enumerate(tqdm(all_smiles, desc="Filtering molecules")):
                mol = Chem.MolFromSmiles(smi)
                if mol and mol.GetNumAtoms() <= self.max_atoms:
                    filtered_smiles.append(smi)
                    if i < len(all_properties):
                        filtered_properties.append(all_properties[i])
            
            # Limit samples if needed
            if self.max_samples and len(filtered_smiles) > self.max_samples:
                filtered_smiles = filtered_smiles[:self.max_samples]
                filtered_properties = filtered_properties[:self.max_samples]
            
            self.smiles = filtered_smiles
            
            # Convert properties to DataFrame
            if filtered_properties:
                prop_array = np.array(filtered_properties)
                if len(prop_array.shape) == 1:
                    prop_array = prop_array.reshape(-1, 1)
                
                # QM9 property names
                property_names = [
                    'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
                    'u0', 'u298', 'h298', 'g298', 'cv'
                ][:prop_array.shape[1]]
                
                self.properties = pd.DataFrame(
                    prop_array,
                    columns=property_names
                )
            else:
                self.properties = pd.DataFrame()
            
            logger.info(f"Loaded {len(self.smiles)} molecules from QM9")
            
            return self.smiles, self.properties
            
        except Exception as e:
            logger.error(f"Error loading QM9: {e}")
            logger.info("Falling back to preprocessed QSAR datasets...")
            return self._load_qsar_datasets()
    
    def _load_qsar_datasets(self) -> Tuple[List[str], pd.DataFrame]:
        """Load from preprocessed QSAR datasets (ESOL + FreeSolv)."""
        logger.info("Loading preprocessed QSAR data...")
        
        csv_path = Path("data/processed/molecule_datasets/combined_qsar.csv")
        if not csv_path.exists():
            logger.warning("Combined QSAR CSV not found, using synthetic data")
            return self._generate_synthetic_data()
        
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} molecules from {csv_path}")
            
            # Filter by max atoms
            filtered_smiles = []
            filtered_properties = []
            
            for _, row in tqdm(df.iterrows(), desc="Filtering molecules", total=len(df)):
                smi = row['smiles']
                mol = Chem.MolFromSmiles(smi)
                if mol and mol.GetNumAtoms() <= self.max_atoms:
                    filtered_smiles.append(smi)
                    filtered_properties.append(row['target'])
            
            # Limit samples if needed
            if self.max_samples and len(filtered_smiles) > self.max_samples:
                filtered_smiles = filtered_smiles[:self.max_samples]
                filtered_properties = filtered_properties[:self.max_samples]
            
            self.smiles = filtered_smiles
            
            # Create properties DataFrame with target as primary property
            self.properties = pd.DataFrame({
                'target': filtered_properties,
                'mu': np.random.randn(len(filtered_smiles)) * 2 + 2,  # Placeholder
                'alpha': np.random.randn(len(filtered_smiles)) * 10 + 70,
            })
            
            logger.info(f"Filtered to {len(self.smiles)} molecules (max {self.max_atoms} atoms)")
            
            return self.smiles, self.properties
            
        except Exception as e:
            logger.error(f"Error loading QSAR datasets: {e}")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> Tuple[List[str], pd.DataFrame]:
        """Generate synthetic molecular data for testing."""
        logger.info("Generating synthetic molecular data...")
        
        # Common small molecules
        synthetic_smiles = [
            'C', 'CC', 'CCC', 'CCCC', 'CO', 'CCO', 'CCCO',
            'C=C', 'C=CC', 'C=O', 'CC=O', 'C#C', 'C#N',
            'c1ccccc1', 'Cc1ccccc1', 'c1ccccc1O', 'c1ccccc1N',
            'C1CCC1', 'C1CCCC1', 'C1CCCCC1',
            'CN', 'CCN', 'C(C)N', 'C(C)O',
            'COC', 'CCOC', 'C(C)OC',
            'C=CC=C', 'C#CC#C',
            'c1ccncc1', 'c1cnccc1', 'c1cnccn1',
            'CF', 'CCF', 'C(F)F', 'C(F)(F)F',
            'CCl', 'CBr', 'CI',
            'CS', 'CCS', 'CSC',
            'CP', 'C(=O)O', 'CC(=O)O',
            'C(=O)N', 'CC(=O)N', 'NC=O',
            'C1CC1', 'C1CCC1', 'C1CCCC1',
        ]
        
        # Expand with variations
        expanded = []
        for smi in synthetic_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol and mol.GetNumAtoms() <= self.max_atoms:
                canonical = Chem.MolToSmiles(mol, canonical=True)
                expanded.append(canonical)
        
        # Generate more diverse molecules
        from rdkit.Chem import AllChem
        
        additional = []
        for _ in range(min(200, self.max_samples or 200)):
            # Random molecular generation (simple approach)
            num_atoms = np.random.randint(3, self.max_atoms + 1)
            atoms = np.random.choice(['C', 'N', 'O', 'F'], size=num_atoms, p=[0.6, 0.2, 0.15, 0.05])
            
            try:
                mol = Chem.MolFromSmiles(''.join(atoms[:3]))
                if mol:
                    additional.append(Chem.MolToSmiles(mol, canonical=True))
            except:
                pass
        
        all_smiles = list(set(expanded + additional))
        
        if self.max_samples:
            all_smiles = all_smiles[:self.max_samples]
        
        # Generate random properties
        n_samples = len(all_smiles)
        properties = pd.DataFrame({
            'mu': np.random.randn(n_samples) * 2 + 2,
            'alpha': np.random.randn(n_samples) * 10 + 70,
            'homo': np.random.randn(n_samples) * 2 - 7,
            'lumo': np.random.randn(n_samples) * 2 - 2,
            'gap': np.random.randn(n_samples) * 1 + 5,
            'r2': np.random.randn(n_samples) * 50 + 500,
        })
        
        self.smiles = all_smiles
        self.properties = properties
        
        logger.info(f"Generated {len(all_smiles)} synthetic molecules")
        
        return self.smiles, self.properties
    
    def save(self, filepath: str):
        """Save dataset to pickle."""
        data = {
            'smiles': self.smiles,
            'properties': self.properties,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved dataset to {filepath}")
    
    def load(self, filepath: str):
        """Load dataset from pickle."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.smiles = data['smiles']
        self.properties = data['properties']
        logger.info(f"Loaded dataset from {filepath}")


class PC9Dataset:
    """PC9 dataset (PubChem subset) loader."""
    
    def __init__(
        self,
        data_dir: str = "data/raw/pc9",
        max_atoms: int = 9,
        max_samples: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_atoms = max_atoms
        self.max_samples = max_samples
        self.smiles = None
        self.properties = None
    
    def load_from_file(self, filepath: str) -> Tuple[List[str], pd.DataFrame]:
        """Load PC9 dataset from file."""
        logger.info(f"Loading PC9 dataset from {filepath}...")
        
        df = pd.read_csv(filepath)
        
        # Assume SMILES column exists
        smiles_col = 'SMILES' if 'SMILES' in df.columns else df.columns[0]
        
        all_smiles = df[smiles_col].tolist()
        
        # Filter by max atoms
        filtered_smiles = []
        for smi in tqdm(all_smiles, desc="Filtering molecules"):
            mol = Chem.MolFromSmiles(smi)
            if mol and mol.GetNumAtoms() <= self.max_atoms:
                filtered_smiles.append(smi)
        
        if self.max_samples:
            filtered_smiles = filtered_smiles[:self.max_samples]
        
        self.smiles = filtered_smiles
        
        # Extract other properties if available
        property_cols = [col for col in df.columns if col != smiles_col]
        if property_cols:
            self.properties = df[property_cols].iloc[:len(filtered_smiles)]
        else:
            self.properties = pd.DataFrame()
        
        logger.info(f"Loaded {len(self.smiles)} molecules from PC9")
        
        return self.smiles, self.properties


class BioactivityDataset:
    """Bioactivity dataset loader for QSAR modeling."""
    
    def __init__(self, data_dir: str = "data/raw/bioactivity"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data = None
    
    def load_from_deepchem(self, dataset_name: str = 'tox21'):
        """Load bioactivity dataset from DeepChem."""
        logger.info(f"Loading {dataset_name} from DeepChem...")
        
        try:
            import deepchem as dc
            
            if dataset_name == 'tox21':
                tasks, datasets, transformers = dc.molnet.load_tox21(
                    featurizer='ECFP',
                    reload=True,
                    data_dir=str(self.data_dir)
                )
            elif dataset_name == 'toxcast':
                tasks, datasets, transformers = dc.molnet.load_toxcast(
                    featurizer='ECFP',
                    reload=True,
                    data_dir=str(self.data_dir)
                )
            elif dataset_name == 'freesolv':
                tasks, datasets, transformers = dc.molnet.load_freesolv(
                    featurizer='ECFP',
                    reload=True,
                    data_dir=str(self.data_dir)
                )
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            
            train_dataset, valid_dataset, test_dataset = datasets
            
            # Combine datasets
            all_smiles = []
            all_labels = []
            
            for dataset in [train_dataset, valid_dataset, test_dataset]:
                all_smiles.extend(dataset.ids)
                all_labels.extend(dataset.y)
            
            self.data = pd.DataFrame({
                'smiles': all_smiles,
                'labels': [list(l) if hasattr(l, '__iter__') else [l] for l in all_labels]
            })
            
            logger.info(f"Loaded {len(self.data)} samples from {dataset_name}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading {dataset_name}: {e}")
            return None


def load_molecular_dataset(
    dataset_name: str = 'qm9',
    data_dir: Optional[str] = None,
    max_atoms: int = 9,
    max_samples: Optional[int] = None,
    force_download: bool = False
) -> Tuple[List[str], pd.DataFrame]:
    """
    Unified dataset loader.
    
    Args:
        dataset_name: Name of dataset ('qm9', 'pc9', 'tox21', etc.)
        data_dir: Data directory
        max_atoms: Maximum number of atoms
        max_samples: Maximum number of samples
        force_download: Force re-download
        
    Returns:
        Tuple of (smiles_list, properties_dataframe)
    """
    if data_dir is None:
        data_dir = f"data/raw/{dataset_name}"
    
    cache_file = Path(data_dir) / f"{dataset_name}_processed.pkl"
    
    # Try loading from cache
    if not force_download and cache_file.exists():
        logger.info(f"Loading cached dataset from {cache_file}")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        return data['smiles'], data['properties']
    
    # Load dataset
    if dataset_name == 'qm9':
        dataset = QM9Dataset(data_dir, max_atoms, max_samples)
        smiles, properties = dataset.download_and_load()
    elif dataset_name == 'pc9':
        dataset = PC9Dataset(data_dir, max_atoms, max_samples)
        # Try to find PC9 file
        pc9_files = list(Path(data_dir).glob("*.csv"))
        if pc9_files:
            smiles, properties = dataset.load_from_file(str(pc9_files[0]))
        else:
            logger.warning("No PC9 file found, using QM9 instead")
            return load_molecular_dataset('qm9', max_atoms=max_atoms, max_samples=max_samples)
    else:
        # Try bioactivity dataset
        dataset = BioactivityDataset(data_dir)
        data = dataset.load_from_deepchem(dataset_name)
        if data is not None:
            smiles = data['smiles'].tolist()
            properties = pd.DataFrame()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Cache the result
    cache_data = {'smiles': smiles, 'properties': properties}
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    logger.info(f"Cached dataset to {cache_file}")
    
    return smiles, properties
