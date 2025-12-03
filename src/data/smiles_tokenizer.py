"""
SMILES tokenizer for molecular sequence processing.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

from rdkit import Chem

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class SMILESTokenizer:
    """Tokenizer for SMILES strings."""
    
    # SMILES tokens pattern
    SMI_REGEX_PATTERN = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        max_length: int = 100,
    ):
        self.max_length = max_length
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_file = vocab_file
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.unk_token = '<UNK>'
        
        self.special_tokens = [
            self.pad_token,
            self.start_token,
            self.end_token,
            self.unk_token,
        ]
        
        if vocab_file and Path(vocab_file).exists():
            self.load_vocab(vocab_file)
        else:
            # Initialize with special tokens
            self._init_vocab()
    
    def _init_vocab(self):
        """Initialize vocabulary with special tokens."""
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
    
    def tokenize(self, smiles: str) -> List[str]:
        """
        Tokenize a SMILES string.
        
        Args:
            smiles: SMILES string
            
        Returns:
            List of tokens
        """
        regex = re.compile(self.SMI_REGEX_PATTERN)
        tokens = [token for token in regex.findall(smiles)]
        return tokens
    
    def build_vocab(self, smiles_list: List[str], min_freq: int = 1):
        """
        Build vocabulary from SMILES list.
        
        Args:
            smiles_list: List of SMILES strings
            min_freq: Minimum frequency for token inclusion
        """
        logger.info(f"Building vocabulary from {len(smiles_list)} SMILES...")
        
        # Count token frequencies
        token_freq = {}
        for smiles in smiles_list:
            tokens = self.tokenize(smiles)
            for token in tokens:
                token_freq[token] = token_freq.get(token, 0) + 1
        
        # Initialize with special tokens
        self._init_vocab()
        
        # Add tokens above frequency threshold
        for token, freq in sorted(token_freq.items(), key=lambda x: -x[1]):
            if freq >= min_freq and token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.reverse_vocab[idx] = token
        
        logger.info(f"Vocabulary size: {len(self.vocab)}")
    
    def encode(
        self,
        smiles: str,
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True,
    ) -> List[int]:
        """
        Encode SMILES to token indices.
        
        Args:
            smiles: SMILES string
            add_special_tokens: Add START and END tokens
            padding: Pad to max_length
            truncation: Truncate to max_length
            
        Returns:
            List of token indices
        """
        tokens = self.tokenize(smiles)
        
        if add_special_tokens:
            tokens = [self.start_token] + tokens + [self.end_token]
        
        if truncation and len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            if add_special_tokens:
                tokens[-1] = self.end_token
        
        # Convert to indices
        indices = [
            self.vocab.get(token, self.vocab[self.unk_token])
            for token in tokens
        ]
        
        if padding and len(indices) < self.max_length:
            indices += [self.vocab[self.pad_token]] * (self.max_length - len(indices))
        
        return indices
    
    def decode(
        self,
        indices: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token indices to SMILES.
        
        Args:
            indices: List of token indices
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            SMILES string
        """
        tokens = []
        for idx in indices:
            token = self.reverse_vocab.get(idx, self.unk_token)
            
            if skip_special_tokens and token in self.special_tokens:
                if token == self.end_token:
                    break
                continue
            
            tokens.append(token)
        
        return ''.join(tokens)
    
    def batch_encode(
        self,
        smiles_list: List[str],
        **kwargs
    ) -> List[List[int]]:
        """Encode a batch of SMILES."""
        return [self.encode(smiles, **kwargs) for smiles in smiles_list]
    
    def batch_decode(
        self,
        indices_list: List[List[int]],
        **kwargs
    ) -> List[str]:
        """Decode a batch of token indices."""
        return [self.decode(indices, **kwargs) for indices in indices_list]
    
    def save_vocab(self, filepath: str):
        """Save vocabulary to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        vocab_data = {
            'vocab': self.vocab,
            'max_length': self.max_length,
            'special_tokens': self.special_tokens,
        }
        
        with open(filepath, 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        logger.info(f"Saved vocabulary to {filepath}")
    
    def load_vocab(self, filepath: str):
        """Load vocabulary from JSON file."""
        with open(filepath, 'r') as f:
            vocab_data = json.load(f)
        
        self.vocab = vocab_data['vocab']
        self.max_length = vocab_data['max_length']
        self.special_tokens = vocab_data['special_tokens']
        
        # Rebuild reverse vocab
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
        logger.info(f"Loaded vocabulary from {filepath} (size: {len(self.vocab)})")
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    @property
    def pad_token_id(self) -> int:
        """Get PAD token ID."""
        return self.vocab[self.pad_token]
    
    @property
    def start_token_id(self) -> int:
        """Get START token ID."""
        return self.vocab[self.start_token]
    
    @property
    def end_token_id(self) -> int:
        """Get END token ID."""
        return self.vocab[self.end_token]
    
    @property
    def unk_token_id(self) -> int:
        """Get UNK token ID."""
        return self.vocab[self.unk_token]


def create_tokenizer_from_smiles(
    smiles_list: List[str],
    vocab_file: str = "data/processed/vocab.json",
    max_length: int = 100,
    min_freq: int = 1,
    save: bool = True,
) -> SMILESTokenizer:
    """
    Create and train a tokenizer from SMILES list.
    
    Args:
        smiles_list: List of SMILES strings
        vocab_file: Path to save vocabulary
        max_length: Maximum sequence length
        min_freq: Minimum token frequency
        save: Whether to save vocabulary
        
    Returns:
        Trained tokenizer
    """
    tokenizer = SMILESTokenizer(max_length=max_length)
    tokenizer.build_vocab(smiles_list, min_freq=min_freq)
    
    if save:
        tokenizer.save_vocab(vocab_file)
    
    return tokenizer


def validate_smiles(smiles: str) -> bool:
    """
    Validate if a SMILES string is valid.
    
    Args:
        smiles: SMILES string
        
    Returns:
        True if valid, False otherwise
    """
    if not smiles or not isinstance(smiles, str):
        return False
    
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Convert SMILES to canonical form.
    
    Args:
        smiles: Input SMILES
        
    Returns:
        Canonical SMILES or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def filter_valid_smiles(smiles_list: List[str]) -> List[str]:
    """
    Filter list to keep only valid SMILES.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Filtered list of valid SMILES
    """
    valid = []
    for smiles in smiles_list:
        if validate_smiles(smiles):
            canonical = canonicalize_smiles(smiles)
            if canonical:
                valid.append(canonical)
    
    logger.info(f"Filtered {len(valid)}/{len(smiles_list)} valid SMILES")
    return valid
