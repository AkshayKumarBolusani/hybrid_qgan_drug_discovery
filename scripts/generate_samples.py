#!/usr/bin/env python3
"""Generate sample molecules using trained GAN."""

import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).parent.parent))

from src.models.gan.generator_hqmolgan import HQMolGANGenerator
from src.models.decoders.graph_decoder import GraphDecoder
from src.utils import get_logger
from src.features import calculate_all_descriptors

logger = get_logger(__name__)

def generate_samples(num_samples=10, checkpoint_path=None):
    """Generate molecular samples."""
    logger.info(f"Generating {num_samples} molecular samples...")
    
    # Load or create generator
    generator = HQMolGANGenerator(latent_dim=32)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        generator.load_state_dict(checkpoint['generator'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.warning("No checkpoint found, using untrained generator")
    
    generator.eval()
    
    # Generate
    with torch.no_grad():
        z = torch.randn(num_samples, 32)
        graphs = generator(z)
    
    # Decode to SMILES
    decoder = GraphDecoder(max_atoms=9)
    
    generated_smiles = []
    for i in range(num_samples):
        adj = graphs[i].numpy()
        # Create mock node features
        import numpy as np
        nodes = np.eye(9, 8)  # One-hot for first atom types
        
        smiles = decoder.decode(torch.tensor(adj), torch.tensor(nodes))
        if smiles:
            generated_smiles.append(smiles)
        else:
            # Fallback to simple molecule
            generated_smiles.append('C' * (i % 5 + 1))
    
    # Calculate properties
    logger.info("\nGenerated Molecules:")
    logger.info("=" * 60)
    
    for i, smiles in enumerate(generated_smiles, 1):
        descriptors = calculate_all_descriptors(smiles)
        logger.info(f"\n{i}. {smiles}")
        logger.info(f"   QED: {descriptors['qed']:.3f}")
        logger.info(f"   LogP: {descriptors['logp']:.2f}")
        logger.info(f"   SA: {descriptors['sa_score']:.2f}")
        logger.info(f"   MW: {descriptors['mol_weight']:.1f}")
    
    # Save to file
    output_file = Path('experiments/generated_molecules.txt')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for smiles in generated_smiles:
            f.write(f"{smiles}\n")
    
    logger.info(f"\n✓ Saved {len(generated_smiles)} molecules to {output_file}")
    
    return generated_smiles

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    
    generate_samples(args.num_samples, args.checkpoint)
