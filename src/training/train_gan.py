"""Training script for HQ-MolGAN."""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.gan.generator_hqmolgan import HQMolGANGenerator
from src.models.gan.discriminator_molgan import MolGANDiscriminator
from src.data import load_molecular_dataset, create_dataloader
from src.utils import TrainingLogger, get_config

def train_gan(config_path=None, epochs=100, batch_size=32):
    """Train GAN model."""
    # Setup
    device = torch.device('cpu')
    logger = TrainingLogger('gan_training', 'logs/gan')
    
    # Load data
    logger.log_message("Loading dataset...")
    smiles, properties = load_molecular_dataset('qm9', max_atoms=9, max_samples=1000)
    dataloader = create_dataloader(smiles, batch_size=batch_size, shuffle=True, mode='graph')
    
    # Models
    generator = HQMolGANGenerator(latent_dim=32).to(device)
    discriminator = MolGANDiscriminator().to(device)
    
    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            adj = batch['adj'].to(device)
            batch_size_real = adj.size(0)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            real_labels = torch.ones(batch_size_real, 1).to(device)
            fake_labels = torch.zeros(batch_size_real, 1).to(device)
            
            real_output = discriminator(adj)
            d_loss_real = criterion(real_output, real_labels)
            
            z = torch.randn(batch_size_real, 32).to(device)
            fake_adj = generator(z)
            fake_output = discriminator(fake_adj.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            
            z = torch.randn(batch_size_real, 32).to(device)
            fake_adj = generator(z)
            fake_output = discriminator(fake_adj)
            g_loss = criterion(fake_output, real_labels)
            
            g_loss.backward()
            g_optimizer.step()
            
            # Logging
            if batch_idx % 10 == 0:
                logger.log_step({
                    'd_loss': d_loss.item(),
                    'g_loss': g_loss.item()
                })
        
        logger.log_epoch(epoch, {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item()
        })
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = Path('experiments/checkpoints')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
            }, checkpoint_dir / f'gan_epoch_{epoch+1}.pt')
    
    logger.log_message("Training complete!")
    return generator, discriminator

if __name__ == '__main__':
    train_gan(epochs=50, batch_size=32)
