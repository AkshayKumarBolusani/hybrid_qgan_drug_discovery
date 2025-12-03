'''Classical MolGAN Discriminator.'''
import torch
import torch.nn as nn

class MolGANDiscriminator(nn.Module):
    def __init__(self, input_dim=81, hidden_dims=[512, 256, 128]):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.3))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)
