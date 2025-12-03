'''Hybrid Quantum MolGAN Generator.'''
import torch
import torch.nn as nn
from ...quantum import create_quantum_layer

class HQMolGANGenerator(nn.Module):
    def __init__(self, latent_dim=32, hidden_dims=[128, 256, 512], output_dim=81, use_quantum=True):
        super().__init__()
        self.latent_dim = latent_dim
        
        layers = []
        in_dim = latent_dim
        for i, h_dim in enumerate(hidden_dims):
            if use_quantum and i == len(hidden_dims) // 2:
                # Add quantum layer in the middle
                layers.append(nn.Linear(in_dim, 8))
                layers.append(create_quantum_layer("vvrq", n_qubits=8, n_layers=4))
                in_dim = 8
            else:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
                in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Tanh())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.model(z).view(z.size(0), 9, 9)
