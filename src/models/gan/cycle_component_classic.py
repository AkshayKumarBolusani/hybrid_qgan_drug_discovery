'''Classic Cycle Component.'''
import torch
import torch.nn as nn

class ClassicCycleComponent(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=2):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x) + x  # Residual
