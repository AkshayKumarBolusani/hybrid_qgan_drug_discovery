'''Transformer-based Discriminator.'''
import torch
import torch.nn as nn

class TransformerDiscriminator(nn.Module):
    def __init__(self, input_dim=81, hidden_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        return self.classifier(x.squeeze(1))
