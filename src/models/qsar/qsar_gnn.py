"""GNN-based QSAR model."""
import torch
import torch.nn as nn

class GNNQSARModel(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=3, output_dim=1):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, adj):
        for conv in self.conv_layers:
            x = torch.relu(conv(x))
            x = torch.bmm(adj, x)  # Graph convolution
        x = x.mean(dim=1)  # Global pooling
        return self.output(x)
