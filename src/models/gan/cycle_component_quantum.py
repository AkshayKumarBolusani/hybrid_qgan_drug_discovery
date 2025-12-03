'''Quantum Cycle Component.'''
import torch
import torch.nn as nn
from ...quantum import create_quantum_layer

class QuantumCycleComponent(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.pre = nn.Linear(hidden_dim, 8)
        self.quantum = create_quantum_layer("efq", n_qubits=8)
        self.post = nn.Linear(8, hidden_dim)
    
    def forward(self, x):
        residual = x
        x = self.pre(x)
        x = self.quantum(x)
        x = self.post(x)
        return x + residual
