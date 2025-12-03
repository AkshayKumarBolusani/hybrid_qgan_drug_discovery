"""Exponential Fourier Quantum (EFQ) circuit."""
import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

class EFQ(nn.Module):
    """EFQ quantum layer with Fourier features."""
    
    def __init__(self, n_qubits=8, n_layers=3, fourier_features=16):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.fourier_features = fourier_features
        self.weights = nn.Parameter(torch.randn(n_qubits * n_layers * 2) * 0.1)
        
    @qml.qnode(qml.device("default.qubit", wires=8))
    def quantum_circuit(self, inputs, weights):
        # Fourier encoding
        for i in range(min(len(inputs), 8)):
            qml.RY(inputs[i] * np.pi, wires=i)
            qml.RZ(inputs[i] * np.pi, wires=i)
        
        # Variational layers
        idx = 0
        for layer in range(3):
            for qubit in range(8):
                qml.RX(weights[idx], wires=qubit)
                qml.RY(weights[idx+1], wires=qubit)
                idx += 2
            for qubit in range(7):
                qml.CNOT(wires=[qubit, qubit+1])
        
        return [qml.expval(qml.PauliZ(i)) for i in range(8)]
    
    def forward(self, x):
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            out = self.quantum_circuit(x[i].detach().numpy(), self.weights.detach().numpy())
            outputs.append(out)
        return torch.tensor(outputs, dtype=torch.float32)
