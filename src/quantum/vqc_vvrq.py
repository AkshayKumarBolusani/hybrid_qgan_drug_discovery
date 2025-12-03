"""Vanilla Variational Repetitive Quantum (VVRQ) circuit.

This implementation builds a PennyLane QNode in __init__ with interface="torch"
so forward can pass torch tensors directly. Avoids method-binding issues with
@qml.qnode on instance methods.
"""
import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

class VVRQ(nn.Module):
    """VVRQ quantum layer."""
    
    def __init__(self, n_qubits=8, n_layers=4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_params = n_qubits * n_layers * 3
        self.weights = nn.Parameter(torch.randn(self.n_params) * 0.1)
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Build QNode with closure over n_qubits and n_layers
        n_qubits = self.n_qubits
        n_layers = self.n_layers

        @qml.qnode(self.dev, interface="torch")
        def _qnode(inputs, weights):
            # Encoding
            for i in range(min(inputs.shape[0], n_qubits)):
                qml.RY(inputs[i], wires=i)

            # Variational layers
            idx = 0
            for _ in range(n_layers):
                for qubit in range(n_qubits):
                    qml.RX(weights[idx], wires=qubit)
                    qml.RY(weights[idx + 1], wires=qubit)
                    qml.RZ(weights[idx + 2], wires=qubit)
                    idx += 3
                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._qnode = _qnode
    
    def forward(self, x):
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            # Ensure 1D tensor for inputs per sample
            inp = x[i].reshape(-1)
            out = self._qnode(inp, self.weights)
            # QNode returns a list-like; convert to torch tensor
            outputs.append(torch.as_tensor(out, dtype=torch.float32))
        return torch.stack(outputs, dim=0)
