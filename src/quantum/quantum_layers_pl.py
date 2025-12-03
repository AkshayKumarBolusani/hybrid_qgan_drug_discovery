"""PennyLane quantum layers."""
from .vqc_vvrq import VVRQ
from .vqc_efq import EFQ

def create_quantum_layer(layer_type="vvrq", n_qubits=8, n_layers=4):
    if layer_type == "vvrq":
        return VVRQ(n_qubits=n_qubits, n_layers=n_layers)
    elif layer_type == "efq":
        return EFQ(n_qubits=n_qubits, n_layers=n_layers)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
