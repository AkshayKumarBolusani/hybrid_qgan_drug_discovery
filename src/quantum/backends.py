"""Quantum computing backends (PennyLane/Qiskit)."""
import pennylane as qml
import numpy as np

def get_pennylane_device(n_qubits=8, device_name="default.qubit", shots=None):
    """Get PennyLane device."""
    return qml.device(device_name, wires=n_qubits, shots=shots)

def get_qiskit_backend(backend_name="aer_simulator"):
    """Get Qiskit backend."""
    try:
        from qiskit_aer import Aer
        return Aer.get_backend(backend_name)
    except:
        return None
