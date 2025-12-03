"""Quantum computing modules."""
from .vqc_vvrq import VVRQ
from .vqc_efq import EFQ
from .quantum_layers_pl import create_quantum_layer
from .backends import get_pennylane_device, get_qiskit_backend

__all__ = ['VVRQ', 'EFQ', 'create_quantum_layer', 'get_pennylane_device', 'get_qiskit_backend']
