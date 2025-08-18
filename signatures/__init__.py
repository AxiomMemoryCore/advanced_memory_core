"""
Multi-Signature System

Hierarchical signature computation for exact matching and compositional caching.
"""

from .multi_signature import MultiSignatureComputer, SignatureTier
from .delta_signature import DeltaSignatureComputer
from .tempo_signature import TempoSignatureComputer

__all__ = [
    'MultiSignatureComputer',
    'SignatureTier', 
    'DeltaSignatureComputer',
    'TempoSignatureComputer'
]
