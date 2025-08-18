"""
Foundation Systems - Safety and reliability infrastructure.

Critical systems that prevent corruption and enable reliable development:
- Invariant Gate: Safety net with executable checks
- Golden Oracle Set: Immutable regression protection  
- Strict Epoching: Version control and compatibility
"""

from .invariant_gate import InvariantGate, Invariant, InvariantSeverity, InvariantViolation
from .golden_oracle import GoldenOracleSet, OracleCase, OracleResult
from .epoching import EpochManager, EpochTuple, CompatibilityMatrix

__all__ = [
    'InvariantGate',
    'Invariant', 
    'InvariantSeverity',
    'InvariantViolation',
    'GoldenOracleSet',
    'OracleCase',
    'OracleResult',
    'EpochManager',
    'EpochTuple',
    'CompatibilityMatrix'
]

