"""
Advanced Memory System - Core Module

Foundational data structures and interfaces for the advanced memory system.
"""

from .memory_interface import MemorySystemInterface, MemoryQuery, MemoryResult
from .provenance import ProvenanceTuple
from .latency_budget import LatencyBudget, StageBudget

__all__ = [
    'MemorySystemInterface',
    'MemoryQuery', 
    'MemoryResult',
    'ProvenanceTuple',
    'LatencyBudget',
    'StageBudget'
]
