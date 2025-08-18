#!/usr/bin/env python3
"""
Core memory system interfaces and data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
from enum import Enum

class QueryType(Enum):
    """Types of memory queries for routing"""
    EXACT_MATCH = "exact_match"
    SUBGRAPH_COMPOSE = "subgraph_compose" 
    TEMPORAL_PATTERN = "temporal_pattern"
    FULL_REASONING = "full_reasoning"

@dataclass
class MemoryQuery:
    """Standardized memory query across all systems"""
    query_type: QueryType
    input_data: Any  # Could be scene graph, grid, text, etc.
    context: Dict[str, Any] = None
    max_results: int = 10
    latency_budget_ms: float = 20.0
    metadata: Dict[str, Any] = None

@dataclass
class MemoryResult:
    """Standardized memory result from any system"""
    data: Any
    confidence: float
    latency_ms: float
    cache_tier: str  # "L1", "L2", "L3", "miss"
    provenance: 'ProvenanceTuple'
    metadata: Dict[str, Any] = None

class MemorySystemInterface(ABC):
    """Base interface for all memory subsystems"""
    
    def __init__(self, system_name: str):
        self.system_name = system_name
        self.enabled = True
        self.stats = {
            'queries': 0,
            'hits': 0,
            'misses': 0,
            'avg_latency_ms': 0.0,
            'avg_confidence': 0.0
        }
    
    @abstractmethod
    def query(self, memory_query: MemoryQuery) -> MemoryResult:
        """Execute memory query - must be implemented by each system"""
        pass
    
    @abstractmethod
    def store(self, data: Any, metadata: Dict[str, Any]) -> bool:
        """Store new memory - must be implemented by each system"""
        pass
    
    def update_stats(self, result: MemoryResult):
        """Update performance statistics"""
        self.stats['queries'] += 1
        if result.cache_tier != "miss":
            self.stats['hits'] += 1
        else:
            self.stats['misses'] += 1
        
        # Update rolling averages
        n = self.stats['queries']
        old_latency = self.stats['avg_latency_ms']
        old_confidence = self.stats['avg_confidence']
        
        self.stats['avg_latency_ms'] = (old_latency * (n-1) + result.latency_ms) / n
        self.stats['avg_confidence'] = (old_confidence * (n-1) + result.confidence) / n
