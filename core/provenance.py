#!/usr/bin/env python3
"""
Provenance tracking for complete memory audit trails.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime
import uuid

@dataclass
class ProvenanceTuple:
    """Complete provenance information for memory operations"""
    source: str                    # Which component created this
    timestamp: datetime           # When it was created
    parameters: Dict[str, Any]    # Parameters used
    verifier: Optional[str]       # What verified this result
    score: float                  # Confidence/quality score
    operation_id: str            # Unique operation identifier
    cu_cost: float = 0.0         # Cognitive unit cost
    
    @classmethod
    def create(cls, source: str, parameters: Dict[str, Any], 
               verifier: Optional[str] = None, score: float = 1.0,
               cu_cost: float = 0.0) -> 'ProvenanceTuple':
        """Create a new provenance tuple with current timestamp"""
        return cls(
            source=source,
            timestamp=datetime.now(),
            parameters=parameters.copy(),
            verifier=verifier,
            score=score,
            operation_id=str(uuid.uuid4()),
            cu_cost=cu_cost
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'parameters': self.parameters,
            'verifier': self.verifier,
            'score': self.score,
            'operation_id': self.operation_id,
            'cu_cost': self.cu_cost
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProvenanceTuple':
        """Create from dictionary for deserialization"""
        return cls(
            source=data['source'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            parameters=data['parameters'],
            verifier=data.get('verifier'),
            score=data['score'],
            operation_id=data['operation_id'],
            cu_cost=data.get('cu_cost', 0.0)
        )
