#!/usr/bin/env python3
"""
Delta-signature computation for incremental updates.

Computes hashes of changes between consecutive scenes to enable
incremental updates without full graph rebuilds.
"""

import hashlib
from typing import Any, Dict, Set
from .multi_signature import MultiSignature

class DeltaSignatureComputer:
    """Computes delta signatures between consecutive scenes"""
    
    def __init__(self):
        self.salt = b"delta_signature_salt"
    
    def compute_delta_signature(self, prev_signature: MultiSignature, 
                               curr_signature: MultiSignature) -> bytes:
        """
        Compute delta signature between two multi-signatures.
        
        Args:
            prev_signature: Previous scene's multi-signature
            curr_signature: Current scene's multi-signature
            
        Returns:
            Delta signature hash
        """
        hasher = hashlib.blake2b(key=self.salt)
        
        # Compare object signatures
        prev_objects = set(prev_signature.object_signatures.keys())
        curr_objects = set(curr_signature.object_signatures.keys())
        
        added_objects = curr_objects - prev_objects
        removed_objects = prev_objects - curr_objects
        common_objects = prev_objects & curr_objects
        
        # Hash changes
        hasher.update(f"added:{len(added_objects)}".encode())
        hasher.update(f"removed:{len(removed_objects)}".encode())
        
        # Check for modified objects
        modified_count = 0
        for obj_id in common_objects:
            if (prev_signature.object_signatures[obj_id] != 
                curr_signature.object_signatures[obj_id]):
                modified_count += 1
        
        hasher.update(f"modified:{modified_count}".encode())
        
        return hasher.digest()
    
    def is_minimal_change(self, delta_signature: bytes, threshold: float = 0.1) -> bool:
        """Check if delta represents minimal change"""
        # Simple heuristic - in practice would be more sophisticated
        delta_hash = int.from_bytes(delta_signature[:4], 'big')
        normalized = (delta_hash % 1000) / 1000.0
        return normalized < threshold
