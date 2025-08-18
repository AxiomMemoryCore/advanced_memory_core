#!/usr/bin/env python3
"""
Tempo-signature computation for temporal patterns.

Maintains rolling hashes over temporal windows for detecting
repeating patterns and short loops.
"""

import hashlib
from collections import deque
from typing import Any, Dict, List, Optional
from .multi_signature import MultiSignature

class TempoSignatureComputer:
    """Computes tempo signatures over temporal windows"""
    
    def __init__(self, window_size: int = 8):
        self.window_size = window_size
        self.signature_window = deque(maxlen=window_size)
        self.salt = b"tempo_signature_salt"
    
    def add_signature(self, signature: MultiSignature):
        """Add a new signature to the temporal window"""
        self.signature_window.append(signature.scene_signature)
    
    def compute_tempo_signature(self) -> Optional[bytes]:
        """Compute tempo signature over current window"""
        if len(self.signature_window) < 2:
            return None
        
        hasher = hashlib.blake2b(key=self.salt)
        
        # Hash sequence of scene signatures
        for sig in self.signature_window:
            hasher.update(sig)
        
        return hasher.digest()
    
    def detect_loop(self, min_loop_length: int = 2) -> Optional[int]:
        """Detect if current window contains a repeating loop"""
        if len(self.signature_window) < min_loop_length * 2:
            return None
        
        window_list = list(self.signature_window)
        
        # Check for repeating patterns
        for loop_len in range(min_loop_length, len(window_list) // 2 + 1):
            if self._has_repeating_pattern(window_list, loop_len):
                return loop_len
        
        return None
    
    def _has_repeating_pattern(self, sequence: List[bytes], pattern_length: int) -> bool:
        """Check if sequence has repeating pattern of given length"""
        if len(sequence) < pattern_length * 2:
            return False
        
        pattern = sequence[-pattern_length:]
        prev_pattern = sequence[-pattern_length*2:-pattern_length]
        
        return pattern == prev_pattern
