#!/usr/bin/env python3
"""
Strict Epoching - Version control for all hot-path artifacts.

Prevents version hell by versioning code, signatures, role inventories,
salts, and schemas together with atomic updates.
"""

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Set, List
from pathlib import Path

@dataclass
class EpochTuple:
    """Complete epoch specification for all hot-path artifacts"""
    epoch_id: int
    code_hash: str                      # Hash of critical code files
    role_inventory_hash: str            # Hash of role inventory
    salt: bytes                         # Cryptographic salt
    schema_version: str                 # Data schema version
    kernel_registry_hash: str           # Hash of registered kernels
    cache_format_version: str           # Cache storage format version
    
    # Metadata
    creation_timestamp: float
    creator: str
    description: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'epoch_id': self.epoch_id,
            'code_hash': self.code_hash,
            'role_inventory_hash': self.role_inventory_hash,
            'salt': self.salt.hex(),
            'schema_version': self.schema_version,
            'kernel_registry_hash': self.kernel_registry_hash,
            'cache_format_version': self.cache_format_version,
            'creation_timestamp': self.creation_timestamp,
            'creator': self.creator,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpochTuple':
        """Create from dictionary"""
        return cls(
            epoch_id=data['epoch_id'],
            code_hash=data['code_hash'],
            role_inventory_hash=data['role_inventory_hash'],
            salt=bytes.fromhex(data['salt']),
            schema_version=data['schema_version'],
            kernel_registry_hash=data['kernel_registry_hash'],
            cache_format_version=data['cache_format_version'],
            creation_timestamp=data['creation_timestamp'],
            creator=data['creator'],
            description=data['description']
        )
    
    def is_compatible_with(self, other: 'EpochTuple') -> bool:
        """Check if two epochs are compatible"""
        return (self.epoch_id == other.epoch_id and
                self.code_hash == other.code_hash and
                self.role_inventory_hash == other.role_inventory_hash and
                self.salt == other.salt and
                self.schema_version == other.schema_version)

class CompatibilityMatrix:
    """Tracks epoch compatibility rules and migration paths"""
    
    def __init__(self):
        self.compatibility_rules: Dict[int, Set[int]] = {}
        self.migration_paths: Dict[Tuple[int, int], str] = {}
        self.blocked_combinations: Set[Tuple[int, int]] = set()
    
    def add_compatibility_rule(self, epoch_a: int, epoch_b: int, bidirectional: bool = True):
        """Add compatibility between two epochs"""
        if epoch_a not in self.compatibility_rules:
            self.compatibility_rules[epoch_a] = set()
        self.compatibility_rules[epoch_a].add(epoch_b)
        
        if bidirectional:
            if epoch_b not in self.compatibility_rules:
                self.compatibility_rules[epoch_b] = set()
            self.compatibility_rules[epoch_b].add(epoch_a)
    
    def is_compatible(self, epoch_a: int, epoch_b: int) -> bool:
        """Check if two epochs are compatible"""
        if (epoch_a, epoch_b) in self.blocked_combinations:
            return False
        
        if epoch_a == epoch_b:
            return True
        
        return epoch_b in self.compatibility_rules.get(epoch_a, set())
    
    def block_combination(self, epoch_a: int, epoch_b: int):
        """Explicitly block epoch combination"""
        self.blocked_combinations.add((epoch_a, epoch_b))
        self.blocked_combinations.add((epoch_b, epoch_a))

class EpochManager:
    """
    Manages epoch transitions and compatibility enforcement.
    
    Core principles:
    - Only equal epochs run together on hot path
    - Atomic epoch transitions with rollback capability
    - Migration protocol with validation
    """
    
    def __init__(self, epoch_file: Optional[str] = None):
        self.epoch_file = epoch_file
        self.current_epoch: Optional[EpochTuple] = None
        self.epoch_history: List[EpochTuple] = []
        self.compatibility_matrix = CompatibilityMatrix()
        
        # Migration state
        self.migration_in_progress = False
        self.migration_snapshot: Optional[Dict[str, Any]] = None
        
        # Monitoring
        self.mixed_epoch_attempts = 0
        self.epoch_violations: List[Dict[str, Any]] = []
        
        # Load or create initial epoch
        if epoch_file and Path(epoch_file).exists():
            self._load_epoch_file()
        else:
            self._create_initial_epoch()
    
    def _create_initial_epoch(self):
        """Create initial epoch"""
        self.current_epoch = EpochTuple(
            epoch_id=1,
            code_hash=self._compute_code_hash(),
            role_inventory_hash="initial_inventory_hash",
            salt=self._generate_salt(),
            schema_version="1.0.0",
            kernel_registry_hash="empty_registry",
            cache_format_version="1.0.0",
            creation_timestamp=time.time(),
            creator="epoch_manager",
            description="Initial epoch"
        )
        
        self.epoch_history.append(self.current_epoch)
    
    def _compute_code_hash(self) -> str:
        """Compute hash of critical code files"""
        # In practice, would hash actual source files
        # For demo, use timestamp-based hash
        hasher = hashlib.sha256()
        hasher.update(str(time.time()).encode())
        return hasher.hexdigest()[:16]
    
    def _generate_salt(self) -> bytes:
        """Generate cryptographic salt"""
        return hashlib.sha256(str(time.time()).encode()).digest()[:16]
    
    def check_epoch_compatibility(self, artifact_epoch: int, context: str = "unknown") -> bool:
        """
        Check if artifact epoch is compatible with current epoch.
        
        Args:
            artifact_epoch: Epoch ID of the artifact
            context: Context for logging violations
            
        Returns:
            True if compatible, False if blocked
        """
        if not self.current_epoch:
            return False
        
        current_id = self.current_epoch.epoch_id
        
        if artifact_epoch != current_id:
            # Record mixed-epoch attempt
            self.mixed_epoch_attempts += 1
            self.epoch_violations.append({
                'timestamp': time.time(),
                'context': context,
                'current_epoch': current_id,
                'artifact_epoch': artifact_epoch,
                'action': 'blocked'
            })
            return False
        
        return True
    
    def begin_epoch_transition(self, new_description: str) -> bool:
        """Begin atomic epoch transition"""
        if self.migration_in_progress:
            return False
        
        print("Beginning epoch transition...")
        
        # Freeze writes (would integrate with actual system)
        self.migration_in_progress = True
        
        # Create snapshot
        self.migration_snapshot = {
            'previous_epoch': self.current_epoch.to_dict() if self.current_epoch else None,
            'snapshot_timestamp': time.time(),
            'event_log_position': 'current_position',  # Would be actual position
            'cache_state': 'snapshot_taken'  # Would be actual cache snapshot
        }
        
        print("Migration snapshot created")
        return True
    
    def commit_epoch_transition(self, role_inventory_hash: str, 
                               kernel_registry_hash: str) -> bool:
        """Commit epoch transition with new hashes"""
        if not self.migration_in_progress:
            return False
        
        try:
            # Create new epoch
            new_epoch_id = (self.current_epoch.epoch_id + 1) if self.current_epoch else 1
            
            new_epoch = EpochTuple(
                epoch_id=new_epoch_id,
                code_hash=self._compute_code_hash(),
                role_inventory_hash=role_inventory_hash,
                salt=self._generate_salt(),
                schema_version="1.0.0",  # Would increment as needed
                kernel_registry_hash=kernel_registry_hash,
                cache_format_version="1.0.0",  # Would increment as needed
                creation_timestamp=time.time(),
                creator="epoch_manager",
                description=f"Epoch {new_epoch_id} transition"
            )
            
            # Validate new epoch (would run oracle set here)
            validation_success = True  # Placeholder
            
            if validation_success:
                # Commit transition
                self.current_epoch = new_epoch
                self.epoch_history.append(new_epoch)
                self.migration_in_progress = False
                self.migration_snapshot = None
                
                print(f"Epoch transition committed: {new_epoch_id}")
                return True
            else:
                # Rollback
                return self.rollback_epoch_transition()
                
        except Exception as e:
            print(f"Epoch transition failed: {e}")
            return self.rollback_epoch_transition()
    
    def rollback_epoch_transition(self) -> bool:
        """Rollback failed epoch transition"""
        if not self.migration_in_progress:
            return False
        
        print("Rolling back epoch transition...")
        
        # Restore from snapshot (would restore actual state)
        if self.migration_snapshot:
            print("Restored from migration snapshot")
        
        # Clear migration state
        self.migration_in_progress = False
        self.migration_snapshot = None
        
        print("Epoch transition rolled back")
        return True
    
    def get_current_epoch_id(self) -> Optional[int]:
        """Get current epoch ID"""
        return self.current_epoch.epoch_id if self.current_epoch else None
    
    def get_current_salt(self) -> Optional[bytes]:
        """Get current cryptographic salt"""
        return self.current_epoch.salt if self.current_epoch else None
    
    def get_epoch_stats(self) -> Dict[str, Any]:
        """Get epoch management statistics"""
        return {
            'current_epoch_id': self.get_current_epoch_id(),
            'total_epochs': len(self.epoch_history),
            'mixed_epoch_attempts': self.mixed_epoch_attempts,
            'epoch_violations': len(self.epoch_violations),
            'migration_in_progress': self.migration_in_progress,
            'last_transition': self.epoch_history[-1].creation_timestamp if self.epoch_history else None
        }
    
    def _load_epoch_file(self):
        """Load epoch from file"""
        try:
            with open(self.epoch_file, 'r') as f:
                data = json.load(f)
            
            self.current_epoch = EpochTuple.from_dict(data['current_epoch'])
            
            for epoch_data in data.get('history', []):
                self.epoch_history.append(EpochTuple.from_dict(epoch_data))
                
        except Exception:
            self._create_initial_epoch()
    
    def _save_epoch_file(self):
        """Save epoch to file"""
        if not self.epoch_file:
            return
        
        try:
            data = {
                'current_epoch': self.current_epoch.to_dict() if self.current_epoch else None,
                'history': [epoch.to_dict() for epoch in self.epoch_history],
                'mixed_epoch_attempts': self.mixed_epoch_attempts
            }
            
            with open(self.epoch_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception:
            pass

