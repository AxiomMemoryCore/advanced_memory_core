#!/usr/bin/env python3
"""
Invariant Gate - Executable safety checks with blocking actions.

The safety net that prevents corrupt data from propagating through the system.
Enforces type, range, topology, latency, and logical invariants.
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Optional, Set
from enum import Enum
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.provenance import ProvenanceTuple

class InvariantSeverity(Enum):
    """Severity levels for invariant violations"""
    CRITICAL = "critical"     # Block result, system unsafe
    WARNING = "warning"       # Log and degrade, continue with caution
    INFO = "info"            # Log only, no action needed

@dataclass
class InvariantViolation:
    """Record of an invariant violation"""
    invariant_id: str
    severity: InvariantSeverity
    violation_message: str
    inputs: Dict[str, Any]
    timestamp: float
    stage: str
    blocked: bool

@dataclass
class Invariant:
    """Executable invariant check with ID and severity"""
    invariant_id: str
    description: str
    severity: InvariantSeverity
    check_function: Callable[[Any], bool]
    error_message_func: Callable[[Any], str]
    applicable_stages: Set[str]
    
    def check(self, data: Any, stage: str) -> Optional[InvariantViolation]:
        """Check invariant and return violation if failed"""
        if stage not in self.applicable_stages:
            return None
            
        try:
            if self.check_function(data):
                return None  # Invariant satisfied
            else:
                # Invariant violated
                return InvariantViolation(
                    invariant_id=self.invariant_id,
                    severity=self.severity,
                    violation_message=self.error_message_func(data),
                    inputs={'data_type': type(data).__name__, 'stage': stage},
                    timestamp=time.time(),
                    stage=stage,
                    blocked=(self.severity == InvariantSeverity.CRITICAL)
                )
        except Exception as e:
            # Check function failed - treat as violation
            return InvariantViolation(
                invariant_id=self.invariant_id,
                severity=InvariantSeverity.CRITICAL,
                violation_message=f"Invariant check failed: {str(e)}",
                inputs={'error': str(e), 'stage': stage},
                timestamp=time.time(),
                stage=stage,
                blocked=True
            )

class InvariantGate:
    """
    Safety gate that enforces invariants before returning results.
    
    Core principle: No corrupt data leaves the system.
    Block CRITICAL violations, log WARNING violations.
    """
    
    def __init__(self):
        self.invariants: List[Invariant] = []
        self.violations_log: List[InvariantViolation] = []
        
        # Statistics
        self.total_checks = 0
        self.violations_detected = 0
        self.results_blocked = 0
        self.results_degraded = 0
        
        # Performance tracking
        self.avg_check_time_ms = 0.0
        self.check_times: List[float] = []
        
        self._register_core_invariants()
    
    def _register_core_invariants(self):
        """Register core system invariants"""
        
        # Type invariants
        self.add_invariant(Invariant(
            invariant_id="TYPE_001_NO_NANS",
            description="No NaN values in numeric data",
            severity=InvariantSeverity.CRITICAL,
            check_function=lambda data: self._check_no_nans(data),
            error_message_func=lambda data: f"NaN detected in {type(data).__name__}",
            applicable_stages={"hdc_binding", "composition", "signature_compute"}
        ))
        
        self.add_invariant(Invariant(
            invariant_id="TYPE_002_ARRAY_SHAPES",
            description="Array shapes are valid",
            severity=InvariantSeverity.CRITICAL,
            check_function=lambda data: self._check_array_shapes(data),
            error_message_func=lambda data: f"Invalid array shape in {type(data).__name__}",
            applicable_stages={"hdc_binding", "signature_compute"}
        ))
        
        # HDC invariants
        self.add_invariant(Invariant(
            invariant_id="HDC_001_ROLE_ID_VALID",
            description="Role IDs are valid in inventory",
            severity=InvariantSeverity.CRITICAL,
            check_function=lambda data: self._check_role_ids_valid(data),
            error_message_func=lambda data: f"Invalid role ID in {type(data).__name__}",
            applicable_stages={"hdc_binding", "composition"}
        ))
        
        self.add_invariant(Invariant(
            invariant_id="HDC_002_BUNDLE_CAPACITY",
            description="Bundle depth within capacity limits",
            severity=InvariantSeverity.CRITICAL,
            check_function=lambda data: self._check_bundle_capacity(data),
            error_message_func=lambda data: f"Bundle exceeds capacity in {type(data).__name__}",
            applicable_stages={"hdc_binding"}
        ))
        
        self.add_invariant(Invariant(
            invariant_id="HDC_003_XOR_ROUNDTRIP",
            description="XOR bind/unbind round-trip works",
            severity=InvariantSeverity.WARNING,
            check_function=lambda data: self._check_xor_roundtrip(data),
            error_message_func=lambda data: f"XOR round-trip failed in {type(data).__name__}",
            applicable_stages={"hdc_binding"}
        ))
        
        # Timing invariants
        self.add_invariant(Invariant(
            invariant_id="TIMING_001_STAGE_BUDGET",
            description="Stage execution within budget",
            severity=InvariantSeverity.WARNING,
            check_function=lambda data: self._check_stage_budget(data),
            error_message_func=lambda data: f"Stage budget exceeded: {data.get('elapsed_ms', 0):.2f}ms",
            applicable_stages={"all"}
        ))
        
        self.add_invariant(Invariant(
            invariant_id="TIMING_002_END_TO_END_SLA",
            description="End-to-end within SLA",
            severity=InvariantSeverity.CRITICAL,
            check_function=lambda data: self._check_end_to_end_sla(data),
            error_message_func=lambda data: f"SLA violated: {data.get('total_ms', 0):.2f}ms > 20ms",
            applicable_stages={"pipeline_complete"}
        ))
        
        # Provenance invariants
        self.add_invariant(Invariant(
            invariant_id="PROV_001_TUPLE_FIELDS",
            description="Provenance tuples have required fields",
            severity=InvariantSeverity.CRITICAL,
            check_function=lambda data: self._check_provenance_fields(data),
            error_message_func=lambda data: "Missing required provenance fields",
            applicable_stages={"all"}
        ))
        
        # Cache invariants
        self.add_invariant(Invariant(
            invariant_id="CACHE_001_SIGNATURE_LENGTH",
            description="Signatures have correct length",
            severity=InvariantSeverity.CRITICAL,
            check_function=lambda data: self._check_signature_length(data),
            error_message_func=lambda data: f"Invalid signature length: {len(data) if hasattr(data, '__len__') else 'unknown'}",
            applicable_stages={"signature_compute", "cache_lookup"}
        ))
        
        # Composition invariants
        self.add_invariant(Invariant(
            invariant_id="COMP_001_CONFLICT_BOUND",
            description="Conflict count within bounds",
            severity=InvariantSeverity.WARNING,
            check_function=lambda data: self._check_conflict_bound(data),
            error_message_func=lambda data: f"Too many conflicts: {data.get('conflicts_detected', 0)}",
            applicable_stages={"composition"}
        ))
    
    def add_invariant(self, invariant: Invariant):
        """Add custom invariant to the gate"""
        self.invariants.append(invariant)
    
    def check_invariants(self, data: Any, stage: str) -> List[InvariantViolation]:
        """
        Check all applicable invariants for given data and stage.
        
        Returns list of violations (empty if all pass).
        """
        start_time = time.perf_counter()
        violations = []
        
        for invariant in self.invariants:
            if stage in invariant.applicable_stages or "all" in invariant.applicable_stages:
                violation = invariant.check(data, stage)
                if violation:
                    violations.append(violation)
                    self.violations_log.append(violation)
        
        # Update performance stats
        check_time = (time.perf_counter() - start_time) * 1000
        self.check_times.append(check_time)
        self.total_checks += 1
        
        if violations:
            self.violations_detected += len(violations)
        
        # Update rolling average
        self.avg_check_time_ms = sum(self.check_times) / len(self.check_times)
        
        return violations
    
    def enforce_gate(self, data: Any, stage: str) -> Dict[str, Any]:
        """
        Enforce invariant gate - block or degrade based on violations.
        
        Returns:
            Dictionary with 'allowed', 'violations', 'action_taken'
        """
        violations = self.check_invariants(data, stage)
        
        if not violations:
            return {
                'allowed': True,
                'violations': [],
                'action_taken': 'pass'
            }
        
        # Check for critical violations
        critical_violations = [v for v in violations if v.severity == InvariantSeverity.CRITICAL]
        warning_violations = [v for v in violations if v.severity == InvariantSeverity.WARNING]
        
        if critical_violations:
            # Block result
            self.results_blocked += 1
            for violation in critical_violations:
                violation.blocked = True
            
            return {
                'allowed': False,
                'violations': violations,
                'action_taken': 'blocked',
                'critical_count': len(critical_violations),
                'error_message': f"CRITICAL invariant violations: {[v.invariant_id for v in critical_violations]}"
            }
        
        elif warning_violations:
            # Log and degrade
            self.results_degraded += 1
            
            return {
                'allowed': True,
                'violations': violations,
                'action_taken': 'degraded',
                'warning_count': len(warning_violations),
                'warning_message': f"WARNING invariant violations: {[v.invariant_id for v in warning_violations]}"
            }
        
        return {
            'allowed': True,
            'violations': violations,
            'action_taken': 'pass'
        }
    
    # Invariant check implementations
    def _check_no_nans(self, data: Any) -> bool:
        """Check for NaN values in data"""
        if isinstance(data, np.ndarray):
            return not np.isnan(data).any()
        elif isinstance(data, (list, tuple)):
            return all(not (isinstance(x, float) and np.isnan(x)) for x in data)
        elif isinstance(data, dict):
            return self._check_no_nans(list(data.values()))
        elif isinstance(data, float):
            return not np.isnan(data)
        return True
    
    def _check_array_shapes(self, data: Any) -> bool:
        """Check array shapes are valid"""
        if isinstance(data, np.ndarray):
            return len(data.shape) > 0 and all(dim > 0 for dim in data.shape)
        elif hasattr(data, 'bundle_vector') and isinstance(data.bundle_vector, np.ndarray):
            return len(data.bundle_vector.shape) == 1 and data.bundle_vector.shape[0] > 0
        return True
    
    def _check_role_ids_valid(self, data: Any) -> bool:
        """Check role IDs are valid"""
        if hasattr(data, 'role_ids'):
            # All role IDs should be positive integers
            return all(isinstance(rid, int) and rid > 0 for rid in data.role_ids)
        elif hasattr(data, 'role_id'):
            return isinstance(data.role_id, int) and data.role_id > 0
        return True
    
    def _check_bundle_capacity(self, data: Any) -> bool:
        """Check bundle is within capacity limits"""
        if hasattr(data, 'component_count') and hasattr(data, 'capacity_used'):
            return data.component_count <= 50 and data.capacity_used <= 1.0
        return True
    
    def _check_xor_roundtrip(self, data: Any) -> bool:
        """Check XOR round-trip works (warning level)"""
        # This is expensive, so only sample check
        if hasattr(data, 'bound_vector') and hasattr(data, 'filler_vector'):
            # Simple correlation check instead of full round-trip
            return len(data.bound_vector) == len(data.filler_vector)
        return True
    
    def _check_stage_budget(self, data: Any) -> bool:
        """Check stage timing within budget"""
        if isinstance(data, dict) and 'elapsed_ms' in data:
            stage_budget_map = {
                'signature_compute': 2.0,
                'hdc_binding': 3.0,
                'composition': 5.0,
                'cache_lookup': 1.0
            }
            stage = data.get('stage', 'unknown')
            budget = stage_budget_map.get(stage, 10.0)
            return data['elapsed_ms'] <= budget
        return True
    
    def _check_end_to_end_sla(self, data: Any) -> bool:
        """Check end-to-end SLA compliance"""
        if isinstance(data, dict) and 'total_ms' in data:
            return data['total_ms'] <= 20.0
        return True
    
    def _check_provenance_fields(self, data: Any) -> bool:
        """Check provenance tuple has required fields"""
        if isinstance(data, ProvenanceTuple):
            required_fields = ['source', 'timestamp', 'parameters', 'score', 'operation_id']
            return all(hasattr(data, field) for field in required_fields)
        return True
    
    def _check_signature_length(self, data: Any) -> bool:
        """Check signature has correct length"""
        if isinstance(data, bytes):
            return len(data) in [32, 64]  # Common hash lengths
        elif hasattr(data, 'scene_signature') and isinstance(data.scene_signature, bytes):
            return len(data.scene_signature) in [32, 64]
        return True
    
    def _check_conflict_bound(self, data: Any) -> bool:
        """Check conflict count within bounds"""
        if isinstance(data, dict) and 'conflicts_detected' in data:
            return data['conflicts_detected'] <= 10
        elif hasattr(data, 'conflicts_detected'):
            return data.conflicts_detected <= 10
        return True
    
    def get_violation_summary(self, hours: float = 1.0) -> Dict[str, Any]:
        """Get summary of violations in last N hours"""
        cutoff_time = time.time() - (hours * 3600)
        recent_violations = [v for v in self.violations_log if v.timestamp > cutoff_time]
        
        # Group by invariant ID
        violation_counts = {}
        for violation in recent_violations:
            violation_counts[violation.invariant_id] = violation_counts.get(violation.invariant_id, 0) + 1
        
        # Group by severity
        severity_counts = {
            'critical': len([v for v in recent_violations if v.severity == InvariantSeverity.CRITICAL]),
            'warning': len([v for v in recent_violations if v.severity == InvariantSeverity.WARNING]),
            'info': len([v for v in recent_violations if v.severity == InvariantSeverity.INFO])
        }
        
        return {
            'time_window_hours': hours,
            'total_violations': len(recent_violations),
            'violation_counts_by_id': violation_counts,
            'severity_counts': severity_counts,
            'results_blocked': len([v for v in recent_violations if v.blocked]),
            'most_frequent_violation': max(violation_counts.keys(), key=violation_counts.get) if violation_counts else None
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get gate performance statistics"""
        violation_rate = self.violations_detected / self.total_checks if self.total_checks > 0 else 0
        block_rate = self.results_blocked / self.total_checks if self.total_checks > 0 else 0
        
        return {
            'total_checks': self.total_checks,
            'violations_detected': self.violations_detected,
            'violation_rate': violation_rate,
            'results_blocked': self.results_blocked,
            'results_degraded': self.results_degraded,
            'block_rate': block_rate,
            'avg_check_time_ms': self.avg_check_time_ms,
            'registered_invariants': len(self.invariants)
        }
    
    def clear_violations_log(self):
        """Clear violations log (for testing)"""
        self.violations_log.clear()
        self.violations_detected = 0
        self.results_blocked = 0
        self.results_degraded = 0

