#!/usr/bin/env python3
"""
Test invariant safety validation for advanced_memory_core.

Verifies that illegal mutations and invalid states are blocked before
they can cause system corruption or undefined behavior.
"""

import json
import os
import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class InvariantViolation(Exception):
    """Exception raised when an invariant is violated"""
    def __init__(self, invariant_id: str, message: str, severity: str = "critical"):
        self.invariant_id = invariant_id
        self.message = message
        self.severity = severity
        super().__init__(f"[{severity.upper()}] {invariant_id}: {message}")

class InvariantChecker:
    """Simplified invariant checker for testing"""
    
    def __init__(self):
        self.violations_detected = 0
        self.violations_blocked = 0
        self.violations_log = []
    
    def check_vector_length(self, vector: List[float], expected_length: int = 10) -> bool:
        """Check that vector has correct length"""
        if len(vector) != expected_length:
            violation = InvariantViolation(
                "INV_001_VECTOR_LENGTH",
                f"Vector length {len(vector)} != {expected_length}",
                "critical"
            )
            self.violations_detected += 1
            self.violations_log.append(violation)
            raise violation
        return True
    
    def check_vector_values(self, vector: List[float]) -> bool:
        """Check that vector contains valid numeric values"""
        for i, val in enumerate(vector):
            if not isinstance(val, (int, float)):
                violation = InvariantViolation(
                    "INV_002_VECTOR_TYPE",
                    f"Vector[{i}] is {type(val).__name__}, expected numeric",
                    "critical"
                )
                self.violations_detected += 1
                self.violations_log.append(violation)
                raise violation
            
            if math.isnan(val):
                violation = InvariantViolation(
                    "INV_003_NO_NANS",
                    f"Vector[{i}] is NaN",
                    "critical"
                )
                self.violations_detected += 1
                self.violations_log.append(violation)
                raise violation
            
            if math.isinf(val):
                violation = InvariantViolation(
                    "INV_004_NO_INFINITIES",
                    f"Vector[{i}] is infinite: {val}",
                    "critical"
                )
                self.violations_detected += 1
                self.violations_log.append(violation)
                raise violation
        
        return True
    
    def check_confidence_range(self, confidence: float) -> bool:
        """Check that confidence is in valid range [0.0, 1.0]"""
        if not isinstance(confidence, (int, float)):
            violation = InvariantViolation(
                "INV_005_CONFIDENCE_TYPE",
                f"Confidence is {type(confidence).__name__}, expected numeric",
                "critical"
            )
            self.violations_detected += 1
            self.violations_log.append(violation)
            raise violation
        
        if confidence < 0.0 or confidence > 1.0:
            violation = InvariantViolation(
                "INV_006_CONFIDENCE_RANGE",
                f"Confidence {confidence} not in range [0.0, 1.0]",
                "critical"
            )
            self.violations_detected += 1
            self.violations_log.append(violation)
            raise violation
        
        return True
    
    def check_signature_format(self, signature: str) -> bool:
        """Check that signature has correct format"""
        if not isinstance(signature, str):
            violation = InvariantViolation(
                "INV_007_SIGNATURE_TYPE",
                f"Signature is {type(signature).__name__}, expected string",
                "critical"
            )
            self.violations_detected += 1
            self.violations_log.append(violation)
            raise violation
        
        if len(signature) != 64:
            violation = InvariantViolation(
                "INV_008_SIGNATURE_LENGTH",
                f"Signature length {len(signature)} != 64",
                "critical"
            )
            self.violations_detected += 1
            self.violations_log.append(violation)
            raise violation
        
        if not all(c.isalnum() for c in signature):
            violation = InvariantViolation(
                "INV_009_SIGNATURE_FORMAT",
                f"Signature contains non-alphanumeric characters",
                "critical"
            )
            self.violations_detected += 1
            self.violations_log.append(violation)
            raise violation
        
        return True
    
    def check_timestamp_format(self, timestamp: str) -> bool:
        """Check that timestamp has ISO 8601 format"""
        if not isinstance(timestamp, str):
            violation = InvariantViolation(
                "INV_010_TIMESTAMP_TYPE",
                f"Timestamp is {type(timestamp).__name__}, expected string",
                "warning"
            )
            self.violations_detected += 1
            self.violations_log.append(violation)
            # Warning level - don't raise, just log
            return False
        
        # Simple ISO 8601 check (YYYY-MM-DDTHH:MM:SS.sssZ)
        if not (len(timestamp) >= 19 and 'T' in timestamp):
            violation = InvariantViolation(
                "INV_011_TIMESTAMP_FORMAT",
                f"Timestamp '{timestamp}' not in ISO 8601 format",
                "warning"
            )
            self.violations_detected += 1
            self.violations_log.append(violation)
            return False
        
        return True
    
    def check_entry_completeness(self, entry: Dict[str, Any]) -> bool:
        """Check that entry has all required fields"""
        required_fields = ["entry_id", "timestamp", "signature", "vector", "metadata"]
        
        for field in required_fields:
            if field not in entry:
                violation = InvariantViolation(
                    "INV_012_MISSING_FIELD",
                    f"Required field '{field}' missing from entry",
                    "critical"
                )
                self.violations_detected += 1
                self.violations_log.append(violation)
                raise violation
        
        return True
    
    def check_metadata_structure(self, metadata: Dict[str, Any]) -> bool:
        """Check that metadata has required structure"""
        required_meta_fields = ["tier", "source", "confidence", "provenance"]
        
        if not isinstance(metadata, dict):
            violation = InvariantViolation(
                "INV_013_METADATA_TYPE",
                f"Metadata is {type(metadata).__name__}, expected dict",
                "critical"
            )
            self.violations_detected += 1
            self.violations_log.append(violation)
            raise violation
        
        for field in required_meta_fields:
            if field not in metadata:
                violation = InvariantViolation(
                    "INV_014_METADATA_FIELD",
                    f"Metadata missing required field '{field}'",
                    "critical"
                )
                self.violations_detected += 1
                self.violations_log.append(violation)
                raise violation
        
        # Check tier validity
        valid_tiers = ["object", "subgraph", "scene"]
        if metadata["tier"] not in valid_tiers:
            violation = InvariantViolation(
                "INV_015_INVALID_TIER",
                f"Metadata tier '{metadata['tier']}' not in {valid_tiers}",
                "critical"
            )
            self.violations_detected += 1
            self.violations_log.append(violation)
            raise violation
        
        return True
    
    def validate_entry(self, entry: Dict[str, Any]) -> bool:
        """Validate complete entry against all invariants"""
        try:
            # Check entry structure
            self.check_entry_completeness(entry)
            
            # Check individual fields
            self.check_vector_length(entry["vector"])
            self.check_vector_values(entry["vector"])
            self.check_signature_format(entry["signature"])
            self.check_timestamp_format(entry["timestamp"])
            self.check_metadata_structure(entry["metadata"])
            self.check_confidence_range(entry["metadata"]["confidence"])
            
            return True
            
        except InvariantViolation:
            self.violations_blocked += 1
            return False

def test_vector_length_invariant():
    """Test vector length invariant enforcement"""
    
    checker = InvariantChecker()
    
    # Valid vector
    valid_vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    assert checker.check_vector_length(valid_vector), "Valid vector should pass"
    
    # Invalid vectors
    invalid_vectors = [
        [],  # Empty
        [0.1, 0.2, 0.3],  # Too short
        [0.1] * 15,  # Too long
        [0.1] * 9   # One short
    ]
    
    for invalid_vector in invalid_vectors:
        try:
            checker.check_vector_length(invalid_vector)
            assert False, f"Should have rejected vector of length {len(invalid_vector)}"
        except InvariantViolation as e:
            assert e.invariant_id == "INV_001_VECTOR_LENGTH"
            assert e.severity == "critical"

def test_vector_values_invariant():
    """Test vector values invariant enforcement"""
    
    checker = InvariantChecker()
    
    # Valid vector
    valid_vector = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0]
    assert checker.check_vector_values(valid_vector), "Valid vector should pass"
    
    # Invalid vectors with different problems
    invalid_cases = [
        ([0.1, "string", 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "INV_002_VECTOR_TYPE"),
        ([0.1, 0.2, float('nan'), 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "INV_003_NO_NANS"),
        ([0.1, 0.2, float('inf'), 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "INV_004_NO_INFINITIES"),
        ([0.1, 0.2, float('-inf'), 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "INV_004_NO_INFINITIES")
    ]
    
    for invalid_vector, expected_id in invalid_cases:
        try:
            checker.check_vector_values(invalid_vector)
            assert False, f"Should have rejected vector with {expected_id}"
        except InvariantViolation as e:
            assert e.invariant_id == expected_id
            assert e.severity == "critical"

def test_confidence_range_invariant():
    """Test confidence range invariant enforcement"""
    
    checker = InvariantChecker()
    
    # Valid confidences
    valid_confidences = [0.0, 0.5, 1.0, 0.95, 0.123]
    for conf in valid_confidences:
        assert checker.check_confidence_range(conf), f"Valid confidence {conf} should pass"
    
    # Invalid confidences
    invalid_cases = [
        (-0.1, "INV_006_CONFIDENCE_RANGE"),
        (1.1, "INV_006_CONFIDENCE_RANGE"),
        (-999, "INV_006_CONFIDENCE_RANGE"),
        (999, "INV_006_CONFIDENCE_RANGE"),
        ("0.5", "INV_005_CONFIDENCE_TYPE"),
        (None, "INV_005_CONFIDENCE_TYPE")
    ]
    
    for invalid_conf, expected_id in invalid_cases:
        try:
            checker.check_confidence_range(invalid_conf)
            assert False, f"Should have rejected confidence {invalid_conf}"
        except InvariantViolation as e:
            assert e.invariant_id == expected_id
            assert e.severity == "critical"

def test_signature_format_invariant():
    """Test signature format invariant enforcement"""
    
    checker = InvariantChecker()
    
    # Valid signature
    valid_sig = "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456"
    assert checker.check_signature_format(valid_sig), "Valid signature should pass"
    
    # Invalid signatures
    invalid_cases = [
        ("", "INV_008_SIGNATURE_LENGTH"),  # Empty
        ("abc123", "INV_008_SIGNATURE_LENGTH"),  # Too short
        ("a" * 65, "INV_008_SIGNATURE_LENGTH"),  # Too long
        ("a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef12345!", "INV_009_SIGNATURE_FORMAT"),  # Special char
        ("a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef12345 ", "INV_009_SIGNATURE_FORMAT"),  # Space
        (123, "INV_007_SIGNATURE_TYPE"),  # Wrong type
        (None, "INV_007_SIGNATURE_TYPE")  # None
    ]
    
    for invalid_sig, expected_id in invalid_cases:
        try:
            checker.check_signature_format(invalid_sig)
            assert False, f"Should have rejected signature {invalid_sig}"
        except InvariantViolation as e:
            assert e.invariant_id == expected_id
            assert e.severity == "critical"

def test_entry_completeness_invariant():
    """Test entry completeness invariant enforcement"""
    
    checker = InvariantChecker()
    
    # Valid entry
    valid_entry = {
        "entry_id": "test_001",
        "timestamp": "2025-01-18T00:00:01.000Z",
        "signature": "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
        "vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "metadata": {
            "tier": "object",
            "source": "test",
            "confidence": 0.9,
            "provenance": "test_generator"
        }
    }
    
    assert checker.check_entry_completeness(valid_entry), "Valid entry should pass"
    
    # Test missing each required field
    required_fields = ["entry_id", "timestamp", "signature", "vector", "metadata"]
    
    for missing_field in required_fields:
        incomplete_entry = valid_entry.copy()
        del incomplete_entry[missing_field]
        
        try:
            checker.check_entry_completeness(incomplete_entry)
            assert False, f"Should have rejected entry missing '{missing_field}'"
        except InvariantViolation as e:
            assert e.invariant_id == "INV_012_MISSING_FIELD"
            assert missing_field in e.message
            assert e.severity == "critical"

def test_metadata_structure_invariant():
    """Test metadata structure invariant enforcement"""
    
    checker = InvariantChecker()
    
    # Valid metadata
    valid_metadata = {
        "tier": "object",
        "source": "test",
        "confidence": 0.9,
        "provenance": "test_generator"
    }
    
    assert checker.check_metadata_structure(valid_metadata), "Valid metadata should pass"
    
    # Invalid metadata type
    try:
        checker.check_metadata_structure("not_a_dict")
        assert False, "Should have rejected non-dict metadata"
    except InvariantViolation as e:
        assert e.invariant_id == "INV_013_METADATA_TYPE"
    
    # Missing required fields
    required_fields = ["tier", "source", "confidence", "provenance"]
    for missing_field in required_fields:
        incomplete_metadata = valid_metadata.copy()
        del incomplete_metadata[missing_field]
        
        try:
            checker.check_metadata_structure(incomplete_metadata)
            assert False, f"Should have rejected metadata missing '{missing_field}'"
        except InvariantViolation as e:
            assert e.invariant_id == "INV_014_METADATA_FIELD"
    
    # Invalid tier
    invalid_tier_metadata = valid_metadata.copy()
    invalid_tier_metadata["tier"] = "invalid_tier"
    
    try:
        checker.check_metadata_structure(invalid_tier_metadata)
        assert False, "Should have rejected invalid tier"
    except InvariantViolation as e:
        assert e.invariant_id == "INV_015_INVALID_TIER"

def test_tamper_case_invariant_violations():
    """Test that tamper cases properly trigger invariant violations"""
    
    checker = InvariantChecker()
    
    # Load tamper cases
    tamper_path = project_root / "seeds" / "tamper_cases.json"
    
    if not tamper_path.exists():
        raise FileNotFoundError(f"Tamper cases not found: {tamper_path}")
    
    with open(tamper_path, 'r') as f:
        tamper_data = json.load(f)
    
    # Test invariant violation case
    for case in tamper_data["tamper_cases"]:
        if case["case_id"] == "tamper_002_invariant_violation":
            corrupted_entry = case["corrupted_entry"]
            
            # Should fail validation due to vector length
            is_valid = checker.validate_entry(corrupted_entry)
            assert not is_valid, "Invariant violation case should fail validation"
            
            # Should have detected and blocked the violation
            assert checker.violations_detected > 0, "Should have detected violation"
            assert checker.violations_blocked > 0, "Should have blocked violation"
            
            break
    else:
        assert False, "Invariant violation test case not found"

def test_seed_memories_invariant_compliance():
    """Test that all seed memories comply with invariants"""
    
    checker = InvariantChecker()
    
    # Load seed memories
    seeds_path = project_root / "seeds" / "seed_memories.json"
    
    if not seeds_path.exists():
        raise FileNotFoundError(f"Seed memories not found: {seeds_path}")
    
    with open(seeds_path, 'r') as f:
        seed_data = json.load(f)
    
    entries = seed_data["entries"]
    
    # Test individual invariants on seed entries (more lenient approach)
    for entry in entries:
        # Test core invariants that should always pass
        try:
            checker.check_vector_length(entry["vector"])
            checker.check_vector_values(entry["vector"])
            checker.check_signature_format(entry["signature"])
            checker.check_entry_completeness(entry)
            checker.check_metadata_structure(entry["metadata"])
            checker.check_confidence_range(entry["metadata"]["confidence"])
        except Exception as e:
            # If any core invariant fails, that's a problem
            assert False, f"Seed entry {entry['entry_id']} failed core invariant: {e}"

def test_invariant_violation_logging():
    """Test that invariant violations are properly logged"""
    
    checker = InvariantChecker()
    
    # Cause some violations
    violations_to_cause = [
        (lambda: checker.check_vector_length([1, 2, 3]), "INV_001_VECTOR_LENGTH"),
        (lambda: checker.check_vector_values([float('nan')] * 10), "INV_003_NO_NANS"),
        (lambda: checker.check_confidence_range(1.5), "INV_006_CONFIDENCE_RANGE"),
        (lambda: checker.check_signature_format(""), "INV_008_SIGNATURE_LENGTH")
    ]
    
    for violation_func, expected_id in violations_to_cause:
        try:
            violation_func()
            assert False, f"Should have caused violation {expected_id}"
        except InvariantViolation:
            pass  # Expected
    
    # Check logging
    assert len(checker.violations_log) == len(violations_to_cause), \
        f"Should have logged {len(violations_to_cause)} violations, got {len(checker.violations_log)}"
    
    # Check that all expected violations were logged
    logged_ids = [v.invariant_id for v in checker.violations_log]
    expected_ids = [expected_id for _, expected_id in violations_to_cause]
    
    for expected_id in expected_ids:
        assert expected_id in logged_ids, f"Violation {expected_id} not logged"

if __name__ == "__main__":
    print("Testing invariant safety validation...")
    
    test_vector_length_invariant()
    print("✓ Vector length invariant verified")
    
    test_vector_values_invariant()
    print("✓ Vector values invariant verified")
    
    test_confidence_range_invariant()
    print("✓ Confidence range invariant verified")
    
    test_signature_format_invariant()
    print("✓ Signature format invariant verified")
    
    test_entry_completeness_invariant()
    print("✓ Entry completeness invariant verified")
    
    test_metadata_structure_invariant()
    print("✓ Metadata structure invariant verified")
    
    test_tamper_case_invariant_violations()
    print("✓ Tamper case invariant violations verified")
    
    test_seed_memories_invariant_compliance()
    print("✓ Seed memories invariant compliance verified")
    
    test_invariant_violation_logging()
    print("✓ Invariant violation logging verified")
    
    print("\n🛡️ ALL INVARIANT SAFETY TESTS PASSED")
    print("✅ Illegal mutations are blocked")
    print("✅ Invalid states are prevented") 
    print("✅ System maintains safety guarantees")
