#!/usr/bin/env python3
"""
HDC Operations Tests

Tests the fundamental HDC operations with measurable properties.
No falsification - tests actual mathematical properties.
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdc.hdc_operations import HDCOperations, BoundVector, Bundle
from hdc.role_inventory import RoleInventory

class TestHDCOperations(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.hdc = HDCOperations(dimension=1000, max_bundle_size=10)
        self.inventory = RoleInventory(dimension=1000, seed=42)
        
    def test_xor_binding_mathematical_property(self):
        """Test XOR binding follows mathematical properties"""
        # Get role and create filler
        role = self.inventory.get_role_vector(1)
        filler = np.random.choice([-1, 1], size=1000).astype(np.int8)
        
        # Bind
        bound_vec = self.hdc.bind(role, filler, 1)
        
        # Test XOR property: (A XOR B) XOR A = B
        recovered = self.hdc.unbind(bound_vec.bound_vector, role)
        
        # Should recover original filler exactly
        self.assertTrue(np.array_equal(recovered, filler), 
                       "XOR unbinding should recover original filler exactly")
        
    def test_binding_is_commutative_in_xor_sense(self):
        """Test that XOR binding has expected properties"""
        role1 = self.inventory.get_role_vector(1)
        role2 = self.inventory.get_role_vector(2)
        filler = np.random.choice([-1, 1], size=1000).astype(np.int8)
        
        # Bind with each role
        bound1 = self.hdc.bind(role1, filler, 1)
        bound2 = self.hdc.bind(role2, filler, 2)
        
        # Unbind should work correctly
        recovered1 = self.hdc.unbind(bound1.bound_vector, role1)
        recovered2 = self.hdc.unbind(bound2.bound_vector, role2)
        
        self.assertTrue(np.array_equal(recovered1, filler))
        self.assertTrue(np.array_equal(recovered2, filler))
        
        # Different roles should produce different bound vectors
        self.assertFalse(np.array_equal(bound1.bound_vector, bound2.bound_vector),
                        "Different roles should produce different bound vectors")
    
    def test_bundle_majority_vote_property(self):
        """Test that bundling follows majority vote correctly"""
        # Create multiple bound vectors with known patterns
        role = self.inventory.get_role_vector(1)
        
        # Create 5 fillers, 3 with +1 in first position, 2 with -1
        bound_vectors = []
        for i in range(5):
            filler = np.random.choice([-1, 1], size=1000).astype(np.int8)
            if i < 3:
                filler[0] = 1  # Majority will be +1
            else:
                filler[0] = -1  # Minority will be -1
            
            bound_vec = self.hdc.bind(role, filler, 1)
            bound_vectors.append(bound_vec)
        
        # Bundle
        bundle = self.hdc.bundle(bound_vectors)
        
        # Unbind and check first position
        recovered = self.hdc.unbind(bundle.bundle_vector, role)
        
        # First position should be +1 (majority vote)
        self.assertEqual(recovered[0], 1, 
                        "Majority vote should produce +1 in first position")
    
    def test_corruption_measurement_accuracy(self):
        """Test corruption measurement gives accurate results"""
        original = np.random.choice([-1, 1], size=1000).astype(np.int8)
        
        # Test perfect match
        corruption_perfect = self.hdc.measure_corruption(original, original)
        self.assertEqual(corruption_perfect, 0.0, "Perfect match should have 0% corruption")
        
        # Test known corruption
        corrupted = original.copy()
        corrupted[:100] = -corrupted[:100]  # Flip 100 bits = 10% corruption
        
        corruption_10pct = self.hdc.measure_corruption(original, corrupted)
        self.assertAlmostEqual(corruption_10pct, 0.1, places=3, 
                              msg="10% bit flip should measure as 10% corruption")
        
        # Test complete corruption
        completely_corrupted = -original  # Flip all bits
        corruption_100pct = self.hdc.measure_corruption(original, completely_corrupted)
        self.assertEqual(corruption_100pct, 1.0, "Complete flip should be 100% corruption")
    
    def test_bundle_capacity_limits(self):
        """Test that bundle capacity limits are enforced"""
        role = self.inventory.get_role_vector(1)
        
        # Create more bound vectors than max_bundle_size
        bound_vectors = []
        for i in range(self.hdc.max_bundle_size + 5):  # Exceed limit
            filler = np.random.choice([-1, 1], size=1000).astype(np.int8)
            bound_vec = self.hdc.bind(role, filler, 1)
            bound_vectors.append(bound_vec)
        
        # Should raise ValueError for exceeding capacity
        with self.assertRaises(ValueError, msg="Should reject bundles exceeding capacity"):
            self.hdc.bundle(bound_vectors)
    
    def test_corruption_increases_with_bundle_size(self):
        """Test that corruption increases as bundle size grows"""
        role = self.inventory.get_role_vector(1)
        target_filler = np.random.choice([-1, 1], size=1000).astype(np.int8)
        
        corruption_rates = []
        
        # Test bundle sizes from 1 to 8
        for bundle_size in range(1, 9):
            bound_vectors = []
            
            # First vector is our target
            target_bound = self.hdc.bind(role, target_filler, 1)
            bound_vectors.append(target_bound)
            
            # Add random vectors to increase noise
            for _ in range(bundle_size - 1):
                noise_filler = np.random.choice([-1, 1], size=1000).astype(np.int8)
                noise_bound = self.hdc.bind(role, noise_filler, 1)
                bound_vectors.append(noise_bound)
            
            # Bundle and unbind
            bundle = self.hdc.bundle(bound_vectors)
            recovered = self.hdc.unbind(bundle.bundle_vector, role)
            
            # Measure corruption
            corruption = self.hdc.measure_corruption(target_filler, recovered)
            corruption_rates.append(corruption)
        
        # Corruption should generally increase with bundle size
        # (Allow some variance due to randomness)
        self.assertGreater(corruption_rates[-1], corruption_rates[0],
                          "Corruption should increase with bundle size")
        self.assertEqual(corruption_rates[0], 0.0, 
                        "Single-item bundle should have zero corruption")
    
    def test_performance_tracking_is_real(self):
        """Test that performance tracking measures actual operations"""
        # Reset stats
        self.hdc.reset_stats()
        
        # Perform operations
        role = self.inventory.get_role_vector(1)
        filler = np.random.choice([-1, 1], size=1000).astype(np.int8)
        
        # Do 5 bind operations
        for _ in range(5):
            self.hdc.bind(role, filler, 1)
        
        # Do 3 unbind operations
        bound = self.hdc.bind(role, filler, 1).bound_vector
        for _ in range(3):
            self.hdc.unbind(bound, role)
        
        stats = self.hdc.get_performance_stats()
        
        # Verify counts
        self.assertEqual(stats['operations']['bind_count'], 6)  # 5 + 1 from unbind test
        self.assertEqual(stats['operations']['unbind_count'], 3)
        
        # Verify timing is reasonable (should be microseconds to milliseconds)
        self.assertGreater(stats['latency_ms']['avg_bind'], 0)
        self.assertLess(stats['latency_ms']['avg_bind'], 100)  # Should be under 100ms
    
    def test_role_vector_determinism(self):
        """Test that role vectors are deterministic with same seed"""
        inv1 = RoleInventory(dimension=100, seed=123)
        inv2 = RoleInventory(dimension=100, seed=123)
        
        # Same seed should produce identical vectors
        for role_id in range(1, 11):
            vec1 = inv1.get_role_vector(role_id)
            vec2 = inv2.get_role_vector(role_id)
            self.assertTrue(np.array_equal(vec1, vec2),
                           f"Role {role_id} should be identical with same seed")
        
        # Different seed should produce different vectors
        inv3 = RoleInventory(dimension=100, seed=456)
        vec_different = inv3.get_role_vector(1)
        vec_original = inv1.get_role_vector(1)
        self.assertFalse(np.array_equal(vec_different, vec_original),
                        "Different seeds should produce different vectors")

if __name__ == '__main__':
    unittest.main(verbosity=2)
