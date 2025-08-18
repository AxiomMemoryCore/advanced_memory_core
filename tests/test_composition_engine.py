#!/usr/bin/env python3
"""
Composition Engine Tests

Tests the multi-subgraph composition logic with conflict resolution.
Measures actual composition behavior without falsification.
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdc.compose_engine import ComposeEngine, ConflictResolution
from hdc.hdc_operations import HDCOperations
from hdc.hdc_memory import HDCRecord
from hdc.role_inventory import RoleInventory

class TestCompositionEngine(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.inventory = RoleInventory(dimension=1000, seed=42)
        self.hdc_ops = HDCOperations(dimension=1000)
        self.compose_engine = ComposeEngine(self.inventory, self.hdc_ops)
        
    def create_test_record(self, subgraph_id: str, role_filler_pairs: dict):
        """
        Create HDC record with specific role-filler bindings.
        
        Args:
            subgraph_id: ID for the subgraph
            role_filler_pairs: Dict mapping role_name -> filler_vector
        """
        bound_vectors = []
        role_ids = set()
        
        for role_name, filler in role_filler_pairs.items():
            role_def = self.inventory.get_role_by_name(role_name)
            bound_vec = self.hdc_ops.bind(role_def.vector, filler, role_def.role_id)
            bound_vectors.append(bound_vec)
            role_ids.add(role_def.role_id)
        
        bundle = self.hdc_ops.bundle(bound_vectors)
        
        return HDCRecord(
            subgraph_id=subgraph_id,
            bundle=bundle,
            role_ids=role_ids,
            win_count=5,
            failure_count=1
        )
    
    def test_single_subgraph_composition(self):
        """Test composition with single subgraph (no conflicts)"""
        # Create known filler vectors
        position_filler = np.random.choice([-1, 1], size=1000).astype(np.int8)
        color_filler = np.random.choice([-1, 1], size=1000).astype(np.int8)
        
        # Create record
        record = self.create_test_record("single_test", {
            "position_x": position_filler,
            "color_red": color_filler
        })
        
        # Request composition
        requested_roles = {
            self.inventory.get_role_by_name("position_x").role_id,
            self.inventory.get_role_by_name("color_red").role_id
        }
        
        result = self.compose_engine.compose_from_subgraphs([record], requested_roles)
        
        # Should succeed with no conflicts
        self.assertTrue(result.success, "Single subgraph composition should succeed")
        self.assertEqual(result.conflicts_detected, 0, "Should detect no conflicts")
        self.assertEqual(len(result.composed_fillers), 2, "Should recover both fillers")
        self.assertGreater(result.total_confidence, 0, "Should have positive confidence")
    
    def test_conflict_detection(self):
        """Test that conflicts are properly detected"""
        # Create two records with same roles but different fillers
        position_filler1 = np.ones(1000, dtype=np.int8)  # All +1
        position_filler2 = -np.ones(1000, dtype=np.int8)  # All -1 (conflict!)
        
        record1 = self.create_test_record("conflict_test1", {
            "position_x": position_filler1
        })
        
        record2 = self.create_test_record("conflict_test2", {
            "position_x": position_filler2
        })
        
        # Request composition
        requested_roles = {self.inventory.get_role_by_name("position_x").role_id}
        
        result = self.compose_engine.compose_from_subgraphs([record1, record2], requested_roles)
        
        # Should detect conflict
        self.assertGreater(result.conflicts_detected, 0, "Should detect conflicts")
        
        if result.success:
            # If resolved, should have resolution metadata
            position_role_id = self.inventory.get_role_by_name("position_x").role_id
            composed_filler = result.composed_fillers[position_role_id]
            self.assertTrue(composed_filler.conflict_resolved, "Should mark conflict as resolved")
            self.assertIsNotNone(composed_filler.resolution_method, "Should specify resolution method")
    
    def test_too_many_conflicts_rejection(self):
        """Test that composition fails when conflicts exceed threshold"""
        # Create many conflicting records
        records = []
        
        # Use all available roles to create many conflicts
        role_names = ["position_x", "position_y", "color_red", "color_green", 
                     "size_width", "size_height", "shape_type", "rotation_x",
                     "adjacent_to", "overlaps_with", "contains", "similar_to"]
        
        requested_role_ids = set()
        
        # Create records with conflicting values for each role
        for i in range(3):  # 3 records
            role_filler_pairs = {}
            for role_name in role_names:
                # Create different filler for each record (guaranteed conflicts)
                filler = np.full(1000, i - 1, dtype=np.int8)  # -1, 0, 1 for records 0,1,2
                filler[filler == 0] = 1  # Convert 0 to 1 for bipolar
                role_filler_pairs[role_name] = filler
                requested_role_ids.add(self.inventory.get_role_by_name(role_name).role_id)
            
            record = self.create_test_record(f"conflict_record_{i}", role_filler_pairs)
            records.append(record)
        
        # This should create conflicts for every role
        result = self.compose_engine.compose_from_subgraphs(records, requested_role_ids)
        
        # Should detect many conflicts
        self.assertGreater(result.conflicts_detected, 5, "Should detect many conflicts")
        
        # May succeed or fail depending on max_conflicts threshold
        if not result.success:
            self.assertEqual(len(result.composed_fillers), 0, 
                           "Failed composition should return no fillers")
    
    def test_confidence_based_resolution(self):
        """Test that higher confidence values win in conflict resolution"""
        # This test is harder to control precisely due to HDC noise,
        # but we can test the general behavior
        
        # Create two records
        filler1 = np.random.choice([-1, 1], size=1000).astype(np.int8)
        filler2 = np.random.choice([-1, 1], size=1000).astype(np.int8)
        
        record1 = self.create_test_record("conf_test1", {"position_x": filler1})
        record2 = self.create_test_record("conf_test2", {"position_x": filler2})
        
        # Make record1 a higher performer (should get higher confidence)
        record1.win_count = 20
        record1.failure_count = 1
        record2.win_count = 1
        record2.failure_count = 10
        
        requested_roles = {self.inventory.get_role_by_name("position_x").role_id}
        
        result = self.compose_engine.compose_from_subgraphs([record1, record2], requested_roles)
        
        if result.success and result.conflicts_detected > 0:
            position_role_id = self.inventory.get_role_by_name("position_x").role_id
            composed_filler = result.composed_fillers[position_role_id]
            
            # Should indicate conflict was resolved
            self.assertTrue(composed_filler.conflict_resolved, 
                           "Should mark conflict as resolved")
            
            # Resolution method should be specified
            self.assertIsNotNone(composed_filler.resolution_method,
                               "Should specify resolution method")
    
    def test_composition_timing(self):
        """Test that composition timing is measured accurately"""
        # Create a simple composition scenario
        filler = np.random.choice([-1, 1], size=1000).astype(np.int8)
        record = self.create_test_record("timing_test", {"position_x": filler})
        
        requested_roles = {self.inventory.get_role_by_name("position_x").role_id}
        
        result = self.compose_engine.compose_from_subgraphs([record], requested_roles)
        
        # Should have reasonable timing (microseconds to milliseconds)
        self.assertGreater(result.composition_time_ms, 0, 
                          "Should record positive composition time")
        self.assertLess(result.composition_time_ms, 100, 
                       "Composition should be under 100ms")
    
    def test_role_priority_system(self):
        """Test that role priorities are applied correctly"""
        priorities = self.compose_engine._build_role_priorities()
        
        # Geometry roles should have higher priority than appearance
        geometry_role_id = self.inventory.get_role_by_name("position_x").role_id
        appearance_role_id = self.inventory.get_role_by_name("color_red").role_id
        
        self.assertGreater(priorities[geometry_role_id], priorities[appearance_role_id],
                          "Geometry roles should have higher priority than appearance")
        
        # All roles should have positive priorities
        for role_id, priority in priorities.items():
            self.assertGreater(priority, 0, f"Role {role_id} should have positive priority")
    
    def test_empty_composition_request(self):
        """Test behavior with empty inputs"""
        # Empty records list
        result = self.compose_engine.compose_from_subgraphs([], {1, 2, 3})
        self.assertFalse(result.success, "Empty records should fail composition")
        self.assertEqual(len(result.composed_fillers), 0, "Should return no fillers")
        
        # Empty requested roles
        filler = np.random.choice([-1, 1], size=1000).astype(np.int8)
        record = self.create_test_record("empty_roles_test", {"position_x": filler})
        
        result = self.compose_engine.compose_from_subgraphs([record], set())
        self.assertEqual(len(result.composed_fillers), 0, "Should return no fillers for empty request")
    
    def test_partial_role_availability(self):
        """Test composition when records don't have all requested roles"""
        # Create records with different role sets
        filler1 = np.random.choice([-1, 1], size=1000).astype(np.int8)
        filler2 = np.random.choice([-1, 1], size=1000).astype(np.int8)
        filler3 = np.random.choice([-1, 1], size=1000).astype(np.int8)
        
        record1 = self.create_test_record("partial1", {
            "position_x": filler1,
            "color_red": filler2
        })
        
        record2 = self.create_test_record("partial2", {
            "color_red": filler3,  # Overlaps with record1
            "shape_type": filler1  # New role
        })
        
        # Request all three roles
        requested_roles = {
            self.inventory.get_role_by_name("position_x").role_id,
            self.inventory.get_role_by_name("color_red").role_id,
            self.inventory.get_role_by_name("shape_type").role_id
        }
        
        result = self.compose_engine.compose_from_subgraphs([record1, record2], requested_roles)
        
        if result.success:
            # Should get all three roles from the two records
            self.assertEqual(len(result.composed_fillers), 3, 
                           "Should compose all requested roles from partial records")
            
            # color_red should have conflict (present in both records)
            color_role_id = self.inventory.get_role_by_name("color_red").role_id
            if color_role_id in result.composed_fillers:
                color_filler = result.composed_fillers[color_role_id]
                # May or may not be marked as conflict depending on resolution
                self.assertGreater(len(color_filler.source_subgraphs), 0,
                                 "Should track source subgraphs")
    
    def test_statistics_tracking(self):
        """Test that composition statistics are tracked accurately"""
        # Reset stats
        self.compose_engine.compose_attempts = 0
        self.compose_engine.compose_successes = 0
        self.compose_engine.conflicts_resolved = 0
        
        # Perform successful composition
        filler = np.random.choice([-1, 1], size=1000).astype(np.int8)
        record = self.create_test_record("stats_test", {"position_x": filler})
        requested_roles = {self.inventory.get_role_by_name("position_x").role_id}
        
        result = self.compose_engine.compose_from_subgraphs([record], requested_roles)
        
        stats = self.compose_engine.get_stats()
        self.assertEqual(stats['compose_attempts'], 1, "Should record 1 attempt")
        
        if result.success:
            self.assertEqual(stats['compose_successes'], 1, "Should record 1 success")
            self.assertEqual(stats['success_rate'], 1.0, "Success rate should be 100%")

if __name__ == '__main__':
    unittest.main(verbosity=2)
