#!/usr/bin/env python3
"""
Memory Arena Tests

Tests the memory arena's admission, eviction, and capacity management.
Focuses on measurable behaviors and edge cases.
"""

import unittest
import time
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hdc.hdc_memory import HDCMemoryArena, HDCRecord
from hdc.hdc_operations import Bundle

class TestMemoryArena(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.arena = HDCMemoryArena(capacity=10, pinned_capacity=3)
        
    def create_mock_bundle(self):
        """Create a mock bundle for testing"""
        return Bundle(
            bundle_vector=np.random.choice([-1, 1], size=100).astype(np.int8),
            component_count=3,
            capacity_used=0.6,
            corruption_estimate=0.1,
            bound_vectors=[]
        )
    
    def create_test_record(self, subgraph_id: str):
        """Create a test HDC record"""
        return HDCRecord(
            subgraph_id=subgraph_id,
            bundle=self.create_mock_bundle(),
            role_ids={1, 2, 3},
            win_count=0,
            failure_count=0
        )
    
    def test_admission_policy_enforcement(self):
        """Test that admission policy requires multiple observations"""
        record = self.create_test_record("test_subgraph")
        
        # Should not admit without observations
        self.assertFalse(self.arena.should_admit("test_subgraph"))
        success = self.arena.store_record(record)
        self.assertFalse(success, "Should not admit without sufficient observations")
        
        # Observe once - still not enough
        self.arena.observe_subgraph("test_subgraph")
        self.assertFalse(self.arena.should_admit("test_subgraph"))
        
        # Observe twice - now should admit
        self.arena.observe_subgraph("test_subgraph")
        self.assertTrue(self.arena.should_admit("test_subgraph"))
        
        success = self.arena.store_record(record)
        self.assertTrue(success, "Should admit after sufficient observations")
    
    def test_capacity_limits_enforced(self):
        """Test that capacity limits are strictly enforced"""
        # Fill arena to capacity
        for i in range(self.arena.capacity):
            subgraph_id = f"subgraph_{i}"
            
            # Observe to meet admission policy
            self.arena.observe_subgraph(subgraph_id)
            self.arena.observe_subgraph(subgraph_id)
            
            record = self.create_test_record(subgraph_id)
            success = self.arena.store_record(record)
            self.assertTrue(success, f"Should admit item {i} within capacity")
        
        # Verify arena is full
        self.assertEqual(len(self.arena.records), self.arena.capacity)
        
        # Try to add one more - should trigger eviction
        extra_id = "extra_subgraph"
        self.arena.observe_subgraph(extra_id)
        self.arena.observe_subgraph(extra_id)
        
        extra_record = self.create_test_record(extra_id)
        success = self.arena.store_record(extra_record)
        
        # Should succeed but arena should still be at capacity
        self.assertTrue(success, "Should succeed by evicting existing item")
        self.assertEqual(len(self.arena.records), self.arena.capacity)
        
        # Verify eviction occurred
        self.assertGreater(self.arena.stats['performance']['evictions'], 0)
    
    def test_eviction_prefers_low_performers(self):
        """Test that eviction prefers items with low win rates"""
        # Add items with different performance profiles
        high_performer_id = "high_performer"
        low_performer_id = "low_performer"
        
        # Set up high performer
        self.arena.observe_subgraph(high_performer_id)
        self.arena.observe_subgraph(high_performer_id)
        high_record = self.create_test_record(high_performer_id)
        high_record.win_count = 10
        high_record.failure_count = 1
        self.arena.store_record(high_record)
        
        # Set up low performer
        self.arena.observe_subgraph(low_performer_id)
        self.arena.observe_subgraph(low_performer_id)
        low_record = self.create_test_record(low_performer_id)
        low_record.win_count = 1
        low_record.failure_count = 10
        self.arena.store_record(low_record)
        
        # Fill rest of arena
        for i in range(self.arena.capacity - 2):
            subgraph_id = f"filler_{i}"
            self.arena.observe_subgraph(subgraph_id)
            self.arena.observe_subgraph(subgraph_id)
            self.arena.store_record(self.create_test_record(subgraph_id))
        
        # Add one more to force eviction
        trigger_id = "trigger_eviction"
        self.arena.observe_subgraph(trigger_id)
        self.arena.observe_subgraph(trigger_id)
        self.arena.store_record(self.create_test_record(trigger_id))
        
        # High performer should still be there, low performer likely evicted
        self.assertIsNotNone(self.arena.get_record(high_performer_id),
                           "High performer should not be evicted")
    
    def test_pinning_protects_high_performers(self):
        """Test that pinned items are protected from eviction"""
        # Create a high-performing item that should get pinned
        high_performer_id = "pin_candidate"
        self.arena.observe_subgraph(high_performer_id)
        self.arena.observe_subgraph(high_performer_id)
        
        record = self.create_test_record(high_performer_id)
        self.arena.store_record(record)
        
        # Make it a high performer
        for _ in range(10):
            self.arena.record_win(high_performer_id)
        
        # Should get pinned
        self.assertIn(high_performer_id, self.arena.pinned_records,
                     "High performer should be pinned")
        
        # Fill arena to force evictions
        for i in range(self.arena.capacity + 5):  # Overfill to force evictions
            subgraph_id = f"evictable_{i}"
            self.arena.observe_subgraph(subgraph_id)
            self.arena.observe_subgraph(subgraph_id)
            self.arena.store_record(self.create_test_record(subgraph_id))
        
        # Pinned item should still be there
        self.assertIsNotNone(self.arena.get_record(high_performer_id),
                           "Pinned item should not be evicted")
    
    def test_hit_miss_tracking_accuracy(self):
        """Test that hit/miss statistics are accurate"""
        # Reset stats to new schema
        self.arena.stats = {
            'performance': {
                'hits': 0, 'misses': 0, 'evictions': 0, 'admissions': 0, 'wins': 0, 'losses': 0
            },
            'capacity': {
                'total': self.arena.capacity, 'used': 0, 'pinned': 0, 'pressure': 0.0
            },
            'operations': {
                'pinned_promotions': 0, 'pressure_events': 0
            }
        }
        
        # Add one item
        test_id = "hit_test"
        self.arena.observe_subgraph(test_id)
        self.arena.observe_subgraph(test_id)
        self.arena.store_record(self.create_test_record(test_id))
        
        # Test hits
        for _ in range(3):
            result = self.arena.get_record(test_id)
            self.assertIsNotNone(result, "Should hit stored item")
        
        # Test misses
        for i in range(2):
            result = self.arena.get_record(f"missing_{i}")
            self.assertIsNone(result, "Should miss non-existent item")
        
        stats = self.arena.get_stats()
        self.assertEqual(stats['performance']['hits'], 3, "Should record 3 hits")
        
        # Calculate expected hit rate
        expected_hit_rate = 3 / (3 + 2)  # 3 hits, 2 misses
        self.assertAlmostEqual(stats['performance']['hit_rate'], expected_hit_rate,
                              places=3, msg="Hit rate calculation should be accurate")
    
    def test_memory_pressure_calculation(self):
        """Test that memory pressure is calculated correctly"""
        # Empty arena
        self.assertEqual(self.arena.get_memory_pressure(), 0.0,
                        "Empty arena should have 0% pressure")
        
        # Half full
        half_capacity = self.arena.capacity // 2
        for i in range(half_capacity):
            subgraph_id = f"pressure_test_{i}"
            self.arena.observe_subgraph(subgraph_id)
            self.arena.observe_subgraph(subgraph_id)
            self.arena.store_record(self.create_test_record(subgraph_id))
        
        expected_pressure = half_capacity / self.arena.capacity
        self.assertAlmostEqual(self.arena.get_memory_pressure(), expected_pressure,
                              places=3, msg="Pressure should be calculated correctly")
        
        # Full arena
        for i in range(half_capacity, self.arena.capacity):
            subgraph_id = f"pressure_test_{i}"
            self.arena.observe_subgraph(subgraph_id)
            self.arena.observe_subgraph(subgraph_id)
            self.arena.store_record(self.create_test_record(subgraph_id))
        
        self.assertEqual(self.arena.get_memory_pressure(), 1.0,
                        "Full arena should have 100% pressure")
    
    def test_access_time_updates(self):
        """Test that access times are updated correctly"""
        test_id = "access_time_test"
        self.arena.observe_subgraph(test_id)
        self.arena.observe_subgraph(test_id)
        
        record = self.create_test_record(test_id)
        self.arena.store_record(record)
        
        # Get initial access time
        stored_record = self.arena.get_record(test_id)
        initial_access_time = stored_record.last_access
        
        # Wait a bit
        time.sleep(0.01)  # 10ms
        
        # Access again
        accessed_record = self.arena.get_record(test_id)
        updated_access_time = accessed_record.last_access
        
        # Access time should have been updated
        self.assertGreater(updated_access_time, initial_access_time,
                          "Access time should be updated on retrieval")
    
    def test_win_loss_tracking(self):
        """Test that win/loss tracking works correctly"""
        test_id = "win_loss_test"
        self.arena.observe_subgraph(test_id)
        self.arena.observe_subgraph(test_id)
        
        record = self.create_test_record(test_id)
        self.arena.store_record(record)
        
        # Record some wins and losses
        for _ in range(5):
            self.arena.record_win(test_id)
        
        for _ in range(2):
            self.arena.record_failure(test_id)
        
        # Check counts
        updated_record = self.arena.get_record(test_id)
        self.assertEqual(updated_record.win_count, 5, "Should record 5 wins")
        self.assertEqual(updated_record.failure_count, 2, "Should record 2 failures")
        
        # Check win rate calculation
        expected_win_rate = 5 / (5 + 2)
        self.assertAlmostEqual(updated_record.get_win_rate(), expected_win_rate,
                              places=3, msg="Win rate should be calculated correctly")

if __name__ == '__main__':
    unittest.main(verbosity=2)
