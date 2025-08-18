#!/usr/bin/env python3
"""
Workflow demonstration for advanced memory system.

This script demonstrates the key innovations:
1. Multi-signature hierarchical indexing
2. Event-sourced provenance
3. Latency-budgeted retrieval
4. Compositional caching
"""

import time
import json
from typing import Dict, Any
from core import LatencyBudget, ProvenanceTuple
from signatures import MultiSignatureComputer

def create_sample_scene() -> Dict[str, Any]:
    """Create a sample scene for testing"""
    return {
        'objects': [
            {
                'id': 'obj1',
                'type': 'rectangle',
                'attributes': {'color': 'red', 'size': 'large'}
            },
            {
                'id': 'obj2', 
                'type': 'circle',
                'attributes': {'color': 'blue', 'size': 'small'}
            },
            {
                'id': 'obj3',
                'type': 'triangle',
                'attributes': {'color': 'green', 'size': 'medium'}
            }
        ],
        'relations': [
            {
                'source': 'obj1',
                'target': 'obj2',
                'type': 'adjacent'
            },
            {
                'source': 'obj2',
                'target': 'obj3',
                'type': 'overlaps'
            }
        ],
        'poses': {
            'obj1': {'position': [0, 0, 0], 'rotation': [0, 0, 0], 'scale': [1, 1, 1]},
            'obj2': {'position': [1, 1, 0], 'rotation': [0, 0, 45], 'scale': [0.5, 0.5, 1]},
            'obj3': {'position': [2, 0, 0], 'rotation': [0, 0, 0], 'scale': [0.8, 0.8, 1]}
        },
        'metadata': {
            'scene_type': 'geometric_arrangement',
            'complexity': 'low'
        }
    }

def demonstrate_multi_signature():
    """Demonstrate multi-signature computation"""
    print("=== Multi-Signature Demonstration ===")
    
    # Create signature computer
    sig_computer = MultiSignatureComputer(pose_bins=64, wl_iterations=3)
    
    # Create sample scene
    scene = create_sample_scene()
    
    # Compute multi-signature
    start_time = time.perf_counter()
    multi_sig = sig_computer.compute_multi_signature(scene)
    compute_time = (time.perf_counter() - start_time) * 1000
    
    print(f"Signature computation time: {compute_time:.2f}ms")
    print(f"Object signatures: {len(multi_sig.object_signatures)}")
    print(f"Subgraph signatures: {len(multi_sig.subgraph_signatures)}")
    print(f"Scene signature: {multi_sig.scene_signature.hex()[:16]}...")
    
    # Show all signatures for cache lookup
    all_sigs = multi_sig.get_all_signatures()
    print(f"Total signatures for cache lookup: {len(all_sigs)}")
    
    return multi_sig

def demonstrate_provenance():
    """Demonstrate provenance tracking"""
    print("\n=== Provenance Demonstration ===")
    
    # Create provenance tuple
    provenance = ProvenanceTuple.create(
        source="multi_signature_computer",
        parameters={
            'pose_bins': 64,
            'wl_iterations': 3,
            'scene_objects': 3
        },
        verifier="signature_validator",
        score=0.95,
        cu_cost=2.5
    )
    
    print(f"Operation ID: {provenance.operation_id}")
    print(f"Source: {provenance.source}")
    print(f"Timestamp: {provenance.timestamp}")
    print(f"CU Cost: {provenance.cu_cost}")
    
    # Serialize/deserialize
    provenance_dict = provenance.to_dict()
    restored_provenance = ProvenanceTuple.from_dict(provenance_dict)
    
    print("Provenance serialization: ✓")
    
    return provenance

def demonstrate_latency_budget():
    """Demonstrate latency budgeting"""
    print("\n=== Latency Budget Demonstration ===")
    
    budget = LatencyBudget(total_budget_ms=20.0, abort_threshold_ms=18.0)
    budget.start_request()
    
    # Simulate stages
    stages = [
        ("signature_compute", 1.2),
        ("l1_cache_probe", 0.3),
        ("subgraph_compose", 3.8),
        ("hdc_binding", 2.1)
    ]
    
    for stage_name, duration_ms in stages:
        # Check if we have budget
        from core.latency_budget import StageBudget
        stage_enum = getattr(StageBudget, stage_name.upper(), None)
        
        if stage_enum and budget.check_stage_budget(stage_enum):
            # Simulate work
            time.sleep(duration_ms / 1000)
            budget.record_stage(stage_name)
            print(f"{stage_name}: {duration_ms}ms (✓)")
        else:
            print(f"{stage_name}: SKIPPED (budget exceeded)")
            break
        
        if budget.should_abort():
            print("ABORTING: Budget exceeded")
            break
    
    print(f"Total elapsed: {budget.get_elapsed_ms():.2f}ms")
    print(f"Remaining budget: {budget.get_remaining_ms():.2f}ms")
    print(f"Stage breakdown: {budget.get_stage_breakdown()}")

def demonstrate_compositional_caching():
    """Demonstrate the concept of compositional caching"""
    print("\n=== Compositional Caching Demonstration ===")
    
    # Simulate cache states
    l1_cache = {}  # scene_signature -> result
    l2_cache = {}  # subgraph_signature -> partial_result
    
    # Create two scenes - one is a modification of the other
    scene1 = create_sample_scene()
    scene2 = create_sample_scene()
    
    # Modify scene2 slightly (add one object)
    scene2['objects'].append({
        'id': 'obj4',
        'type': 'square', 
        'attributes': {'color': 'yellow', 'size': 'tiny'}
    })
    
    sig_computer = MultiSignatureComputer()
    
    # Compute signatures for both scenes
    multi_sig1 = sig_computer.compute_multi_signature(scene1)
    multi_sig2 = sig_computer.compute_multi_signature(scene2)
    
    # Simulate caching result for scene1
    l1_cache[multi_sig1.scene_signature] = "COMPLETE_RESULT_1"
    
    # Cache subgraph results
    for subgraph_id, subgraph_sig in multi_sig1.subgraph_signatures.items():
        l2_cache[subgraph_sig] = f"PARTIAL_RESULT_{subgraph_id}"
    
    print("Cached scene1 results")
    print(f"L1 cache size: {len(l1_cache)}")
    print(f"L2 cache size: {len(l2_cache)}")
    
    # Query scene2 - demonstrate retrieval hierarchy
    print("\nQuerying scene2:")
    
    # L1: Exact scene match
    if multi_sig2.scene_signature in l1_cache:
        print("L1 HIT: Exact scene match")
        result = l1_cache[multi_sig2.scene_signature]
    else:
        print("L1 MISS: No exact scene match")
        
        # L2: Subgraph composition
        partial_hits = []
        for subgraph_sig in multi_sig2.subgraph_signatures.values():
            if subgraph_sig in l2_cache:
                partial_hits.append(l2_cache[subgraph_sig])
        
        if partial_hits:
            print(f"L2 PARTIAL HIT: Can compose from {len(partial_hits)} cached subgraphs")
            result = f"COMPOSED_FROM_{len(partial_hits)}_SUBGRAPHS"
        else:
            print("L2 MISS: No subgraph matches - full computation required")
            result = "FULL_COMPUTATION_RESULT"
    
    print(f"Final result: {result}")

def main():
    """Run all demonstrations"""
    print("Advanced Memory System Workflow Demonstration")
    print("=" * 50)
    
    # Run demonstrations
    multi_sig = demonstrate_multi_signature()
    provenance = demonstrate_provenance()
    demonstrate_latency_budget()
    demonstrate_compositional_caching()
    
    print("\n=== Summary ===")
    print("✓ Multi-signature hierarchical indexing")
    print("✓ Event-sourced provenance tracking") 
    print("✓ Latency-budgeted execution")
    print("✓ Compositional caching concept")
    print("\nNext steps:")
    print("- Implement HDC compositional memory")
    print("- Add option-kernel compilation")
    print("- Build event-sourced storage")
    print("- Create production cache layers")

if __name__ == "__main__":
    main()
