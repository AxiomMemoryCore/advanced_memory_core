#!/usr/bin/env python3
"""
Multi-signature system for hierarchical exact matching.

Computes signatures at object, subgraph, and scene levels to enable
compositional caching and partial retrieval.
"""

import hashlib
import secrets
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Set, Tuple
import numpy as np

class SignatureTier(Enum):
    """Signature computation tiers"""
    OBJECT = "object"
    SUBGRAPH = "subgraph" 
    SCENE = "scene"

@dataclass
class MultiSignature:
    """Multi-tier signature for hierarchical matching"""
    object_signatures: Dict[str, bytes]     # object_id -> signature
    subgraph_signatures: Dict[str, bytes]   # subgraph_id -> signature
    scene_signature: bytes                  # Complete scene signature
    
    def get_all_signatures(self) -> Set[bytes]:
        """Get all signatures for cache lookup"""
        signatures = set()
        signatures.update(self.object_signatures.values())
        signatures.update(self.subgraph_signatures.values())
        signatures.add(self.scene_signature)
        return signatures

class MultiSignatureComputer:
    """Computes multi-tier signatures with WL canonicalization"""
    
    def __init__(self, pose_bins: int = 64, wl_iterations: int = 3):
        self.pose_bins = pose_bins
        self.wl_iterations = wl_iterations
        self.salt = secrets.token_bytes(16)
        
    def compute_multi_signature(self, scene_data: Dict[str, Any]) -> MultiSignature:
        """
        Compute multi-tier signatures for a scene.
        
        Args:
            scene_data: Dictionary containing:
                - objects: List of object dictionaries
                - relations: List of relation dictionaries
                - poses: Dictionary of object poses
                - metadata: Scene metadata
        """
        # Extract components
        objects = scene_data.get('objects', [])
        relations = scene_data.get('relations', [])
        poses = scene_data.get('poses', {})
        
        # Compute object signatures
        object_signatures = {}
        for obj in objects:
            obj_id = obj['id']
            obj_sig = self._compute_object_signature(obj, poses.get(obj_id))
            object_signatures[obj_id] = obj_sig
        
        # Build subgraphs and compute signatures
        subgraphs = self._extract_subgraphs(objects, relations)
        subgraph_signatures = {}
        for subgraph_id, subgraph in subgraphs.items():
            subgraph_sig = self._compute_subgraph_signature(subgraph, poses)
            subgraph_signatures[subgraph_id] = subgraph_sig
        
        # Compute scene signature
        scene_signature = self._compute_scene_signature(
            object_signatures, subgraph_signatures, scene_data.get('metadata', {})
        )
        
        return MultiSignature(
            object_signatures=object_signatures,
            subgraph_signatures=subgraph_signatures,
            scene_signature=scene_signature
        )
    
    def _compute_object_signature(self, obj: Dict[str, Any], pose: Any = None) -> bytes:
        """Compute signature for a single object"""
        hasher = hashlib.blake2b(key=self.salt)
        
        # Object type and attributes
        hasher.update(str(obj.get('type', 'unknown')).encode())
        
        # Quantized pose if available
        if pose is not None:
            quantized_pose = self._quantize_pose(pose)
            hasher.update(quantized_pose.tobytes())
        
        # Other attributes (sorted for determinism)
        attrs = obj.get('attributes', {})
        for key in sorted(attrs.keys()):
            hasher.update(f"{key}:{attrs[key]}".encode())
        
        return hasher.digest()
    
    def _compute_subgraph_signature(self, subgraph: Dict[str, Any], poses: Dict[str, Any]) -> bytes:
        """Compute signature for a subgraph using WL canonicalization"""
        # Extract nodes and edges
        nodes = subgraph.get('nodes', [])
        edges = subgraph.get('edges', [])
        
        # WL canonicalization
        canonical_form = self._wl_canonicalize(nodes, edges, poses)
        
        # Hash canonical form
        hasher = hashlib.blake2b(key=self.salt)
        hasher.update(canonical_form.encode())
        
        return hasher.digest()
    
    def _compute_scene_signature(self, object_sigs: Dict[str, bytes], 
                                subgraph_sigs: Dict[str, bytes],
                                metadata: Dict[str, Any]) -> bytes:
        """Compute scene-level signature from components"""
        hasher = hashlib.blake2b(key=self.salt)
        
        # Hash all object signatures (sorted for determinism)
        for obj_id in sorted(object_sigs.keys()):
            hasher.update(object_sigs[obj_id])
        
        # Hash all subgraph signatures (sorted for determinism)  
        for subgraph_id in sorted(subgraph_sigs.keys()):
            hasher.update(subgraph_sigs[subgraph_id])
        
        # Hash metadata
        for key in sorted(metadata.keys()):
            hasher.update(f"{key}:{metadata[key]}".encode())
        
        return hasher.digest()
    
    def _quantize_pose(self, pose: Any) -> np.ndarray:
        """Quantize pose to fixed bins for stability"""
        if isinstance(pose, dict):
            # Handle pose dictionary
            position = pose.get('position', [0, 0, 0])
            rotation = pose.get('rotation', [0, 0, 0])
            scale = pose.get('scale', [1, 1, 1])
            pose_array = np.array(position + rotation + scale)
        elif isinstance(pose, (list, tuple, np.ndarray)):
            pose_array = np.array(pose)
        else:
            # Default pose
            pose_array = np.zeros(9)
        
        # Quantize to bins
        quantized = np.round(pose_array * self.pose_bins).astype(np.int32)
        return quantized
    
    def _extract_subgraphs(self, objects: List[Dict], relations: List[Dict]) -> Dict[str, Dict]:
        """Extract meaningful subgraphs from scene"""
        # For now, create subgraphs based on connected components
        # This could be made more sophisticated based on domain knowledge
        
        subgraphs = {}
        
        # Build adjacency list
        adjacency = {}
        for obj in objects:
            adjacency[obj['id']] = set()
        
        for rel in relations:
            source = rel.get('source')
            target = rel.get('target')
            if source and target:
                adjacency.setdefault(source, set()).add(target)
                adjacency.setdefault(target, set()).add(source)
        
        # Find connected components
        visited = set()
        component_id = 0
        
        for obj_id in adjacency:
            if obj_id not in visited:
                component = self._dfs_component(obj_id, adjacency, visited)
                if len(component) > 1:  # Only store multi-object subgraphs
                    subgraph_id = f"subgraph_{component_id}"
                    subgraphs[subgraph_id] = {
                        'nodes': [obj for obj in objects if obj['id'] in component],
                        'edges': [rel for rel in relations 
                                if rel.get('source') in component and rel.get('target') in component]
                    }
                    component_id += 1
        
        return subgraphs
    
    def _dfs_component(self, start: str, adjacency: Dict[str, Set[str]], visited: Set[str]) -> Set[str]:
        """DFS to find connected component"""
        component = set()
        stack = [start]
        
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                component.add(node)
                stack.extend(adjacency.get(node, set()) - visited)
        
        return component
    
    def _wl_canonicalize(self, nodes: List[Dict], edges: List[Dict], poses: Dict[str, Any]) -> str:
        """Weisfeiler-Lehman canonicalization of subgraph"""
        # Initialize node labels
        node_labels = {}
        for node in nodes:
            node_id = node['id']
            # Create initial label from node type and quantized pose
            label_parts = [node.get('type', 'unknown')]
            
            if node_id in poses:
                quantized_pose = self._quantize_pose(poses[node_id])
                label_parts.append(str(quantized_pose.tolist()))
            
            node_labels[node_id] = hash(tuple(label_parts))
        
        # Build adjacency with edge types
        adjacency = {}
        for node in nodes:
            adjacency[node['id']] = []
        
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            edge_type = edge.get('type', 'unknown')
            
            if source and target:
                adjacency[source].append((target, edge_type))
                adjacency[target].append((source, edge_type))
        
        # WL iterations
        for _ in range(self.wl_iterations):
            new_labels = {}
            for node_id in node_labels:
                # Collect neighbor labels with edge types
                neighbor_labels = []
                for neighbor_id, edge_type in adjacency.get(node_id, []):
                    neighbor_labels.append((node_labels.get(neighbor_id, 0), edge_type))
                
                # Create new label
                neighbor_labels.sort()  # For determinism
                new_label = hash((node_labels[node_id], tuple(neighbor_labels)))
                new_labels[node_id] = new_label
            
            node_labels = new_labels
        
        # Create canonical string
        canonical_labels = sorted(node_labels.values())
        return str(canonical_labels)
