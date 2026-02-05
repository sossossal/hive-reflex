#!/usr/bin/env python3
"""
Advanced Memory Planner for CIM Compiler
========================================

Implements Liveness Analysis and Linear Scan Allocation to minimize SRAM usage 
and correctly handle complex graph topologies (e.g., Residual Connections).

Problem with simple ping-pong buffering:
    Layer 1 -> Layer 2 -> Layer 3
       |                    ^
       ---------------------|
    (Residual Connection)

If:
    L1 writes to Buffer A
    L2 reads A, writes B
    L3 reads B AND A (Residual) -> Needs A to be alive!

    If we simplisticly reuse A for L2's output (ping-pong), L3 reads garbage.

Solution:
    1. Analyze lifetime of every tensor (start_layer, end_layer).
    2. Allocate offsets in a shared Scratchpad memory.
    3. Two tensors can share memory IFF their lifetimes do not overlap.
"""

from typing import List, Dict, Set
import collections

class MemoryPlanner:
    def __init__(self, layers: List[Dict]):
        self.layers = layers
        self.allocations = {} # tensor_name -> offset
        self.total_size = 0
        
    def plan(self):
        """Execute memory planning"""
        print("  [MemoryPlanner] Starting Liveness Analysis...")
        lifetimes = self._analyze_liveness()
        
        print("  [MemoryPlanner] Allocating Buffers...")
        self.total_size = self._allocate_buffers(lifetimes)
        
        print(f"  [MemoryPlanner] Total Scratchpad Size: {self.total_size / 1024:.2f} KB")
        return self.allocations, self.total_size

    def _analyze_liveness(self) -> Dict[str, Dict]:
        """
        Determine [start, end] index for each tensor.
        start: index of layer producing the tensor
        end: index of last layer consuming the tensor
        """
        lifetimes = {} # tensor_name -> {'start': int, 'end': int, 'size': int}
        
        # 1. Initialize with production time (start)
        for i, layer in enumerate(self.layers):
            for out_name in layer['outputs']:
                size = self._calculate_size(layer) # Simplified: assume output size = layer size
                lifetimes[out_name] = {'start': i, 'end': i, 'size': size}
                
        # 2. Update consumption time (end)
        for i, layer in enumerate(self.layers):
            for in_name in layer['inputs']:
                if in_name in lifetimes:
                    lifetimes[in_name]['end'] = max(lifetimes[in_name]['end'], i)
                    
        # Debug output
        # for name, span in lifetimes.items():
        #     print(f"    Tensor '{name}': Live {span['start']} -> {span['end']} ({span['size']} bytes)")
            
        return lifetimes

    def _allocate_buffers(self, lifetimes: Dict[str, Dict]) -> int:
        """
        Linear Scan Allocation (simplified).
        Sort intervals by start time, assign free blocks.
        """
        # Sort by start time
        sorted_tensors = sorted(lifetimes.items(), key=lambda x: x[1]['start'])
        
        active_allocations = [] # (end_time, offset, size)
        max_offset = 0
        
        for name, info in sorted_tensors:
            start, end, size = info['start'], info['end'], info['size']
            
            # Expire old allocations
            # Remove tensors whose end time < current start time
            # In a real compiler, we need more care (end is inclusive consumption).
            # So a tensor is free AFTER the last consumer finishes.
            # Here: if end < start, it's definitely free.
            active_allocations = [a for a in active_allocations if a[0] >= start]
            
            # Find a gap? (Greedy First-Fit)
            # For simplicity in this script, we just append to "virtual free list" or stack
            # A true Linear Scan would maintain a list of free ranges.
            
            # Simple strategy: Check if we can reuse any expired slot? 
            # Actually, let's implement a 'watermark' strategy for simplicity first, 
            # but that doesn't save memory.
            
            # Better strategy: Track 'free_blocks' [(offset, size)]
            # But let's stick to a robust implementation:
            # Checking overlap with all active allocations to find a free offset.
            
            candidate_offset = 0
            while True:
                collision = False
                candidate_end = candidate_offset + size
                
                for _, alloc_offset, alloc_size in active_allocations:
                    # Check overlap [candidate_offset, candidate_end) vs [alloc_offset, alloc_offset+alloc_size)
                    op_start = max(candidate_offset, alloc_offset)
                    op_end = min(candidate_end, alloc_offset + alloc_size)
                    if op_start < op_end:
                        collision = True
                        candidate_offset = alloc_offset + alloc_size # Jump past this block
                        break
                
                if not collision:
                    break
            
            # Found space
            self.allocations[name] = candidate_offset
            active_allocations.append((end, candidate_offset, size))
            max_offset = max(max_offset, candidate_offset + size)
            
        return max_offset

    def _calculate_size(self, layer) -> int:
        """Calculate tensor size in bytes (float32)"""
        # Parse shape string "(1, 64)" or list [1, 64]
        shape = layer.get('shape', '1')
        if isinstance(shape, str):
            import re
            nums = re.findall(r'\d+', shape)
            count = 1
            for n in nums: count *= int(n)
        elif isinstance(shape, (list, tuple, np.ndarray)):
             count = np.prod(shape)
        else:
             count = 256 # Fallback
             
        # LSTM output is hidden_size
        if layer['type'] == 'lstm':
            count = layer.get('hidden_size', 16)
            
        # If explicit shape is missing (e.g. activation), infer from previous?
        # In this standalone planner, we rely on 'shape' being propagated.
        return int(count) * 4 # 4 bytes for float32
    
import numpy as np
