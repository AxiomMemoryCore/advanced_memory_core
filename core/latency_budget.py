#!/usr/bin/env python3
"""
Latency budgeting system with hard caps and graceful degradation.
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum

class StageBudget(Enum):
    """Predefined stage budgets in milliseconds"""
    SIGNATURE_COMPUTE = 1.0
    L1_CACHE_PROBE = 0.5
    L2_SUBGRAPH_COMPOSE = 4.0
    HDC_BINDING = 2.0
    OPTION_KERNEL_EXEC = 8.0
    FULL_REASONING = 15.0

@dataclass
class LatencyBudget:
    """Latency budget tracker with stage-wise monitoring"""
    total_budget_ms: float = 20.0
    abort_threshold_ms: float = 18.0
    
    def __post_init__(self):
        self.start_time: Optional[float] = None
        self.stage_times: Dict[str, float] = {}
        self.aborted = False
        
    def start_request(self):
        """Start timing a new request"""
        self.start_time = time.perf_counter()
        self.stage_times.clear()
        self.aborted = False
    
    def check_stage_budget(self, stage: StageBudget) -> bool:
        """Check if we have budget for a stage"""
        if not self.start_time:
            return True
            
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        return elapsed_ms + stage.value <= self.abort_threshold_ms
    
    def record_stage(self, stage_name: str):
        """Record completion of a stage"""
        if not self.start_time:
            return
            
        current_time = time.perf_counter()
        elapsed_ms = (current_time - self.start_time) * 1000
        self.stage_times[stage_name] = elapsed_ms
    
    def should_abort(self) -> bool:
        """Check if we should abort the current request"""
        if not self.start_time or self.aborted:
            return self.aborted
            
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        if elapsed_ms > self.abort_threshold_ms:
            self.aborted = True
            
        return self.aborted
    
    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        if not self.start_time:
            return 0.0
        return (time.perf_counter() - self.start_time) * 1000
    
    def get_remaining_ms(self) -> float:
        """Get remaining budget in milliseconds"""
        elapsed = self.get_elapsed_ms()
        return max(0.0, self.total_budget_ms - elapsed)
    
    def get_stage_breakdown(self) -> Dict[str, float]:
        """Get timing breakdown by stage"""
        return self.stage_times.copy()
