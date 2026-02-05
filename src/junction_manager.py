
from typing import Dict, List, Optional
import time
from traffic_simulator import TrafficSimulator

class JunctionManager:
    """
    Manages a 4-way junction with Discrete Event Logic and Weighted Priority.
    
    Improvements:
    - Separation of Concerns: Arrivals, Decisions, Departures
    - Simulation Time: Uses internal clock instead of wall-clock
    - Weighted Priority: Queue Length + (Alpha * Wait Time)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # Default Configuration
        self.config = {
            "min_green_time": 5.0,
            "max_green_time": 30.0,
            "queue_threshold": 15,
            "max_wait_time": 45.0,
            "alpha_wait_weight": 0.5, # Weight for wait time in priority score
            "arrival_rates": {
                "North": 12, "South": 10, "East": 8, "West": 15
            }
        }
        if config:
            self.config.update(config)

        # 4 Lanes
        rates = self.config["arrival_rates"]
        self.lanes = {
            lid: TrafficSimulator(lid, arrival_rate_per_min=rate) 
            for lid, rate in rates.items()
        }
        
        # State
        self.active_lane_id = "North"
        self.sim_time = 0.0
        self.last_switch_time = 0.0
        
        # Track when each lane was last green (sim_time)
        self.lane_last_green_times = {lid: 0.0 for lid in self.lanes}
        
        self.events_log = [] 

    def step(self, dt: float = 1.0) -> Dict:
        """
        Advance simulation by dt.
        Returns state dictionary.
        """
        # 1. Update Arrivals
        lane_states = self.update_arrivals(dt)
        
        # 2. Update Simulation Time
        self.sim_time += dt
        
        # 3. Decision Logic (State Change)
        switch_event = self.evaluate_switch_conditions()
        
        # 4. Process Departures
        self.process_departures(dt)
        
        return {
            "lane_states": lane_states,
            "active_lane": self.active_lane_id,
            "green_time": self.sim_time - self.last_switch_time,
            "switch_event": switch_event,
            "sim_time": self.sim_time
        }

    def update_arrivals(self, dt: float) -> Dict:
        states = {}
        for lid, lane in self.lanes.items():
            states[lid] = lane.step(dt)
        return states

    def evaluate_switch_conditions(self) -> Optional[str]:
        """
        Determines if a switch is needed based on Weighted Priority & Constraints.
        """
        state_duration = self.sim_time - self.last_switch_time
        
        # Constraints
        if state_duration < self.config["min_green_time"]:
            return None # Must hold green
            
        must_switch = state_duration >= self.config["max_green_time"]
        
        # Calculate Priority Scores for ALL lanes
        # Score = Queue + (Alpha * WaitTime)
        scores = {}
        for lid, lane in self.lanes.items():
            wait_time = self.sim_time - self.lane_last_green_times[lid]
            if lid == self.active_lane_id:
                wait_time = 0 # Currently active, no wait penalty
            
            score = lane.queue_length + (self.config["alpha_wait_weight"] * wait_time)
            scores[lid] = score
            
        # Find highest priority candidate (excluding current if possible)
        candidates = {k: v for k, v in scores.items() if k != self.active_lane_id}
        if not candidates:
            return None
            
        best_candidate = max(candidates, key=candidates.get)
        best_score = candidates[best_candidate]
        current_score = scores[self.active_lane_id]
        
        current_lane_obj = self.lanes[self.active_lane_id]
        
        # Decision Logic
        trigger_reason = ""
        
        # 1. Empty Lane Optimization (Highest Level)
        if current_lane_obj.queue_length <= 0 and best_score > 5:
             trigger_reason = f"LANE_EMPTY ({self.active_lane_id})"
             
        # 2. Hard Limits (Max Time)
        elif must_switch:
            trigger_reason = f"MAX_TIME_EXPIRED (Active Score: {current_score:.1f})"
            
        # 3. Weighted Priority Switch
        # Switch if Challenger Score is significantly higher than Current
        # e.g., Challenger > Current * 1.5 OR Challenger > Current + Threshold
        elif best_score > (current_score + self.config["queue_threshold"]):
             trigger_reason = f"PRIORITY_OVERRIDE (Score {best_score:.1f} > {current_score:.1f})"
             
        if trigger_reason:
            return self.execute_switch(best_candidate, trigger_reason)
            
        return None

    def execute_switch(self, target_lane: str, reason: str) -> str:
        self.events_log.append({
            "time": f"{self.sim_time:.1f}s",
            "event": "SHIFT_PRIORITY",
            "from": self.active_lane_id,
            "to": target_lane,
            "reason": reason
        })
        self.active_lane_id = target_lane
        self.last_switch_time = self.sim_time
        self.lane_last_green_times[target_lane] = self.sim_time
        return reason

    def process_departures(self, dt: float):
        active_lane = self.lanes[self.active_lane_id]
        active_lane.process_traffic(dt)
