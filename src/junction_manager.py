
from typing import Dict, List, Optional
import time
from traffic_simulator import TrafficSimulator

class JunctionManager:
    """
    Manages a 4-way junction with Discrete Event Logic for signal control.
    """
    
    def __init__(self):
        # 4 Lanes (North, South, East, West)
        self.lanes = {
            "North": TrafficSimulator("North", arrival_rate_per_min=12),
            "South": TrafficSimulator("South", arrival_rate_per_min=10),
            "East": TrafficSimulator("East", arrival_rate_per_min=8),
            "West": TrafficSimulator("West", arrival_rate_per_min=15)
        }
        
        # State: Which lane is green?
        self.active_lane_id = "North"
        self.last_switch_time = time.time()
        
        # Policy Parameters (Discrete Event Thresholds)
        self.min_green_time = 5.0   # Minimum time a light must stay green
        self.max_green_time = 30.0  # Max time before forced switch
        self.queue_threshold = 15   # If another lane exceeds this, consider switching
        
        self.events_log = [] # Log discrete events

    def step(self, dt: float = 1.0) -> Dict:
        """
        Discrete Time Step:
        1. Update traffic arrivals (Stochastic)
        2. Check for State Change Events (Discrete Event Logic)
        3. Process Departures (Flow)
        """
        current_time = time.time()
        state_duration = current_time - self.last_switch_time
        
        # 1. Update Arrivals
        lane_states = {}
        for lane_id, lane in self.lanes.items():
            lane_states[lane_id] = lane.step(dt)
            
        # 2. Discrete Event Logic for Shifting
        # Default: No change
        switch_event = None
        
        # Check constraints
        can_switch = state_duration >= self.min_green_time
        must_switch = state_duration >= self.max_green_time
        
        if can_switch:
            # Check for "Queue Spike" Event in other lanes
            max_queue_lane = None
            max_queue_val = 0
            
            for lane_id, lane in self.lanes.items():
                if lane_id == self.active_lane_id:
                    continue
                if lane.queue_length > max_queue_val:
                    max_queue_val = lane.queue_length
                    max_queue_lane = lane_id
            
            # Logic: Switch if MUST switch OR (Queue > Threshold AND current queue is low)
            current_lane_queue = self.lanes[self.active_lane_id].queue_length
            
            trigger_reason = ""
            if must_switch:
                trigger_reason = "MAX_TIME_EXPIRED"
            elif max_queue_val > self.queue_threshold and current_lane_queue < (max_queue_val / 2):
                trigger_reason = f"QUEUE_SPIKE_DETECTED ({max_queue_lane}: {max_queue_val})"
            
            # Execute Transition
            if trigger_reason and max_queue_lane:
                self.events_log.append({
                    "time": time.strftime("%H:%M:%S"),
                    "event": "SHIFT_PRIORITY",
                    "from": self.active_lane_id,
                    "to": max_queue_lane,
                    "reason": trigger_reason
                })
                self.active_lane_id = max_queue_lane
                self.last_switch_time = current_time
                switch_event = trigger_reason

        # 3. Process Traffic Flow (Active lane discharges vehicles)
        active_lane = self.lanes[self.active_lane_id]
        active_lane.process_traffic(dt)
        
        return {
            "lane_states": lane_states,
            "active_lane": self.active_lane_id,
            "green_time": state_duration,
            "switch_event": switch_event
        }
