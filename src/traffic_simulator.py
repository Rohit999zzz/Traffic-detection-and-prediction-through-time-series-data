
import numpy as np
import time
from typing import Dict, List, Optional

class TrafficSimulator:
    """
    Stochastic Traffic Simulator using Poisson Distribution and Rich Vehicle Objects.
    
    Improvements:
    - Simulation Clock (sim_time)
    - Vehicle Attributes (Type, Arrival Time)
    - Variable Discharge Rates (Trucks slower than Cars)
    """
    
    def __init__(self, lane_id: str, arrival_rate_per_min: float = 10.0, 
                 vehicle_probs: Optional[Dict[str, float]] = None):
        """
        Args:
            lane_id: Identifier for the lane
            arrival_rate_per_min: Lambda for Poisson distribution
            vehicle_probs: Dictionary of vehicle type probabilities
        """
        self.lane_id = lane_id
        self.arrival_rate = arrival_rate_per_min
        
        # Simulation Logic
        self.sim_time = 0.0
        self.queue: List[Dict] = [] # List of {"type": str, "arrival_time": float}
        self.vehicles_passed = 0
        
        # Configuration
        self.vehicle_probs = vehicle_probs or {
            "Car": 0.5,
            "Bike": 0.3,
            "Bus": 0.1,
            "Truck": 0.1
        }
        
        # Discharge Rates (seconds needed to clear intersection)
        self.discharge_costs = {
            "Car": 2.0,    # Takes 2s to clear
            "Bike": 1.0,   # Fast
            "Bus": 3.0,    # Slow
            "Truck": 4.0   # Very Slow
        }
        
    @property
    def queue_length(self) -> int:
        """Backward compatibility for JunctionManager"""
        return len(self.queue)
        
    def step(self, duration_seconds: float = 1.0) -> Dict:
        """
        Advance simulation by duration_seconds.
        """
        self.sim_time += duration_seconds
        
        # 1. Stochastic Arrival (Poisson Process)
        lam = (self.arrival_rate / 60.0) * duration_seconds
        new_vehicles_count = np.random.poisson(lam)
        
        new_vehicle_types = []
        if new_vehicles_count > 0:
            types = list(self.vehicle_probs.keys())
            probs = list(self.vehicle_probs.values())
            # Normalize probs just in case
            prob_sum = sum(probs)
            probs = [p/prob_sum for p in probs]
            
            new_vehicle_types = np.random.choice(types, size=new_vehicles_count, p=probs).tolist()
            
            # Add to Queue
            for v_type in new_vehicle_types:
                self.queue.append({
                    "type": v_type,
                    "arrival_time": self.sim_time
                })
        
        return {
            "lane_id": self.lane_id,
            "new_arrivals": new_vehicles_count,
            "queue_length": len(self.queue),
            "new_types": new_vehicle_types
        }

    def process_traffic(self, green_light_duration: float):
        """
        Simulate traffic flowing out based on vehicle types.
        Returns detailed stats.
        """
        time_remaining = green_light_duration
        discharged_count = 0
        discharged_types = []
        total_wait_time = 0.0
        
        # Process queue until time runs out or queue empty
        while self.queue and time_remaining > 0:
            next_vehicle = self.queue[0]
            v_type = next_vehicle["type"]
            cost = self.discharge_costs.get(v_type, 2.0)
            
            # Can this vehicle clear?
            # We allow at least 1 vehicle if time > 0.5s to prevent lockups
            if time_remaining >= cost or (discharged_count == 0 and time_remaining > 0.5):
                # Discharge
                v = self.queue.pop(0)
                time_remaining -= cost
                discharged_count += 1
                discharged_types.append(v_type)
                
                # Stats
                wait_time = self.sim_time - v["arrival_time"]
                total_wait_time += wait_time
            else:
                break
                
        self.vehicles_passed += discharged_count
        
        return {
            "discharged_count": discharged_count,
            "discharged_types": discharged_types,
            "avg_wait_time": (total_wait_time / discharged_count) if discharged_count > 0 else 0.0
        }
