
import numpy as np
import time
from typing import Dict, List, Optional

class TrafficSimulator:
    """
    Stochastic Traffic Simulator using Poisson Distribution
    Simulates a single lane/path of traffic.
    """
    
    def __init__(self, lane_id: str, arrival_rate_per_min: float = 10.0):
        """
        Args:
            lane_id: Identifier for the lane (e.g., "North", "South")
            arrival_rate_per_min: Lambda for Poisson distribution (vehicles/min)
        """
        self.lane_id = lane_id
        self.arrival_rate = arrival_rate_per_min
        self.queue_length = 0
        self.vehicles_passed = 0
        self.last_update_time = time.time()
        
        # Stochastic parameters (Probability of vehicle types)
        self.vehicle_probs = {
            "Car": 0.5,
            "Bike": 0.3,
            "Bus": 0.1,
            "Truck": 0.1
        }
        
    def step(self, duration_seconds: float = 1.0) -> Dict:
        """
        Advance simulation by duration_seconds.
        Returns state dictionary.
        """
        # 1. Stochastic Arrival (Poisson Process)
        # Expected new vehicles in this duration
        lam = (self.arrival_rate / 60.0) * duration_seconds
        new_vehicles = np.random.poisson(lam)
        
        self.queue_length += new_vehicles
        
        # 2. Stochastic attributes for new vehicles (for visualization/stats)
        new_vehicle_types = []
        if new_vehicles > 0:
            types = list(self.vehicle_probs.keys())
            probs = list(self.vehicle_probs.values())
            new_vehicle_types = np.random.choice(types, size=new_vehicles, p=probs).tolist()
            
        return {
            "lane_id": self.lane_id,
            "new_arrivals": new_vehicles,
            "queue_length": self.queue_length,
            "new_types": new_vehicle_types
        }

    def process_traffic(self, green_light_duration: float, discharge_rate_per_sec: float = 0.5):
        """
        Simulate traffic flowing out during Green light (Discrete Event: DEPARTURE)
        """
        # Ensure at least 1 vehicle leaves if duration > 0.5s, avoiding int(0.5) = 0 stagnation
        # Or use ceil/probabilistic. Simple fix: max(1, ...)
        calculated_discharge = int(green_light_duration * discharge_rate_per_sec)
        max_discharge = max(1, calculated_discharge) if green_light_duration >= 1.0 else calculated_discharge
        
        discharged = min(self.queue_length, max_discharge)
        self.queue_length -= discharged
        self.vehicles_passed += discharged
        
        return discharged
