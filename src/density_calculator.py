"""
Density Calculator Module
Computes weighted traffic density based on vehicle classifications
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict


class DensityCalculator:
    """
    Calculate weighted traffic density from vehicle detections
    """
    
    def __init__(self, weights_config_path: str):
        """
        Initialize density calculator with vehicle weights
        
        Args:
            weights_config_path: Path to vehicle_weights.json
        """
        with open(weights_config_path, 'r') as f:
            config = json.load(f)
        self.vehicle_weights = config['vehicle_weights']
        print(f"✓ Loaded weights for {len(self.vehicle_weights)} vehicle types")
    
    def calculate_frame_density(self, frame_data: Dict) -> Dict:
        """
        Calculate weighted density for a single frame
        
        Args:
            frame_data: Frame detection data with 'detections' and 'roi_area'
            
        Returns:
            Dictionary with density metrics
        """
        detections = frame_data['detections']
        roi_area = frame_data['roi_area']
        
        if roi_area == 0:
            raise ValueError("ROI area cannot be zero")
        
        # Calculate weighted area sum
        total_weighted_area = 0
        total_raw_area = 0
        vehicle_counts = {}
        
        for det in detections:
            vehicle_class = det['class']
            bbox_area = det['bbox_area']
            
            # Get weight (default to 1.0 if class not in config)
            weight = self.vehicle_weights.get(vehicle_class, 1.0)
            
            # Accumulate weighted area
            total_weighted_area += bbox_area * weight
            total_raw_area += bbox_area
            
            # Count vehicles by type
            vehicle_counts[vehicle_class] = vehicle_counts.get(vehicle_class, 0) + 1
        
        # Calculate density metrics
        weighted_density = (total_weighted_area / roi_area) * 100
        raw_density = (total_raw_area / roi_area) * 100
        occupancy_percentage = min(weighted_density, 100.0)  # Cap at 100%
        
        density_metrics = {
            'frame_idx': frame_data['frame_idx'],
            'timestamp': frame_data['timestamp'],
            'vehicle_count': len(detections),
            'vehicle_counts_by_type': vehicle_counts,
            'weighted_density': weighted_density,
            'raw_density': raw_density,
            'occupancy_percentage': occupancy_percentage,
            'total_weighted_area': total_weighted_area,
            'total_raw_area': total_raw_area,
            'roi_area': roi_area
        }
        
        return density_metrics
    
    def calculate_all_frames(self, frame_detections: List[Dict]) -> List[Dict]:
        """
        Calculate density for all frames
        
        Args:
            frame_detections: List of frame detection data
            
        Returns:
            List of density metrics per frame
        """
        density_data = []
        
        for frame_data in frame_detections:
            metrics = self.calculate_frame_density(frame_data)
            density_data.append(metrics)
        
        return density_data
    
    def get_density_statistics(self, density_data: List[Dict]) -> Dict:
        """
        Calculate statistical summary of density metrics
        
        Args:
            density_data: List of density metrics
            
        Returns:
            Dictionary with statistics
        """
        if not density_data:
            return {}
        
        weighted_densities = [d['weighted_density'] for d in density_data]
        vehicle_counts = [d['vehicle_count'] for d in density_data]
        
        # Aggregate vehicle counts by type
        total_by_type = {}
        for frame in density_data:
            for vtype, count in frame['vehicle_counts_by_type'].items():
                total_by_type[vtype] = total_by_type.get(vtype, 0) + count
        
        stats = {
            'total_frames': len(density_data),
            'avg_weighted_density': float(np.mean(weighted_densities)),
            'max_weighted_density': float(np.max(weighted_densities)),
            'min_weighted_density': float(np.min(weighted_densities)),
            'std_weighted_density': float(np.std(weighted_densities)),
            'avg_vehicle_count': float(np.mean(vehicle_counts)),
            'max_vehicle_count': int(np.max(vehicle_counts)),
            'total_vehicles_detected': int(sum(vehicle_counts)),
            'vehicle_distribution': total_by_type
        }
        
        return stats
    
    def save_density_data(self, density_data: List[Dict], output_path: str):
        """
        Save density data to JSON file
        
        Args:
            density_data: List of density metrics
            output_path: Path to save JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(density_data, f, indent=2)
        
        print(f"✓ Density data saved to: {output_file}")
