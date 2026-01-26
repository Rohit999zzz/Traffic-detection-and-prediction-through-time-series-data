"""
Time Series Generator Module
Aggregates vehicle detections into time-series dataset for forecasting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timedelta


class TimeSeriesGenerator:
    """
    Generate time-series dataset from density data
    """
    
    def __init__(self, bin_size_minutes: float = 5):
        """
        Initialize time-series generator
        
        Args:
            bin_size_minutes: Temporal bin size in minutes (default: 5). 
                            Can be float for sub-minute bins (e.g. 1/60 for 1 second).
        """
        self.bin_size_minutes = bin_size_minutes
        self.bin_size_seconds = int(bin_size_minutes * 60)
        # Ensure at least 1 second if non-zero
        if self.bin_size_seconds < 1 and bin_size_minutes > 0:
            self.bin_size_seconds = 1
    
    def aggregate_to_timeseries(self, density_data: List[Dict], 
                                start_datetime: str = None) -> pd.DataFrame:
        """
        Aggregate frame-level density data into time-series bins
        
        Args:
            density_data: List of density metrics per frame
            start_datetime: Starting datetime (ISO format). If None, uses epoch
            
        Returns:
            DataFrame with time-series aggregated data
        """
        if not density_data:
            raise ValueError("No density data provided")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(density_data)
        
        # Create time bins
        df['time_bin'] = (df['timestamp'] // self.bin_size_seconds).astype(int)
        
        # Initialize aggregation results
        timeseries_data = []
        
        # Group by time bin
        for bin_id, group in df.groupby('time_bin'):
            bin_start_time = bin_id * self.bin_size_seconds
            
            # Aggregate vehicle counts by type
            vehicle_type_counts = {}
            for _, row in group.iterrows():
                for vtype, count in row['vehicle_counts_by_type'].items():
                    vehicle_type_counts[vtype] = vehicle_type_counts.get(vtype, 0) + count
            
            # Calculate aggregate metrics
            bin_data = {
                'bin_id': bin_id,
                'timestamp_seconds': bin_start_time,
                'total_vehicle_count': group['vehicle_count'].sum(),
                'avg_vehicle_count': group['vehicle_count'].mean(),
                'max_vehicle_count': group['vehicle_count'].max(),
                'avg_weighted_density': group['weighted_density'].mean(),
                'max_weighted_density': group['weighted_density'].max(),
                'min_weighted_density': group['weighted_density'].min(),
                'avg_occupancy_percentage': group['occupancy_percentage'].mean(),
                'max_occupancy_percentage': group['occupancy_percentage'].max(),
                'frames_in_bin': len(group)
            }
            
            # Add per-vehicle-type counts
            for vtype, count in vehicle_type_counts.items():
                bin_data[f'{vtype}_count'] = count
            
            timeseries_data.append(bin_data)
        
        # Create DataFrame
        ts_df = pd.DataFrame(timeseries_data)
        
        # Fill missing vehicle type columns with 0
        vehicle_types = set()
        for data in density_data:
            vehicle_types.update(data['vehicle_counts_by_type'].keys())
        
        for vtype in vehicle_types:
            col_name = f'{vtype}_count'
            if col_name not in ts_df.columns:
                ts_df[col_name] = 0
            else:
                ts_df[col_name] = ts_df[col_name].fillna(0).astype(int)
        
        # Convert timestamp to datetime if start_datetime provided
        if start_datetime:
            base_dt = pd.to_datetime(start_datetime)
            ts_df['datetime'] = base_dt + pd.to_timedelta(ts_df['timestamp_seconds'], unit='s')
        else:
            # Use epoch-based datetime
            ts_df['datetime'] = pd.to_datetime(ts_df['timestamp_seconds'], unit='s')
        
        # Reorder columns
        cols = ['bin_id', 'datetime', 'timestamp_seconds', 'total_vehicle_count', 
                'avg_vehicle_count', 'max_vehicle_count']
        
        # Add vehicle type columns - filtering out aggregate columns to avoid duplicates
        excluded_cols = {'total_vehicle_count', 'avg_vehicle_count', 'max_vehicle_count'}
        vehicle_cols = [col for col in ts_df.columns if col.endswith('_count') and col not in excluded_cols]
        vehicle_cols.sort()
        cols.extend(vehicle_cols)
        
        # Add density columns
        density_cols = ['avg_weighted_density', 'max_weighted_density', 'min_weighted_density',
                       'avg_occupancy_percentage', 'max_occupancy_percentage', 'frames_in_bin']
        cols.extend(density_cols)
        
        ts_df = ts_df[cols]
        
        return ts_df
    
    def save_timeseries(self, ts_df: pd.DataFrame, output_path: str):
        """
        Save time-series DataFrame to CSV
        
        Args:
            ts_df: Time-series DataFrame
            output_path: Path to save CSV file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        ts_df.to_csv(output_file, index=False)
        print(f"✓ Time-series dataset saved to: {output_file}")
        print(f"  Shape: {ts_df.shape[0]} rows × {ts_df.shape[1]} columns")
        print(f"  Time bins: {self.bin_size_seconds} seconds each")
    
    def get_timeseries_summary(self, ts_df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of time-series data
        
        Args:
            ts_df: Time-series DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_bins': len(ts_df),
            'bin_size_minutes': float(self.bin_size_minutes),
            'total_duration_minutes': float(len(ts_df) * self.bin_size_minutes),
            'avg_vehicles_per_bin': float(ts_df['total_vehicle_count'].mean()),
            'max_vehicles_in_bin': int(ts_df['total_vehicle_count'].max()),
            'avg_density': float(ts_df['avg_weighted_density'].mean()),
            'max_density': float(ts_df['max_weighted_density'].max()),
            'avg_occupancy': float(ts_df['avg_occupancy_percentage'].mean()),
            'peak_occupancy': float(ts_df['max_occupancy_percentage'].max())
        }
        
        # Add vehicle type distribution
        vehicle_cols = [col for col in ts_df.columns if col.endswith('_count') 
                       and col != 'total_vehicle_count']
        vehicle_distribution = {}
        for col in vehicle_cols:
            vtype = col.replace('_count', '')
            vehicle_distribution[vtype] = int(ts_df[col].sum())
        
        summary['vehicle_type_distribution'] = vehicle_distribution
        
        return summary
