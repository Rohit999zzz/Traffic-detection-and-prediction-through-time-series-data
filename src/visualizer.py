"""
Visualization Module
Create plots and visualizations for traffic analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


class TrafficVisualizer:
    """
    Generate visualizations for traffic analysis results
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        sns.set_palette("husl")
        self.colors = sns.color_palette("husl", 13)
    
    def plot_density_timeline(self, ts_df: pd.DataFrame, output_path: str):
        """
        Plot density over time
        
        Args:
            ts_df: Time-series DataFrame
            output_path: Path to save plot
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Plot 1: Vehicle count over time
        axes[0].plot(ts_df['bin_id'], ts_df['total_vehicle_count'], 
                    marker='o', linewidth=2, markersize=4, color='#2E86AB')
        axes[0].fill_between(ts_df['bin_id'], ts_df['total_vehicle_count'], 
                            alpha=0.3, color='#2E86AB')
        axes[0].set_xlabel('Time Bin', fontsize=11)
        axes[0].set_ylabel('Total Vehicle Count', fontsize=11)
        axes[0].set_title('Traffic Volume Over Time', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Weighted density
        axes[1].plot(ts_df['bin_id'], ts_df['avg_weighted_density'], 
                    label='Average Density', linewidth=2, color='#A23B72')
        axes[1].plot(ts_df['bin_id'], ts_df['max_weighted_density'], 
                    label='Peak Density', linewidth=2, linestyle='--', color='#F18F01')
        axes[1].fill_between(ts_df['bin_id'], 
                            ts_df['min_weighted_density'],
                            ts_df['max_weighted_density'], 
                            alpha=0.2, color='#A23B72')
        axes[1].set_xlabel('Time Bin', fontsize=11)
        axes[1].set_ylabel('Weighted Density (%)', fontsize=11)
        axes[1].set_title('Road Occupancy Density', fontsize=13, fontweight='bold')
        axes[1].legend(loc='best')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Occupancy percentage
        axes[2].plot(ts_df['bin_id'], ts_df['avg_occupancy_percentage'], 
                    linewidth=2, color='#06A77D')
        axes[2].fill_between(ts_df['bin_id'], ts_df['avg_occupancy_percentage'], 
                            alpha=0.3, color='#06A77D')
        axes[2].axhline(y=50, color='orange', linestyle='--', 
                       label='Moderate Congestion (50%)', linewidth=1.5)
        axes[2].axhline(y=75, color='red', linestyle='--', 
                       label='High Congestion (75%)', linewidth=1.5)
        axes[2].set_xlabel('Time Bin', fontsize=11)
        axes[2].set_ylabel('Occupancy (%)', fontsize=11)
        axes[2].set_title('Road Occupancy Percentage', fontsize=13, fontweight='bold')
        axes[2].legend(loc='best')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Density timeline saved to: {output_path}")
    
    def plot_vehicle_distribution(self, ts_df: pd.DataFrame, output_path: str):
        """
        Plot vehicle type distribution
        
        Args:
            ts_df: Time-series DataFrame
            output_path: Path to save plot
        """
        # Get vehicle type columns
        vehicle_cols = [col for col in ts_df.columns if col.endswith('_count') 
                       and col != 'total_vehicle_count']
        
        if not vehicle_cols:
            print("⚠ No vehicle type data found for distribution plot")
            return
        
        # Calculate totals
        vehicle_totals = {}
        for col in vehicle_cols:
            vtype = col.replace('_count', '')
            vehicle_totals[vtype] = ts_df[col].sum()
        
        # Sort by count
        vehicle_totals = dict(sorted(vehicle_totals.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        # Filter out zero counts
        vehicle_totals = {k: v for k, v in vehicle_totals.items() if v > 0}
        
        if not vehicle_totals:
            print("⚠ No vehicles detected for distribution plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar chart
        vehicles = list(vehicle_totals.keys())
        counts = list(vehicle_totals.values())
        colors_list = self.colors[:len(vehicles)]
        
        axes[0].bar(vehicles, counts, color=colors_list, edgecolor='black', linewidth=1.2)
        axes[0].set_xlabel('Vehicle Type', fontsize=12)
        axes[0].set_ylabel('Total Count', fontsize=12)
        axes[0].set_title('Vehicle Type Distribution', fontsize=14, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (v, c) in enumerate(zip(vehicles, counts)):
            axes[0].text(i, c + max(counts)*0.01, str(int(c)), 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Pie chart
        axes[1].pie(counts, labels=vehicles, autopct='%1.1f%%', 
                   colors=colors_list, startangle=90, 
                   textprops={'fontsize': 10, 'fontweight': 'bold'})
        axes[1].set_title('Vehicle Type Proportion', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Vehicle distribution saved to: {output_path}")
    
    def plot_heatmap(self, ts_df: pd.DataFrame, output_path: str):
        """
        Plot density heatmap over time
        
        Args:
            ts_df: Time-series DataFrame
            output_path: Path to save plot
        """
        # Reshape data for heatmap (bins vs metrics)
        vehicle_cols = [col for col in ts_df.columns if col.endswith('_count') 
                       and col != 'total_vehicle_count']
        
        if not vehicle_cols:
            print("⚠ No vehicle type data found for heatmap")
            return
        
        # Create matrix: rows = time bins, columns = vehicle types
        heatmap_data = ts_df[vehicle_cols].T
        heatmap_data.columns = ts_df['bin_id'].values
        heatmap_data.index = [col.replace('_count', '') for col in vehicle_cols]
        
        # Filter out all-zero rows
        heatmap_data = heatmap_data.loc[(heatmap_data != 0).any(axis=1)]
        
        if heatmap_data.empty:
            print("⚠ No data for heatmap")
            return
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, 
                   fmt='d', linewidths=0.5, cbar_kws={'label': 'Vehicle Count'},
                   ax=ax)
        
        ax.set_xlabel('Time Bin', fontsize=12)
        ax.set_ylabel('Vehicle Type', fontsize=12)
        ax.set_title('Vehicle Type Distribution Across Time Bins', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Heatmap saved to: {output_path}")
    
    def create_summary_dashboard(self, ts_df: pd.DataFrame, 
                                 summary_stats: Dict, 
                                 output_path: str):
        """
        Create comprehensive summary dashboard
        
        Args:
            ts_df: Time-series DataFrame
            summary_stats: Summary statistics dictionary
            output_path: Path to save dashboard
        """
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Traffic Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Traffic volume timeline
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(ts_df['bin_id'], ts_df['total_vehicle_count'], 
                linewidth=2, color='#2E86AB', marker='o', markersize=3)
        ax1.fill_between(ts_df['bin_id'], ts_df['total_vehicle_count'], 
                        alpha=0.3, color='#2E86AB')
        ax1.set_title('Traffic Volume Timeline', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Bin')
        ax1.set_ylabel('Vehicle Count')
        ax1.grid(True, alpha=0.3)
        
        # 2. Density over time
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.plot(ts_df['bin_id'], ts_df['avg_weighted_density'], 
                linewidth=2, label='Avg Density', color='#A23B72')
        ax2.plot(ts_df['bin_id'], ts_df['max_weighted_density'], 
                linewidth=2, linestyle='--', label='Peak Density', color='#F18F01')
        ax2.set_title('Weighted Density', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time Bin')
        ax2.set_ylabel('Density (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Statistics box
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.axis('off')
        stats_text = f"""
        SUMMARY STATISTICS
        
        Total Bins: {summary_stats['total_bins']}
        Duration: {summary_stats['total_duration_minutes']:.1f} min
        
        Avg Vehicles/Bin: {summary_stats['avg_vehicles_per_bin']:.1f}
        Max Vehicles: {summary_stats['max_vehicles_in_bin']}
        
        Avg Density: {summary_stats['avg_density']:.2f}%
        Max Density: {summary_stats['max_density']:.2f}%
        
        Avg Occupancy: {summary_stats['avg_occupancy']:.2f}%
        Peak Occupancy: {summary_stats['peak_occupancy']:.2f}%
        """
        ax3.text(0.1, 0.5, stats_text, fontsize=10, 
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 4. Vehicle distribution pie
        vehicle_dist = summary_stats.get('vehicle_type_distribution', {})
        if vehicle_dist:
            ax4 = fig.add_subplot(gs[2, :2])
            vehicles = list(vehicle_dist.keys())
            counts = list(vehicle_dist.values())
            colors_list = self.colors[:len(vehicles)]
            ax4.pie(counts, labels=vehicles, autopct='%1.1f%%', 
                   colors=colors_list, startangle=90)
            ax4.set_title('Vehicle Type Distribution', fontsize=12, fontweight='bold')
        
        # 5. Top vehicles bar
        if vehicle_dist:
            ax5 = fig.add_subplot(gs[2, 2])
            top_vehicles = dict(sorted(vehicle_dist.items(), 
                                      key=lambda x: x[1], reverse=True)[:5])
            ax5.barh(list(top_vehicles.keys()), list(top_vehicles.values()), 
                    color=self.colors[:len(top_vehicles)])
            ax5.set_title('Top 5 Vehicle Types', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Count')
            ax5.grid(axis='x', alpha=0.3)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Summary dashboard saved to: {output_path}")
