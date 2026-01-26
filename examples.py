"""
Example Usage Script
Demonstrates how to use the traffic analysis pipeline
"""

import sys
from pathlib import Path

# Example 1: Basic usage with default settings
print("=" * 70)
print("EXAMPLE 1: Basic Usage")
print("=" * 70)
print("""
# Process a video with default settings (5-minute bins, 0.25 confidence)
python main.py --video data/input/traffic.mp4

# This will:
# - Process all frames
# - Use full frame as ROI
# - Generate 5-minute time bins
# - Save annotated video
# - Create all visualizations
""")

# Example 2: High-performance processing
print("\n" + "=" * 70)
print("EXAMPLE 2: Fast Processing (Skip Frames + No Video)")
print("=" * 70)
print("""
# Process every 3rd frame and skip video annotation for faster processing
python main.py \\
  --video data/input/traffic.mp4 \\
  --frame-skip 3 \\
  --no-video

# Useful for:
# - Quick analysis
# - Long videos
# - Limited computational resources
""")

# Example 3: Custom ROI and time bins
print("\n" + "=" * 70)
print("EXAMPLE 3: Custom ROI and Time Bins")
print("=" * 70)
print("""
# Focus on specific road section with 10-minute bins
python main.py \\
  --video data/input/traffic.mp4 \\
  --roi "200,300,1600,900" \\
  --bin-size 10 \\
  --output data/output/custom_roi

# ROI format: "x1,y1,x2,y2"
# - x1,y1: Top-left corner
# - x2,y2: Bottom-right corner
""")

# Example 4: With datetime for real-world timestamps
print("\n" + "=" * 70)
print("EXAMPLE 4: Real-World Timestamps")
print("=" * 70)
print("""
# Add real datetime to time-series data
python main.py \\
  --video data/input/traffic.mp4 \\
  --start-datetime "2026-01-26T08:00:00" \\
  --bin-size 5

# The output CSV will have actual datetime instead of epoch time
# Useful for correlating with external events (festivals, accidents)
""")

# Example 5: High confidence threshold
print("\n" + "=" * 70)
print("EXAMPLE 5: High Confidence Detections")
print("=" * 70)
print("""
# Only keep high-confidence detections
python main.py \\
  --video data/input/traffic.mp4 \\
  --confidence 0.5

# Default is 0.25
# Higher values = fewer false positives, but may miss some vehicles
# Lower values = more detections, but may include false positives
""")

# Example 6: Complete workflow
print("\n" + "=" * 70)
print("EXAMPLE 6: Complete Production Workflow")
print("=" * 70)
print("""
# Step 1: Authenticate with Hugging Face (one-time setup)
huggingface-cli login

# Step 2: Process video with optimal settings
python main.py \\
  --video data/input/morning_traffic.mp4 \\
  --output data/output/morning_analysis \\
  --roi "100,200,1800,900" \\
  --bin-size 5 \\
  --confidence 0.3 \\
  --start-datetime "2026-01-26T07:00:00"

# Step 3: Use the generated CSV for LSTM training
# The file data/output/morning_analysis/traffic_timeseries.csv is ready!

# Step 4: Review visualizations
# - density_timeline.png: Traffic patterns over time
# - vehicle_distribution.png: Vehicle type breakdown
# - summary_dashboard.png: Complete overview
""")

# Python API usage example
print("\n" + "=" * 70)
print("EXAMPLE 7: Python API Usage")
print("=" * 70)
print("""
# Use the modules directly in your Python code

from src.vehicle_classifier import VehicleClassifier
from src.density_calculator import DensityCalculator
from src.time_series_generator import TimeSeriesGenerator

# Initialize classifier
classifier = VehicleClassifier(confidence_threshold=0.25)
classifier.load_model()

# Process video
detections = classifier.process_video(
    video_path='data/input/traffic.mp4',
    output_dir='data/output',
    roi=(100, 200, 1800, 900)
)

# Calculate density
calculator = DensityCalculator('config/vehicle_weights.json')
density_data = calculator.calculate_all_frames(detections)

# Generate time-series
ts_gen = TimeSeriesGenerator(bin_size_minutes=5)
ts_df = ts_gen.aggregate_to_timeseries(density_data)
ts_gen.save_timeseries(ts_df, 'data/output/timeseries.csv')
""")

print("\n" + "=" * 70)
print("For more information, see README.md")
print("=" * 70)
