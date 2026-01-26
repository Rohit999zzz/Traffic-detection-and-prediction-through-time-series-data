# Traffic Analysis Pipeline

Real-time vehicle classification and time-series traffic density analysis using VehicleNet-Y26x from Hugging Face.

## Overview

This project implements the vehicle classification component of an intelligent traffic forecasting system. It processes traffic video to:
- Classify vehicles into 13 categories using VehicleNet-Y26x YOLO model
- Calculate weighted traffic density based on vehicle types
- Generate time-series datasets for LSTM forecasting
- Create comprehensive visualizations

## Vehicle Categories (13 Types)

1. Hatchback
2. Sedan
3. SUV
4. MUV (Multi Utility Vehicle)
5. Bus
6. Truck
7. Three-Wheeler
8. Two-Wheeler
9. LCV (Light Commercial Vehicle)
10. Mini-bus
11. Tempo-traveller
12. Bicycle
13. Van

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Authenticate with Hugging Face (required for model access)
huggingface-cli login
```

## Usage

### Basic Usage

```bash
python main.py --video path/to/your/video.mp4
```

### Advanced Usage

```bash
python main.py \
  --video data/input/traffic.mp4 \
  --output data/output \
  --bin-size 5 \
  --confidence 0.25 \
  --frame-skip 1 \
  --roi "100,200,1800,900" \
  --start-datetime "2026-01-26T08:00:00"
```

### Parameters

- `--video`: Path to input video file (required)
- `--output`: Output directory (default: `data/output`)
- `--bin-size`: Time bin size in minutes (default: 5)
- `--confidence`: Detection confidence threshold (default: 0.25)
- `--frame-skip`: Process every Nth frame (default: 1 = all frames)
- `--roi`: Region of Interest as "x1,y1,x2,y2" (optional)
- `--start-datetime`: Start datetime in ISO format (optional)
- `--no-video`: Skip saving annotated video for faster processing
- `--skip-auth-check`: Skip Hugging Face authentication check

## Output Files

The pipeline generates the following outputs in the specified output directory:

1. **traffic_timeseries.csv** - Time-series dataset for LSTM training
2. **detections.json** - Frame-by-frame vehicle detections
3. **density_data.json** - Weighted density calculations
4. **annotated_video.mp4** - Video with bounding boxes and labels
5. **density_timeline.png** - Temporal density visualization
6. **vehicle_distribution.png** - Vehicle type breakdown
7. **vehicle_heatmap.png** - Heatmap of vehicle types over time
8. **summary_dashboard.png** - Comprehensive analysis dashboard
9. **analysis_summary.json** - Complete analysis statistics

## Time-Series Dataset Schema

The generated CSV contains the following columns:

- `bin_id`: Time bin identifier
- `datetime`: Timestamp
- `timestamp_seconds`: Seconds from start
- `total_vehicle_count`: Total vehicles in bin
- `avg_vehicle_count`: Average vehicles per frame
- `max_vehicle_count`: Peak vehicles in bin
- `Hatchback_count`, `Sedan_count`, etc.: Count per vehicle type
- `avg_weighted_density`: Average weighted density (%)
- `max_weighted_density`: Peak weighted density (%)
- `min_weighted_density`: Minimum weighted density (%)
- `avg_occupancy_percentage`: Average road occupancy (%)
- `max_occupancy_percentage`: Peak road occupancy (%)
- `frames_in_bin`: Number of frames in time bin

## Weighted Density Calculation

The system uses vehicle-specific weights to calculate realistic road occupancy:

**Formula**: `Weighted_Density = (Σ(Area_i × Weight_i) / Area_ROI) × 100`

**Vehicle Weights** (relative to Hatchback = 1.0):
- Bicycle: 0.2
- Two-Wheeler: 0.4
- Three-Wheeler: 0.6
- Hatchback: 1.0
- Sedan: 1.1
- SUV: 1.4
- MUV: 1.5
- Van: 1.6
- LCV: 1.8
- Tempo-traveller: 2.0
- Mini-bus: 2.2
- Truck: 2.8
- Bus: 3.0

## Project Structure

```
traffic-analysis/
├── main.py                     # Main pipeline script
├── requirements.txt            # Dependencies
├── README.md                   # This file
├── config/
│   └── vehicle_weights.json   # Vehicle weight configuration
├── src/
│   ├── vehicle_classifier.py  # VehicleNet-Y26x classifier
│   ├── density_calculator.py  # Weighted density computation
│   ├── time_series_generator.py  # Time-series aggregation
│   └── visualizer.py          # Visualization module
└── data/
    ├── input/                 # Input videos
    └── output/                # Generated outputs
```

## Example Workflow

```bash
# 1. Place your video in data/input/
cp /path/to/traffic_video.mp4 data/input/

# 2. Run the pipeline
python main.py --video data/input/traffic_video.mp4 --bin-size 5

# 3. Check outputs
ls data/output/

# 4. Use the time-series CSV for LSTM training
# The traffic_timeseries.csv is ready for your forecasting model!
```

## Next Steps: LSTM Forecasting

The generated `traffic_timeseries.csv` is ready for training your LSTM model with:
- Historical vehicle counts and density metrics
- Event flags for festivals/anomalies
- Multi-variate features for robust forecasting

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended for faster processing)
- Hugging Face account (free)

## Troubleshooting

**Model Loading Error:**
```bash
# Ensure you're authenticated
huggingface-cli login

# Or set token as environment variable
export HF_TOKEN=your_token_here
```

**Out of Memory:**
```bash
# Process fewer frames
python main.py --video input.mp4 --frame-skip 2  # Process every 2nd frame
```

**Slow Processing:**
```bash
# Skip video annotation
python main.py --video input.mp4 --no-video
```

## License

This project uses the VehicleNet-Y26x model from Hugging Face (Perception365/VehicleNet-Y26x).

## Citation

If you use this pipeline in your research, please cite the VehicleNet-Y26x model and Ultralytics YOLO.
