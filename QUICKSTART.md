# Quick Start Guide

## Prerequisites

1. **Python 3.10+** installed
2. **Hugging Face account** (free) - [Sign up here](https://huggingface.co/join)
3. **Video file** of traffic footage

## Installation Steps

### 1. Navigate to Project Directory

```bash
cd traffic-analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Authenticate with Hugging Face

**Option A: Interactive Login (Recommended)**
```bash
huggingface-cli login
```
Then paste your token when prompted.

**Get your token:** https://huggingface.co/settings/tokens

**Option B: Environment Variable**
```bash
# Windows (PowerShell)
$env:HF_TOKEN="your_token_here"

# Windows (CMD)
set HF_TOKEN=your_token_here
```

### 4. Prepare Your Video

Place your traffic video in the `data/input/` directory:

```bash
# Copy your video (example)
copy C:\path\to\your\video.mp4 data\input\traffic.mp4
```

### 5. Run the Pipeline

**Basic usage:**
```bash
python main.py --video data/input/traffic.mp4
```

**With custom settings:**
```bash
python main.py --video data/input/traffic.mp4 --bin-size 5 --confidence 0.3
```

### 6. Check Results

All outputs will be in `data/output/`:
- `traffic_timeseries.csv` - Your time-series dataset ✨
- `annotated_video.mp4` - Video with detections
- `density_timeline.png` - Traffic patterns
- `vehicle_distribution.png` - Vehicle types
- `summary_dashboard.png` - Complete overview

## Common Issues & Solutions

### "Model loading failed"
Make sure you're authenticated:
```bash
huggingface-cli login
```

### "Out of memory"
Process fewer frames:
```bash
python main.py --video data/input/traffic.mp4 --frame-skip 3 --no-video
```

### "Slow processing"
Skip frames or disable video output:
```bash
python main.py --video data/input/traffic.mp4 --frame-skip 2 --no-video
```

## Quick Reference

### Key Arguments

- `--video` - Input video path (required)
- `--bin-size` - Time bin in minutes (default: 5)
- `--confidence` - Detection threshold (default: 0.25)
- `--frame-skip` - Process every Nth frame (default: 1)
- `--roi` - Region of interest "x1,y1,x2,y2"
- `--no-video` - Skip annotated video (faster)

### Example Commands

```bash
# Fastest processing
python main.py --video input.mp4 --frame-skip 5 --no-video

# Most accurate
python main.py --video input.mp4 --confidence 0.5

# Custom time bins
python main.py --video input.mp4 --bin-size 10

# Focus on specific area
python main.py --video input.mp4 --roi "200,300,1600,900"
```

## Next Steps

1. Review outputs in `data/output/`
2. Use `traffic_timeseries.csv` for LSTM training
3. Check `README.md` for detailed documentation
