"""
Traffic Analysis Pipeline - Main Entry Point
Processes traffic video using VehicleNet-Y26s and generates time-series dataset
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from vehicle_classifier import VehicleClassifier
from density_calculator import DensityCalculator
from time_series_generator import TimeSeriesGenerator
from visualizer import TrafficVisualizer


def check_hf_auth():
    """Check if Hugging Face authentication is configured"""
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("✓ Hugging Face authentication found")
            return True
        else:
            print("⚠ Hugging Face authentication not found")
            print("\nTo authenticate, run:")
            print("  huggingface-cli login")
            print("\nOr set HF_TOKEN environment variable")
            return False
    except Exception as e:
        print(f"⚠ Could not check HF authentication: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Traffic Analysis Pipeline using VehicleNet-Y26s'
    )
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', type=str, default='data/output',
                       help='Output directory (default: data/output)')
    parser.add_argument('--bin-size', type=float, default=5,
                       help='Time bin size in minutes (default: 5)')
    parser.add_argument('--bin-size-seconds', type=int, default=None,
                       help='Time bin size in seconds (overrides --bin-size if set)')
    parser.add_argument('--confidence', type=float, default=0.25,
                       help='Detection confidence threshold (default: 0.25)')
    parser.add_argument('--frame-skip', type=int, default=1,
                       help='Process every Nth frame (default: 1 = all frames)')
    parser.add_argument('--roi', type=str, default=None,
                       help='Region of Interest as "x1,y1,x2,y2" (default: full frame)')
    parser.add_argument('--start-datetime', type=str, default=None,
                       help='Start datetime in ISO format (default: epoch)')
    parser.add_argument('--no-video', action='store_true',
                       help='Skip saving annotated video (faster processing)')
    parser.add_argument('--skip-auth-check', action='store_true',
                       help='Skip Hugging Face authentication check')
    
    args = parser.parse_args()
    
    # Validate video file
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"✗ Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Parse ROI if provided
    roi = None
    if args.roi:
        try:
            roi = tuple(map(int, args.roi.split(',')))
            if len(roi) != 4:
                raise ValueError
            print(f"✓ Using ROI: {roi}")
        except:
            print("✗ Error: ROI must be in format 'x1,y1,x2,y2'")
            sys.exit(1)
    
    # Setup paths
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_dir = Path(__file__).parent / 'config'
    weights_config = config_dir / 'vehicle_weights.json'
    
    print("=" * 70)
    print("TRAFFIC ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"\nVideo: {video_path}")
    print(f"Output: {output_dir}")
    # Determine bin size
    bin_size = args.bin_size
    if args.bin_size_seconds is not None:
        bin_size = args.bin_size_seconds / 60.0
        print(f"Bin Size: {args.bin_size_seconds} seconds")
    else:
        print(f"Bin Size: {args.bin_size} minutes")
        
    print(f"Confidence: {args.confidence}")
    print(f"Frame Skip: {args.frame_skip}")
    print()
    
    # Check HF authentication
    if not args.skip_auth_check:
        check_hf_auth()
        print()
    
    # ========================================================================
    # STEP 1: Vehicle Classification
    # ========================================================================
    print("=" * 70)
    print("STEP 1: VEHICLE CLASSIFICATION")
    print("=" * 70)
    
    classifier = VehicleClassifier(
        confidence_threshold=args.confidence,
        device='auto'
    )
    
    if not classifier.load_model():
        print("\n✗ Failed to load model. Exiting.")
        sys.exit(1)
    
    print("\nProcessing video...")
    detections = classifier.process_video(
        video_path=str(video_path),
        output_dir=str(output_dir),
        roi=roi,
        save_annotated=not args.no_video,
        frame_skip=args.frame_skip
    )
    
    # Print detection summary
    summary = classifier.get_detection_summary()
    print("\nDetection Summary:")
    print(f"  Total Frames: {summary['total_frames']}")
    print(f"  Total Detections: {summary['total_detections']}")
    print(f"  Avg Detections/Frame: {summary['avg_detections_per_frame']:.2f}")
    print("\n  Vehicle Distribution:")
    for vtype, count in sorted(summary['class_distribution'].items(), 
                               key=lambda x: x[1], reverse=True):
        print(f"    {vtype}: {count}")
    
    # ========================================================================
    # STEP 2: Density Calculation
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: WEIGHTED DENSITY CALCULATION")
    print("=" * 70)
    
    calculator = DensityCalculator(weights_config_path=str(weights_config))
    
    density_data = calculator.calculate_all_frames(detections)
    
    # Save density data
    density_file = output_dir / 'density_data.json'
    calculator.save_density_data(density_data, str(density_file))
    
    # Print density statistics
    stats = calculator.get_density_statistics(density_data)
    print("\nDensity Statistics:")
    print(f"  Avg Weighted Density: {stats['avg_weighted_density']:.2f}%")
    print(f"  Max Weighted Density: {stats['max_weighted_density']:.2f}%")
    print(f"  Avg Vehicle Count: {stats['avg_vehicle_count']:.2f}")
    print(f"  Max Vehicle Count: {stats['max_vehicle_count']}")
    
    # ========================================================================
    # STEP 3: Time-Series Generation
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: TIME-SERIES DATASET GENERATION")
    print("=" * 70)
    
    ts_generator = TimeSeriesGenerator(bin_size_minutes=bin_size)
    
    ts_df = ts_generator.aggregate_to_timeseries(
        density_data=density_data,
        start_datetime=args.start_datetime
    )
    
    # Save time-series dataset
    ts_file = output_dir / 'traffic_timeseries.csv'
    ts_generator.save_timeseries(ts_df, str(ts_file))
    
    # Print time-series summary
    ts_summary = ts_generator.get_timeseries_summary(ts_df)
    print("\nTime-Series Summary:")
    print(f"  Total Bins: {ts_summary['total_bins']}")
    print(f"  Duration: {ts_summary['total_duration_minutes']:.1f} minutes")
    print(f"  Avg Vehicles/Bin: {ts_summary['avg_vehicles_per_bin']:.1f}")
    print(f"  Max Vehicles in Bin: {ts_summary['max_vehicles_in_bin']}")
    print(f"  Avg Density: {ts_summary['avg_density']:.2f}%")
    print(f"  Peak Occupancy: {ts_summary['peak_occupancy']:.2f}%")
    
    # ========================================================================
    # STEP 4: Visualization
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    visualizer = TrafficVisualizer()
    
    # Create plots
    visualizer.plot_density_timeline(ts_df, str(output_dir / 'density_timeline.png'))
    visualizer.plot_vehicle_distribution(ts_df, str(output_dir / 'vehicle_distribution.png'))
    visualizer.plot_heatmap(ts_df, str(output_dir / 'vehicle_heatmap.png'))
    visualizer.create_summary_dashboard(ts_df, ts_summary, 
                                       str(output_dir / 'summary_dashboard.png'))
    
    # ========================================================================
    # STEP 5: Save Final Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: SAVING SUMMARY")
    print("=" * 70)
    
    final_summary = {
        'video_file': str(video_path),
        'processing_datetime': datetime.now().isoformat(),
        'parameters': {
            'bin_size_minutes': bin_size,
            'bin_size_seconds': args.bin_size_seconds if args.bin_size_seconds else int(bin_size * 60),
            'confidence_threshold': args.confidence,
            'frame_skip': args.frame_skip,
            'roi': roi
        },
        'detection_summary': summary,
        'density_statistics': stats,
        'timeseries_summary': ts_summary
    }
    
    summary_file = output_dir / 'analysis_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"✓ Analysis summary saved to: {summary_file}")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nOutput Files:")
    print(f"  1. Time-Series Dataset: {ts_file}")
    print(f"  2. Detections (JSON): {output_dir / 'detections.json'}")
    print(f"  3. Density Data (JSON): {density_file}")
    if not args.no_video:
        print(f"  4. Annotated Video: {output_dir / 'annotated_video.mp4'}")
    print(f"  5. Density Timeline: {output_dir / 'density_timeline.png'}")
    print(f"  6. Vehicle Distribution: {output_dir / 'vehicle_distribution.png'}")
    print(f"  7. Vehicle Heatmap: {output_dir / 'vehicle_heatmap.png'}")
    print(f"  8. Summary Dashboard: {output_dir / 'summary_dashboard.png'}")
    print(f"  9. Analysis Summary: {summary_file}")
    print("\n✓ Ready for LSTM training!")
    print("=" * 70)


if __name__ == '__main__':
    main()
