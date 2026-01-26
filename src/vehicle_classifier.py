"""
Vehicle Classifier Module
Uses VehicleNet-Y26x from Hugging Face for vehicle detection and classification
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from tqdm import tqdm


class VehicleClassifier:
    """
    Vehicle classification pipeline using VehicleNet-Y26x YOLO model
    """
    
    def __init__(self, model_name: str = "Perception365/VehicleNet-Y26s", 
                 confidence_threshold: float = 0.25,
                 device: str = 'auto'):
        """
        Initialize the vehicle classifier
        
        Args:
            model_name: Hugging Face model identifier
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self.frame_detections = []
        
    def load_model(self):
        """Load the VehicleNet-Y26s model from Hugging Face"""
        try:
            from ultralytics import YOLO
            # Keep manual download to avoid Windows path issues with hf:// strings
            from huggingface_hub import hf_hub_download, list_repo_files
            
            print(f"Loading model: {self.model_name}")
            print(f"Searching for weights in Hugging Face repo: {self.model_name}...")
            
            try:
                # 1. Try 'weights/best.pt' (standard structure for this repo)
                print(f"Downloading weights/best.pt...")
                model_path = hf_hub_download(
                    repo_id=self.model_name,
                    filename="weights/best.pt"
                )
            except Exception:
                # 2. Search for any .pt file if specific path fails
                print("Specific 'weights/best.pt' not found, searching repo...")
                files = list_repo_files(repo_id=self.model_name)
                pt_files = [f for f in files if f.endswith('.pt')]
                
                if not pt_files:
                    raise FileNotFoundError("No .pt weights found in repository")
                
                # Pick the best candidate
                if 'best.pt' in pt_files:
                    weight_file = 'best.pt'
                elif 'model.pt' in pt_files:
                    weight_file = 'model.pt'
                else:
                    weight_file = pt_files[0] 
                
                print(f"Found alternative weight file: {weight_file}")
                model_path = hf_hub_download(
                    repo_id=self.model_name,
                    filename=weight_file
                )

            print(f"✓ Downloaded to: {model_path}")
            
            # Load with YOLO
            self.model = YOLO(model_path)
            print(f"✓ Model loaded successfully")
            return True
                
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            print("\nTroubleshooting:")
            print("1. IMPORTANT: This model is RESTRICTED. You must visit:")
            print(f"   https://huggingface.co/{self.model_name}")
            print("   and accept the access terms.")
            print("2. Ensure you're authenticated in the terminal:")
            print("   huggingface-cli login")
            print("3. Check internet connection")
            return False
    
    def process_video(self, 
                     video_path: str, 
                     output_dir: str,
                     roi: Optional[Tuple[int, int, int, int]] = None,
                     save_annotated: bool = True,
                     frame_skip: int = 1) -> List[Dict]:
        """
        Process video and extract vehicle detections
        
        Args:
            video_path: Path to input video file
            output_dir: Directory to save outputs
            roi: Region of Interest as (x1, y1, x2, y2). None = full frame
            save_annotated: Whether to save annotated video
            frame_skip: Process every Nth frame (1 = all frames)
            
        Returns:
            List of detection dictionaries per frame
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nVideo Info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.2f} seconds")
        
        # Setup video writer if saving annotated output
        writer = None
        if save_annotated:
            output_video = output_path / "annotated_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        
        # Calculate ROI area
        if roi is None:
            roi = (0, 0, width, height)
        roi_area = (roi[2] - roi[0]) * (roi[3] - roi[1])
        
        # Process frames
        self.frame_detections = []
        frame_idx = 0
        processed_count = 0
        
        pbar = tqdm(total=total_frames, desc="Processing video")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames if needed
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                pbar.update(1)
                continue
            
            # Calculate timestamp
            timestamp = frame_idx / fps
            
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            # Extract detections
            frame_data = {
                'frame_idx': frame_idx,
                'timestamp': timestamp,
                'detections': [],
                'roi_area': roi_area
            }
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Check if detection is within ROI
                    if not self._is_in_roi(x1, y1, x2, y2, roi):
                        continue
                    
                    # Get class and confidence
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = result.names[cls_id]
                    
                    # Calculate bounding box area
                    bbox_area = (x2 - x1) * (y2 - y1)
                    
                    detection = {
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'bbox_area': float(bbox_area)
                    }
                    
                    frame_data['detections'].append(detection)
                    
                    # Draw on frame if saving
                    if save_annotated:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                    (0, 255, 0), 2)
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1)-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw ROI rectangle
            if save_annotated and roi != (0, 0, width, height):
                cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), 
                            (255, 0, 0), 2)
            
            # Add frame info
            if save_annotated:
                info_text = f"Frame: {frame_idx} | Time: {timestamp:.2f}s | Vehicles: {len(frame_data['detections'])}"
                cv2.putText(frame, info_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                writer.write(frame)
            
            self.frame_detections.append(frame_data)
            
            frame_idx += 1
            processed_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        if writer:
            writer.release()
        
        print(f"\n✓ Processed {processed_count} frames")
        print(f"✓ Total detections: {sum(len(f['detections']) for f in self.frame_detections)}")
        
        # Save detections to JSON
        detections_file = output_path / "detections.json"
        with open(detections_file, 'w') as f:
            json.dump(self.frame_detections, f, indent=2)
        print(f"✓ Detections saved to: {detections_file}")
        
        if save_annotated:
            print(f"✓ Annotated video saved to: {output_video}")
        
        return self.frame_detections
    
    def _is_in_roi(self, x1: float, y1: float, x2: float, y2: float, 
                   roi: Tuple[int, int, int, int]) -> bool:
        """Check if bounding box center is within ROI"""
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return (roi[0] <= center_x <= roi[2] and roi[1] <= center_y <= roi[3])
    
    def get_detection_summary(self) -> Dict:
        """Get summary statistics of detections"""
        if not self.frame_detections:
            return {}
        
        total_detections = sum(len(f['detections']) for f in self.frame_detections)
        
        # Count by class
        class_counts = {}
        for frame in self.frame_detections:
            for det in frame['detections']:
                cls = det['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
        
        summary = {
            'total_frames': len(self.frame_detections),
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / len(self.frame_detections),
            'class_distribution': class_counts
        }
        
        return summary
