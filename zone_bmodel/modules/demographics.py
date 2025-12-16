"""
Person detection module with automatic Sophon/PyTorch detection.
Automatically uses Sophon BModel if available, otherwise falls back to PyTorch/YOLO.
"""

import os
from typing import Dict, List, Optional

import numpy as np

# Try to import Sophon first
try:
    import sophon.sail as sail
    SOPHON_AVAILABLE = True
except ImportError:
    SOPHON_AVAILABLE = False
    sail = None

# Try to import PyTorch/YOLO as fallback
try:
    import torch
    from ultralytics import YOLO
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    YOLO = None


def _detect_sophon_model(model_path: Optional[str] = None) -> Optional[str]:
    """
    Check if a Sophon BModel is available.
    Returns the path to the BModel if found, None otherwise.
    """
    if not SOPHON_AVAILABLE:
        return None
    
    # If explicit path provided, check if it's a BModel
    if model_path and os.path.exists(model_path):
        if model_path.endswith('.bmodel'):
            return model_path
    
    # Check common locations for BModel
    possible_paths = [
        "models/yolov8n_bm1684x.bmodel",
        "models/yolov8n.bmodel",
        "models/yolov8_bm1684x.bmodel",
        "models/yolov8.bmodel",
        os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n_bm1684x.bmodel"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


class PersonDetector:
    """
    Performs person detection for tracking purposes.
    Automatically uses Sophon BModel if available, otherwise uses PyTorch/YOLO.
    """

    def __init__(self, detection_model_path: Optional[str] = None) -> None:
        self.detection_model_path = detection_model_path
        self.detector_type = None
        self.detector = None
        
        # Check for Sophon BModel first
        sophon_model = _detect_sophon_model(detection_model_path)
        if sophon_model:
            try:
                from .demographics_sophon import PersonDetectorSOPHON
                self.detector = PersonDetectorSOPHON(sophon_model)
                self.detector_type = "sophon"
                print(f"✅ Using Sophon BModel: {sophon_model}")
                return
            except Exception as exc:
                print(f"⚠️  Failed to load Sophon detector: {exc}")
                print("   Falling back to PyTorch/YOLO...")
        
        # Fall back to PyTorch/YOLO
        if not PYTORCH_AVAILABLE:
            raise ImportError(
                "Neither Sophon SDK nor PyTorch/Ultralytics is available. "
                "Please install one of them:\n"
                "  - For Sophon: Install Sophon SDK (sophon.sail)\n"
                "  - For PyTorch: pip install torch ultralytics"
            )
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"PersonDetector using PyTorch/YOLO on device: {self.device}")
        try:
            self.detector = YOLO(detection_model_path or "yolov8n.pt")
            self.detector_type = "pytorch"
        except Exception as exc:
            print(f"Error loading YOLO model: {exc}")
            self.detector = None

    def analyze_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Analyze frame and return person detections.
        Works with both Sophon and PyTorch backends.
        """
        if self.detector is None:
            return []
        
        if self.detector_type == "sophon":
            # Sophon detector
            return self.detector.analyze_frame(frame)
        else:
            # PyTorch/YOLO detector
            # Configure YOLO to reduce NMS time and improve performance
            results = self.detector(
                frame,
                verbose=False,
                conf=0.5,  # Confidence threshold
                iou=0.45,  # IoU threshold for NMS (lower = faster, less strict)
                max_det=100,  # Maximum detections per image (reduce if too many people)
                agnostic_nms=False,  # Class-agnostic NMS
            )
            detections: List[Dict] = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    if int(box.cls) != 0 or box.conf < 0.5:
                        continue
                    bbox = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = bbox
                    centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    detections.append(
                        {
                            "bbox": (x1, y1, x2, y2),
                            "centroid": centroid,
                            "confidence": float(box.conf.cpu().numpy()[0]),
                        }
                    )
            return detections

