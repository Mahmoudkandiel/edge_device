"""
SOPHON-optimized person detection using BModel.
This module provides YOLOv8 person detection using Sophon TPU acceleration.
"""

from typing import Dict, List, Optional
import os
import numpy as np
import cv2

try:
    import sophon.sail as sail
    SOPHON_AVAILABLE = True
except ImportError:
    SOPHON_AVAILABLE = False
    sail = None


class PersonDetectorSOPHON:
    """
    Person detection using SOPHON BModel for YOLOv8.
    Optimized for BM1684X and other Sophon TPU chips.
    """

    def __init__(self, detection_model_path: Optional[str] = None) -> None:
        if not SOPHON_AVAILABLE:
            raise ImportError(
                "Sophon SDK (sophon.sail) is not available. "
                "Please install Sophon SDK or use the standard PersonDetector."
            )
        
        # Default model path
        if detection_model_path is None:
            # Try common locations
            possible_paths = [
                "yolov8n_1688_f16.bmodel"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    detection_model_path = path
                    break
            
            if detection_model_path is None:
                raise FileNotFoundError(
                    f"BModel not found. Please provide a valid path or place model in one of: {possible_paths}"
                )
        
        if not os.path.exists(detection_model_path):
            raise FileNotFoundError(f"BModel file not found: {detection_model_path}")
        
        self.device_id = 0
        self.model_path = detection_model_path
        
        # Initialize Sophon Engine
        try:
            self.engine = sail.Engine(self.model_path, self.device_id, sail.IOMode.SYSIO)
            self.graph_name = self.engine.get_graph_names()[0]
            
            # Get input/output information
            self.input_names = self.engine.get_input_names(self.graph_name)
            self.output_names = self.engine.get_output_names(self.graph_name)
            
            if not self.input_names:
                raise ValueError("No input found in BModel")
            
            self.input_name = self.input_names[0]
            
            # Get input shape
            input_shape = self.engine.get_input_shape(self.graph_name, self.input_name)
            if len(input_shape) == 4:  # NCHW format
                self.batch_size, self.channels, self.input_h, self.input_w = input_shape
            else:
                raise ValueError(f"Unexpected input shape: {input_shape}")
            
            # YOLOv8 configuration
            self.conf_threshold = 0.5
            self.iou_threshold = 0.45
            self.max_detections = 100
            
            print(f"PersonDetectorSOPHON initialized successfully!")
            print(f"  Model: {self.model_path}")
            print(f"  Input shape: {input_shape}")
            print(f"  Input name: {self.input_name}")
            print(f"  Output names: {self.output_names}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Sophon Engine: {e}")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for YOLOv8 input.
        Converts BGR to RGB, resizes, normalizes, and converts to NCHW format.
        """
        # Resize to model input size
        resized = cv2.resize(frame, (self.input_w, self.input_h))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to float32
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format (HWC -> CHW)
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension (CHW -> NCHW)
        batched = np.expand_dims(transposed, axis=0)
        
        return batched

    def postprocess(self, outputs: List[np.ndarray], frame_shape: tuple) -> List[Dict]:
        """
        Postprocess YOLOv8 outputs to get person detections.
        YOLOv8 output format: [batch, num_detections, 4+num_classes] or [batch, num_detections, 85]
        For COCO: 80 classes + 4 bbox coords + 1 objectness = 85
        """
        detections = []
        
        if len(outputs) == 0:
            return detections
        
        # YOLOv8 typically outputs a single tensor
        # Shape: [1, 8400, 84] for YOLOv8n (no objectness) or [1, 8400, 85] (with objectness)
        output = outputs[0]
        
        # Handle different output formats
        if len(output.shape) == 3:
            batch_size, num_detections, num_features = output.shape
            
            # YOLOv8 format: [x_center, y_center, width, height, class_scores...]
            # or: [objectness, x_center, y_center, width, height, class_scores...]
            
            # Check if first dimension is objectness or bbox
            if num_features == 85:  # With objectness
                boxes = output[0, :, 1:5]  # [8400, 4] - x, y, w, h (normalized)
                # Get class scores (person is class 0 in COCO)
                class_scores = output[0, :, 5:]  # [8400, 80]
                person_scores = class_scores[:, 0]  # [8400] - person class scores
            elif num_features == 84:  # Without objectness
                boxes = output[0, :, :4]  # [8400, 4] - x, y, w, h (normalized)
                class_scores = output[0, :, 4:]  # [8400, 80]
                person_scores = class_scores[:, 0]  # [8400] - person class scores
            else:
                # Try to infer format
                boxes = output[0, :, :4]
                if num_features > 4:
                    class_scores = output[0, :, 4:]
                    if class_scores.shape[1] >= 1:
                        person_scores = class_scores[:, 0]
                    else:
                        person_scores = np.ones(num_detections) * 0.5
                else:
                    person_scores = np.ones(num_detections) * 0.5
        else:
            # Unexpected format, try to handle gracefully
            print(f"Warning: Unexpected output shape: {output.shape}")
            return detections
        
        # Filter by confidence threshold
        valid_indices = np.where(person_scores > self.conf_threshold)[0]
        
        # Apply NMS (Non-Maximum Suppression)
        if len(valid_indices) > 0:
            valid_boxes = boxes[valid_indices]
            valid_scores = person_scores[valid_indices]
            
            # Convert to xyxy format for NMS
            x_center = valid_boxes[:, 0]
            y_center = valid_boxes[:, 1]
            width = valid_boxes[:, 2]
            height = valid_boxes[:, 3]
            
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Apply NMS using OpenCV
            boxes_xyxy = np.column_stack([x1, y1, x2, y2])
            indices = cv2.dnn.NMSBoxes(
                boxes_xyxy.tolist(),
                valid_scores.tolist(),
                self.conf_threshold,
                self.iou_threshold
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                valid_indices = valid_indices[indices]
        
        frame_h, frame_w = frame_shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        for idx in valid_indices[:self.max_detections]:
            # Get box coordinates (normalized center x, y, width, height)
            cx, cy, w, h = boxes[idx]
            confidence = person_scores[idx]
            
            # Convert to pixel coordinates
            x1 = int((cx - w/2) * frame_w)
            y1 = int((cy - h/2) * frame_h)
            x2 = int((cx + w/2) * frame_w)
            y2 = int((cy + h/2) * frame_h)
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, min(x1, frame_w))
            y1 = max(0, min(y1, frame_h))
            x2 = max(0, min(x2, frame_w))
            y2 = max(0, min(y2, frame_h))
            
            # Calculate centroid
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "centroid": centroid,
                "confidence": float(confidence),
            })
        
        return detections

    def analyze_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect people in frame using SOPHON BModel.
        
        Args:
            frame: Input frame in BGR format (numpy array)
            
        Returns:
            List of detection dictionaries with keys: bbox, centroid, confidence
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            # Preprocess
            input_data = self.preprocess(frame)
            
            # Run inference
            input_dict = {self.input_name: input_data}
            output_dict = self.engine.process(self.graph_name, input_dict)
            
            # Get outputs
            outputs = [output_dict[name] for name in self.output_names]
            
            # Postprocess
            detections = self.postprocess(outputs, frame.shape)
            
            return detections
            
        except Exception as e:
            print(f"Error during Sophon inference: {e}")
            return []

