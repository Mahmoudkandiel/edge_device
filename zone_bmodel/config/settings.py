import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class VideoConfig:
    FRAME_SKIP: int = 3
    CONFIDENCE_THRESHOLD: float = 0.6
    TRACKING_THRESHOLD: float = 0.5
    ROI_COLORS: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.ROI_COLORS:
            self.ROI_COLORS = {
                "entry": (0, 255, 0),
                "exit": (0, 0, 255),
                "bypass": (255, 255, 0),
            }


@dataclass
class AnalysisConfig:
    SAVE_VIDEO_OUTPUT: bool = True
    GENERATE_REPORTS: bool = True
    OUTPUT_FORMAT: str = "both"


@dataclass
class SophonConfig:
    """Configuration for Sophon TPU deployment."""
    # Model paths (will auto-detect if None)
    BMODEL_PATH: Optional[str] = None
    
    # Common model locations to check
    MODEL_SEARCH_PATHS: List[str] = field(default_factory=lambda: [
        "models/yolov8n_bm1684x.bmodel",
        "models/yolov8n.bmodel",
        "models/yolov8_bm1684x.bmodel",
        "models/yolov8.bmodel",
    ])
    
    # Device ID for multi-TPU systems
    DEVICE_ID: int = 0
    
    # Performance settings
    USE_FP16: bool = True  # Prefer FP16 over INT8 for better accuracy
    FUSE_PREPROCESS: bool = False  # Enable if model was compiled with --fuse_preprocess


class Config:
    VIDEO = VideoConfig()
    ANALYSIS = AnalysisConfig()
    SOPHON = SophonConfig()

    OUTPUT_DIR = "outputs"
    VIDEO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "videos")
    REPORT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "reports")
    MODELS_DIR = "models"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

