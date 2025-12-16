from typing import Dict, Optional

from .video_processor import VideoProcessor


class LiveProcessor(VideoProcessor):
    """
    Thin wrapper around VideoProcessor providing explicit live camera control/status.
    """

    def __init__(self, roi_config: Dict, *, display_windows: bool = True) -> None:
        super().__init__(roi_config, display_windows=display_windows)
        self.is_processing = False

    def start_live_processing(
        self,
        camera_index: int = 0,
        duration: Optional[float] = None,
        frame_callback=None,
    ) -> Dict:
        self.is_processing = True
        try:
            return super().process_live(camera_index, duration, frame_callback=frame_callback)
        finally:
            self.is_processing = False

    def stop_processing(self) -> None:
        self.is_processing = False

    def get_processing_status(self) -> Dict:
        return {
            "is_processing": self.is_processing,
            "frames_processed": self.frame_count,
            "current_stats": self.analytics.get_current_stats(),
        }

