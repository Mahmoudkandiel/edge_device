import os
import time
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from config.settings import Config
from .analytics import Analytics
from .classifier import MovementClassifier
from .demographics import PersonDetector
from .tracker import PersonTracker


class VideoProcessor:
    """
    Coordinates detection, tracking, classification, and reporting
    for offline video files.
    """

    def __init__(self, roi_config: Dict, *, display_windows: bool = True) -> None:
        self.roi_config = roi_config
        self.frame_count = 0
        self.processing_times: List[float] = []
        self.tracker = PersonTracker()
        self.classifier = MovementClassifier(roi_config)
        self.person_detector = PersonDetector()
        self.analytics = Analytics()
        self.output_video: Optional[cv2.VideoWriter] = None
        self.output_path: Optional[str] = None
        self.display_windows = display_windows

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if Config.ANALYSIS.SAVE_VIDEO_OUTPUT:
                if output_path is None:
                    output_path = os.path.join(Config.VIDEO_OUTPUT_DIR, f"output_{os.path.basename(video_path)}")
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                self.output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                self.output_path = output_path

            start = time.time()
            frame_skip = Config.VIDEO.FRAME_SKIP

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if self.frame_count % frame_skip != 0:
                    self.frame_count += 1
                    continue

                timestamp = time.time()
                processed_frame = self._process_frame(frame, self.frame_count, timestamp)
                if self.output_video is not None:
                    self.output_video.write(processed_frame)
                if self.display_windows and Config.ANALYSIS.SAVE_VIDEO_OUTPUT:
                    cv2.imshow("Video Processing", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                self.frame_count += 1
                if progress_callback is not None:
                    try:
                        progress_callback(self.frame_count, total_frames)
                    except Exception:
                        # Handle StopException from Streamlit gracefully
                        break

            processing_time = time.time() - start
            report = self._final_report(processing_time, total_frames)
            if Config.ANALYSIS.GENERATE_REPORTS:
                self.analytics.save_report(report)
            return report
        finally:
            # Always release resources, even if interrupted
            if cap is not None:
                cap.release()
            if self.output_video is not None:
                self.output_video.release()
            if self.display_windows:
                cv2.destroyAllWindows()

    def _process_frame(self, frame: np.ndarray, frame_count: int, timestamp: float) -> np.ndarray:
        start = time.time()
        detections = self.person_detector.analyze_frame(frame)
        self.tracker.update(detections, frame_count, timestamp)
        for person_id in list(self.tracker.tracks.keys()):
            classification = self.classifier.classify_person_movement(self.tracker, person_id)
            track_data = self.tracker.get_track_data(person_id)
            track_data["classification"] = classification
            self.analytics.update_person_data(person_id, track_data)

        composed = frame.copy()
        composed = self.classifier.draw_classification_zones(composed)
        composed = self.tracker.draw_tracks(composed)
        composed = self._draw_overlay(composed)
        self.processing_times.append(time.time() - start)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        return composed

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 225), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        stats = self.analytics.get_current_stats()
        lines = [
            f"IN: {stats.get('in_count', 0)}",
            f"OUT: {stats.get('out_count', 0)}",
            f"Total: {stats.get('in_count', 0) + stats.get('out_count', 0)}",
            f"Bypassers: {stats['bypassers']}",
            f"Current Inside: {stats['current_inside']}",
            f"Frame: {self.frame_count}",
            f"Proc FPS: {self._processing_fps():.1f}",
        ]
        for idx, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (20, 40 + idx * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
        return frame

    def _processing_fps(self) -> float:
        if not self.processing_times:
            return 0.0
        avg = sum(self.processing_times) / len(self.processing_times)
        return 1.0 / avg if avg else 0.0

    def _final_report(self, processing_time: float, total_frames: int) -> Dict:
        final_stats = self.analytics.get_final_stats()
        report = {
            "processing_info": {
                "video_frames": total_frames,
                "processed_frames": self.frame_count,
                "processing_time_seconds": processing_time,
                "average_fps": self.frame_count / processing_time if processing_time else 0,
                "output_video_path": self.output_path,
            },
            "analytics": final_stats,
            "tracking_summary": {
                "total_tracks": len(self.tracker.get_all_tracks_data()),
                "completed_tracks": final_stats.get("total_people", 0),
            },
        }
        return report

    def process_live(
        self,
        camera_index: int = 0,
        duration: Optional[float] = None,
        frame_callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> Dict:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_index}")
        if self.display_windows:
            print("Starting live processing. Press 'q' to stop.")
        start = time.time()
        self.frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = time.time()
            processed = self._process_frame(frame, self.frame_count, timestamp)
            if frame_callback is not None:
                frame_callback(processed)
            elif self.display_windows:
                cv2.imshow("Live Processing", processed)
            self.frame_count += 1
            if self.display_windows and cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if duration and (time.time() - start) > duration:
                break
        cap.release()
        if self.display_windows:
            cv2.destroyAllWindows()
        processing_time = time.time() - start
        return self._final_report(processing_time, self.frame_count)

