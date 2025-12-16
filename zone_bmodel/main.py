#!/usr/bin/env python3
"""
Enhanced Customer Foot Traffic Analyzer
"""

import os
import sys
from typing import Dict, Optional

from config.settings import Config
from modules.live_processor import LiveProcessor
from modules.roi_selector import ROISelector
from modules.video_processor import VideoProcessor


class TrafficAnalyzer:
    def __init__(self) -> None:
        self.roi_selector = ROISelector()
        self.roi_config: Optional[Dict] = None
        self.current_processor: Optional[VideoProcessor] = None
        print("🚀 Enhanced Customer Foot Traffic Analyzer Initialized!")
        self._check_sophon_status()

    def run(self) -> None:
        while True:
            self._menu()
            choice = input("Enter choice (1-6): ").strip()
            if choice == "1":
                self.setup_roi()
            elif choice == "2":
                self.process_video()
            elif choice == "3":
                self.process_live()
            elif choice == "4":
                self.view_reports()
            elif choice == "5":
                self.show_settings()
            elif choice == "6":
                print("Goodbye 👋")
                break
            else:
                print("Invalid choice, try again.")
            input("\nPress Enter to continue...")

    def _menu(self) -> None:
        print("\n" + "=" * 60)
        print("🌟 Enhanced Customer Foot Traffic Analyzer 🌟")
        print("=" * 60)
        print("1. 🎯 Configure ROI")
        print("2. 📹 Analyze Recorded Video")
        print("3. 🔴 Live Camera Analysis")
        print("4. 📊 View Saved Reports")
        print("5. ⚙️  Settings")
        print("6. 🚪 Exit")
        print("=" * 60)

    def setup_roi(self) -> None:
        use_video = input("Use existing video for ROI? (y/n): ").strip().lower()
        if use_video == "y":
            video_path = input("Video path: ").strip()
            if not os.path.exists(video_path):
                print("Video file not found.")
                return
            self.roi_config = self.roi_selector.select_roi_from_video_stream(video_path)
        else:
            self.roi_config = self.roi_selector.select_roi_interactive()
        if self.roi_config.get("zones"):
            print("ROI configuration saved.")
        else:
            print("No ROI zones recorded.")

    def process_video(self) -> None:
        video_path = input("Video path to analyze: ").strip()
        if not os.path.exists(video_path):
            print("Video file not found.")
            return
        if self.roi_config is None:
            print("No ROI configured yet. A window will open to draw ROI while the video plays.")
            self._select_roi_from_video(video_path)
        else:
            refresh = input("Redefine ROI for this video? (y/N): ").strip().lower()
            if refresh == "y":
                self._select_roi_from_video(video_path)
        if not self._has_valid_roi():
            return
        try:
            self.current_processor = VideoProcessor(self.roi_config)
            report = self.current_processor.process_video(video_path)
            self._display_summary(report)
        except Exception as exc:
            print(f"Video processing failed: {exc}")

    def process_live(self) -> None:
        camera_index = input("Camera index (default 0): ").strip() or "0"
        duration = input("Duration seconds (optional): ").strip()
        duration_val = float(duration) if duration else None
        if self.roi_config is None:
            print("No ROI configured yet. A live preview window will open to draw ROI.")
            self._select_roi_from_camera(int(camera_index))
        else:
            refresh = input("Redefine ROI from camera feed? (y/N): ").strip().lower()
            if refresh == "y":
                self._select_roi_from_camera(int(camera_index))
        if not self._has_valid_roi():
            return
        try:
            processor = LiveProcessor(self.roi_config)
            report = processor.start_live_processing(int(camera_index), duration_val)
            self._display_summary(report)
        except Exception as exc:
            print(f"Live processing failed: {exc}")

    def view_reports(self) -> None:
        reports = sorted(
            [f for f in os.listdir(Config.REPORT_OUTPUT_DIR) if f.endswith(".json")],
            reverse=True,
        )
        if not reports:
            print("No reports found.")
            return
        for idx, filename in enumerate(reports[:10], start=1):
            print(f"{idx}. {filename}")
        choice = input("Select report number (Enter to cancel): ").strip()
        if not choice:
            return
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(reports):
                path = os.path.join(Config.REPORT_OUTPUT_DIR, reports[idx])
                self._display_saved_report(path)
        except Exception:
            print("Invalid selection.")

    def show_settings(self) -> None:
        print(f"Output directory: {Config.OUTPUT_DIR}")
        print(f"Frame skip: {Config.VIDEO.FRAME_SKIP}")
        print(f"Confidence threshold: {Config.VIDEO.CONFIDENCE_THRESHOLD}")
        print(f"Save video output: {Config.ANALYSIS.SAVE_VIDEO_OUTPUT}")
        print(f"Generate reports: {Config.ANALYSIS.GENERATE_REPORTS}")

    def _display_summary(self, report: Dict) -> None:
        analytics = report.get("analytics", {})
        print("\n=== Summary ===")
        print(f"Total People: {analytics.get('total_people', 0)}")
        print(f"Bypassers: {analytics.get('bypassers', 0)}")
        print(f"Current Inside: {analytics.get('current_inside', 0)}")
        print(f"Avg Dwell Time: {analytics.get('avg_dwell_time', 0):.1f}s")
        proc = report.get("processing_info", {})
        print(f"Processing Time: {proc.get('processing_time_seconds', 0):.1f}s")
        print(f"Average FPS: {proc.get('average_fps', 0):.1f}")
        if proc.get("output_video_path"):
            print(f"Output video: {proc['output_video_path']}")

    def _display_saved_report(self, path: str) -> None:
        import json

        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
        print(f"\nReport: {os.path.basename(path)}")
        self._display_summary(report)

    def _select_roi_from_video(self, video_path: str) -> None:
        try:
            roi = self.roi_selector.select_roi_from_video_stream(video_path)
        except ValueError as exc:
            print(f"❌ {exc}")
            return
        if roi.get("zones"):
            self.roi_config = roi
            print("ROI updated.")
        else:
            print("No ROI zones recorded.")

    def _select_roi_from_camera(self, camera_index: int) -> None:
        try:
            roi = self.roi_selector.select_roi_from_camera_stream(camera_index)
        except ValueError as exc:
            print(f"❌ {exc}")
            return
        if roi.get("zones"):
            self.roi_config = roi
            print("ROI updated.")
        else:
            print("No ROI zones recorded.")

    def _has_valid_roi(self) -> bool:
        if not self.roi_config or not self.roi_config.get("zones"):
            print("No ROI configuration available. Please try again.")
            return False
        return True

    def _check_sophon_status(self) -> None:
        """Check and display Sophon SDK status."""
        try:
            from utils.sophon_utils import is_sophon_available, find_bmodel
            if is_sophon_available():
                bmodel = find_bmodel()
                if bmodel:
                    print(f"✅ Sophon TPU detected - Using BModel: {bmodel}")
                else:
                    print("⚠️  Sophon SDK available but no BModel found - will use PyTorch fallback")
            else:
                print("ℹ️  Sophon SDK not available - using PyTorch/YOLO (if available)")
        except ImportError:
            pass  # Utils not critical for operation


if __name__ == "__main__":
    try:
        TrafficAnalyzer().run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

