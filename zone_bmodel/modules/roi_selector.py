import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ROIZone:
    id: str
    name: str
    type: str  # "outer" or "inner"
    points: List[Tuple[int, int]] = field(default_factory=list)
    color: List[int] = field(default_factory=list)


class ROISelector:
    """
    ROI selector that lets the user draw zones (polygons) for Zone A (outer) and Zone B (inner).
    """

    def __init__(self) -> None:
        self.window_name = "ROI Zone Selector"
        self.zones: List[ROIZone] = []
        self.current_zone_type: Optional[str] = None  # "outer" or "inner"
        self.current_points: List[Tuple[int, int]] = []  # Points being drawn for current zone
        self.current_frame: Optional[np.ndarray] = None
        self.paused = False
        self.drawing = False  # Whether currently drawing a zone

    def select_roi_interactive(
        self,
        video_path: Optional[str] = None,
        frame: Optional[np.ndarray] = None,
    ) -> Dict:
        if video_path:
            return self.select_roi_from_video_stream(video_path)
        self._reset_state()
        if frame is not None:
            self.current_frame = frame.copy()
        else:
            self.current_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        return self._run_static_loop()

    def select_roi_from_video_stream(self, video_path: str) -> Dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        def next_frame():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            return frame if ret else None

        config = self._run_stream_loop(next_frame, "video")
        cap.release()
        return config

    def select_roi_from_camera_stream(self, camera_index: int = 0) -> Dict:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera index: {camera_index}")

        def next_frame():
            ret, frame = cap.read()
            return frame if ret else None

        config = self._run_stream_loop(next_frame, f"camera {camera_index}")
        cap.release()
        return config

    def _run_static_loop(self) -> Dict:
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        confirmed: Dict = {}

        while True:
            display = self.current_frame.copy()
            self._draw_zones(display)
            self._draw_instructions(display, streaming=False)
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            action = self._handle_key(key, streaming=False)
            if action == "exit":
                confirmed = {}
                break
            if action == "confirm":
                confirmed = self._get_config()
                break

        cv2.destroyWindow(self.window_name)
        return confirmed

    def _run_stream_loop(self, frame_supplier, label: str) -> Dict:
        self._reset_state()
        print(
            f"\n🎥 ROI Zone selection is running on {label}.\n"
            " - Press 'a' to draw Zone A (Outer), 'b' to draw Zone B (Inner).\n"
            " - Click points to draw polygon, press 'f' to finish current zone.\n"
            " - Press 'p' to pause / resume the stream.\n"
            " - Press 'o' to confirm and save, or 'q' to cancel.\n"
        )

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        confirmed: Dict = {}

        while True:
            if not self.paused or self.current_frame is None:
                frame = frame_supplier()
                if frame is None:
                    print("⚠️ Unable to fetch a frame from the source.")
                    break
                self.current_frame = frame.copy()

            display = self.current_frame.copy()
            self._draw_zones(display)
            self._draw_instructions(display, streaming=True)
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(20) & 0xFF
            action = self._handle_key(key, streaming=True)
            if action == "exit":
                confirmed = {}
                break
            if action == "confirm":
                confirmed = self._get_config()
                break

        cv2.destroyWindow(self.window_name)
        return confirmed

    def _reset_state(self) -> None:
        self.zones = []
        self.current_zone_type = None
        self.current_points = []
        self.current_frame = None
        self.paused = False
        self.drawing = False

    def _mouse_callback(self, event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_zone_type is None:
                print("⚠️ Please select zone type first: 'a' for Zone A (Outer) or 'b' for Zone B (Inner)")
                return
            
            if not self.drawing:
                # Start drawing a new zone
                self.drawing = True
                self.current_points = []
            
            # Add point to current zone
            self.current_points.append((int(x), int(y)))
            print(f"Point {len(self.current_points)} added at ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click to finish current zone (if has at least 3 points)
            if self.drawing and len(self.current_points) >= 3:
                self._finish_current_zone()

    def _finish_current_zone(self) -> None:
        """Finish drawing the current zone and add it to zones list."""
        if not self.drawing or len(self.current_points) < 3:
            return
        
        zone_id = f"zone_{self.current_zone_type}"
        zone_name = f"Zone A (Outer)" if self.current_zone_type == "outer" else "Zone B (Inner)"
        zone_color = [173, 216, 230] if self.current_zone_type == "outer" else [255, 165, 0]
        
        # Remove old zone of same type if exists
        self.zones = [z for z in self.zones if z.type != self.current_zone_type]
        
        # Add new zone
        zone = ROIZone(
            id=zone_id,
            name=zone_name,
            type=self.current_zone_type,
            points=self.current_points.copy(),
            color=zone_color
        )
        self.zones.append(zone)
        
        print(f"✅ {zone_name} completed with {len(self.current_points)} points")
        self.current_points = []
        self.drawing = False
        self.current_zone_type = None

    def _draw_zones(self, image: np.ndarray) -> None:
        # Draw completed zones
        for zone in self.zones:
            if len(zone.points) < 3:
                continue
            
            points_array = np.array(zone.points, dtype=np.int32)
            color = tuple(zone.color) if zone.color else (173, 216, 230)
            
            # Fill polygon
            cv2.fillPoly(image, [points_array], color)
            # Draw polygon outline
            cv2.polylines(image, [points_array], True, (0, 0, 255), 2)
            
            # Add label
            if len(zone.points) > 0:
                centroid = np.mean(points_array, axis=0).astype(int)
                cv2.putText(
                    image,
                    zone.name,
                    tuple(centroid),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
        
        # Draw current zone being drawn
        if self.drawing and len(self.current_points) > 0:
            temp_color = (173, 216, 230) if self.current_zone_type == "outer" else (255, 165, 0)
            
            # Draw points
            for point in self.current_points:
                cv2.circle(image, point, 5, temp_color, -1)
            
            # Draw lines connecting points
            if len(self.current_points) > 1:
                for i in range(len(self.current_points) - 1):
                    cv2.line(image, self.current_points[i], self.current_points[i + 1], temp_color, 2)
            
            # Draw line from last point to mouse (if we could track it)
            # For now, just show the points

    def _draw_instructions(self, image: np.ndarray, streaming: bool) -> None:
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (400, 220), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        
        lines = [
            "[a] Zone A (Outer - Light Blue)",
            "[b] Zone B (Inner - Orange)",
            "[f] Finish current zone",
            "[c] Cancel current zone",
            "[d] Delete last zone",
            "[r] Reset all zones",
            "[s] Save JSON",
        ]
        if streaming:
            lines.append("[p] Pause / Resume")
        lines.append("[o] OK / Confirm")
        lines.append("[q] Cancel")
        
        if self.drawing:
            lines.insert(0, f"Drawing {self.current_zone_type.upper()} zone...")
            lines.insert(1, f"Points: {len(self.current_points)} (Right-click to finish)")
        
        for idx, text in enumerate(lines):
            cv2.putText(
                image,
                text,
                (20, 40 + idx * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

    def _handle_key(self, key: int, streaming: bool) -> Optional[str]:
        if key == 255:
            return None

        key_char = None
        if 0 <= key <= 255:
            try:
                key_char = chr(key).lower()
            except ValueError:
                key_char = None

        if key == ord("q"):
            return "exit"
        if key == ord("o"):
            if not self.zones:
                print("⚠️ Add at least one zone before confirming.")
                return None
            return "confirm"
        
        if key_char == "a":
            if self.drawing:
                print("⚠️ Finish current zone first (press 'f' or right-click)")
            else:
                self.current_zone_type = "outer"
                self.drawing = True
                self.current_points = []
                print("🎨 Drawing Zone A (Outer) - Click points to draw polygon")
        elif key_char == "b":
            if self.drawing:
                print("⚠️ Finish current zone first (press 'f' or right-click)")
            else:
                self.current_zone_type = "inner"
                self.drawing = True
                self.current_points = []
                print("🎨 Drawing Zone B (Inner) - Click points to draw polygon")
        elif key_char == "f":
            # Finish current zone
            self._finish_current_zone()
        elif key_char == "c":
            # Cancel current zone
            if self.drawing:
                self.current_points = []
                self.drawing = False
                self.current_zone_type = None
                print("❌ Current zone cancelled")
        elif key_char == "d":
            # Delete last zone
            if self.zones:
                removed = self.zones.pop()
                print(f"🗑️ Removed {removed.name}")
        elif key_char == "r":
            # Reset all
            self.zones.clear()
            self.current_points = []
            self.drawing = False
            self.current_zone_type = None
            print("🔄 All zones cleared")
        elif key_char == "s":
            self.save_config()
        elif streaming and key_char == "p":
            self.paused = not self.paused
            state = "PAUSED" if self.paused else "RESUMED"
            print(f"Stream {state}.")
        return None

    def _get_config(self) -> Dict:
        """Convert zones to config format compatible with MovementClassifier."""
        zones_config = []
        for zone in self.zones:
            zones_config.append({
                "id": zone.id,
                "name": zone.name,
                "type": zone.type,
                "points": zone.points,
                "color": zone.color
            })
        
        return {
            "zones": zones_config
        }

    def save_config(self, filename: str = "roi_config.json") -> None:
        config = self._get_config()
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"✅ ROI configuration saved to {filename}")

    def load_config(self, filename: str = "roi_config.json") -> Optional[Dict]:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Config file {filename} not found.")
            return None
        
        # Load zones
        self.zones = []
        for zone_data in data.get("zones", []):
            zone = ROIZone(
                id=zone_data.get("id", ""),
                name=zone_data.get("name", ""),
                type=zone_data.get("type", ""),
                points=[tuple(p) for p in zone_data.get("points", [])],
                color=zone_data.get("color", [])
            )
            self.zones.append(zone)
        
        print(f"✅ ROI configuration loaded from {filename}")
        return data


if __name__ == "__main__":
    selector = ROISelector()
    selector.select_roi_interactive()
