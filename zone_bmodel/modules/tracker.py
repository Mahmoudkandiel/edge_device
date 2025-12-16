from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import distance


class PersonTracker:
    """
    Lightweight centroid tracker enhanced with trajectory, entry/exit, and dwell-time metadata.
    """

    def __init__(self, max_disappeared: int = 50, max_distance: float = 50.0) -> None:
        self.next_person_id: int = 0
        self.tracks: Dict[int, Dict] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.track_data: Dict[int, Dict] = defaultdict(
            lambda: {
                "trajectory": deque(maxlen=1000),
                "entry_point": None,
                "exit_point": None,
                "first_seen": None,
                "last_seen": None,
                "entry_time": None,
                "exit_time": None,
                "dwell_time": None,
                "classification": "unknown",
                "entry_crossed": False,
                "exit_crossed": False,
                "entry_crossed_frame": None,
                "exit_crossed_frame": None,
                "zone_history": [],  # Track zone sequence for double validation
            }
        )

    def update(self, detections: List[Dict], frame_count: int, timestamp: float) -> Dict:
        if not detections:
            for person_id in list(self.disappeared.keys()):
                self.disappeared[person_id] += 1
                if self.disappeared[person_id] > self.max_disappeared:
                    self._remove_person(person_id)
            return self.tracks

        if not self.tracks:
            for detection in detections:
                self._add_new_person(detection, frame_count, timestamp)
            return self.tracks

        person_ids = list(self.tracks.keys())
        person_centroids = np.array([self.tracks[pid]["centroid"] for pid in person_ids])
        detection_centroids = np.array([d["centroid"] for d in detections])

        D = distance.cdist(person_centroids, detection_centroids) if len(person_centroids) and len(detection_centroids) else np.empty((0, 0))

        # Improved matching: use Hungarian algorithm approach with better distance handling
        rows = D.min(axis=1).argsort() if D.size else []
        cols = D.argmin(axis=1) if D.size else []
        used_rows, used_cols = set(), set()

        # First pass: match closest pairs
        for row in rows:
            if row in used_rows:
                continue
            col = cols[row]
            if col in used_cols:
                continue
            if D[row, col] < self.max_distance:
                person_id = person_ids[row]
                self._update_person(person_id, detections[col], frame_count, timestamp)
                used_rows.add(row)
                used_cols.add(col)
        
        # Second pass: try to match remaining with adaptive distance threshold
        remaining_rows = [r for r in range(len(person_ids)) if r not in used_rows]
        remaining_cols = [c for c in range(len(detections)) if c not in used_cols]
        if remaining_rows and remaining_cols:
            remaining_person_ids = [person_ids[r] for r in remaining_rows]
            remaining_centroids = np.array([self.tracks[pid]["centroid"] for pid in remaining_person_ids])
            remaining_detections = np.array([detections[c]["centroid"] for c in remaining_cols])
            if len(remaining_centroids) > 0 and len(remaining_detections) > 0:
                D2 = distance.cdist(remaining_centroids, remaining_detections)
                for i, row_idx in enumerate(remaining_rows):
                    min_dist_idx = D2[i].argmin()
                    if D2[i, min_dist_idx] < self.max_distance * 1.5:  # Slightly more lenient for second pass
                        col_idx = remaining_cols[min_dist_idx]
                        if col_idx not in used_cols:
                            person_id = person_ids[row_idx]
                            self._update_person(person_id, detections[col_idx], frame_count, timestamp)
                            used_rows.add(row_idx)
                            used_cols.add(col_idx)

        unused_rows = set(range(len(person_ids))) - used_rows
        unused_cols = set(range(len(detections))) - used_cols

        for row in unused_rows:
            person_id = person_ids[row]
            self.disappeared[person_id] += 1
            if self.disappeared[person_id] > self.max_disappeared:
                self._remove_person(person_id)

        for col in unused_cols:
            self._add_new_person(detections[col], frame_count, timestamp)

        return self.tracks

    def _add_new_person(self, detection: Dict, frame_count: int, timestamp: float) -> None:
        person_id = self.next_person_id
        self.next_person_id += 1
        centroid = detection["centroid"]
        bbox = detection["bbox"]
        self.tracks[person_id] = {"centroid": centroid, "bbox": bbox, "start_frame": frame_count}
        self.disappeared[person_id] = 0
        track_info = self.track_data[person_id]
        track_info["trajectory"].append(centroid)
        track_info["entry_point"] = centroid
        track_info["first_seen"] = timestamp
        track_info["entry_time"] = timestamp
        track_info["last_seen"] = timestamp

    def _update_person(self, person_id: int, detection: Dict, frame_count: int, timestamp: float) -> None:
        centroid = detection["centroid"]
        bbox = detection["bbox"]
        old_centroid = self.tracks[person_id]["centroid"]
        
        # Smooth centroid update for better tracking stability
        if old_centroid is not None:
            # Use weighted average for smoother tracking (70% new, 30% old)
            smoothed_centroid = (
                0.7 * np.array(centroid) + 0.3 * np.array(old_centroid)
            )
            centroid = tuple(smoothed_centroid)
        
        self.tracks[person_id]["centroid"] = centroid
        self.tracks[person_id]["bbox"] = bbox
        self.disappeared[person_id] = 0
        track_info = self.track_data[person_id]
        
        # Only add to trajectory if movement is significant (reduces noise)
        if not track_info["trajectory"] or len(track_info["trajectory"]) == 0:
            track_info["trajectory"].append(centroid)
        else:
            last_point = track_info["trajectory"][-1]
            dist = np.sqrt((centroid[0] - last_point[0])**2 + (centroid[1] - last_point[1])**2)
            if dist > 2.0:  # Only add if moved at least 2 pixels
                track_info["trajectory"].append(centroid)
        
        track_info["last_seen"] = timestamp

    def _remove_person(self, person_id: int) -> None:
        if person_id not in self.tracks:
            return
        track_info = self.track_data[person_id]
        if track_info["entry_time"] and track_info["last_seen"]:
            track_info["dwell_time"] = track_info["last_seen"] - track_info["entry_time"]
            if track_info["trajectory"]:
                track_info["exit_point"] = track_info["trajectory"][-1]
                track_info["exit_time"] = track_info["last_seen"]
        del self.tracks[person_id]
        del self.disappeared[person_id]

    def get_track_data(self, person_id: int) -> Dict:
        return self.track_data.get(person_id, {})

    def get_all_tracks_data(self) -> Dict[int, Dict]:
        return dict(self.track_data)

    def classify_movement(self, person_id: int, roi_lines: List[Dict]) -> str:
        track_info = self.track_data.get(person_id)
        if not track_info or len(track_info["trajectory"]) < 2:
            return "unknown"
        trajectory = list(track_info["trajectory"])
        for line in roi_lines:
            points = line.get("points", [])
            if len(points) >= 2 and self._check_trajectory_intersection(trajectory, points):
                return line.get("type", "unknown")
        return "unknown"

    def draw_tracks(self, image: np.ndarray) -> np.ndarray:
        colors = {
            "entrant": (0, 255, 0),
            "inside": (255, 165, 0),
            "exited": (255, 0, 0),
            "bypasser": (0, 255, 255),
            "pending_inner": (255, 255, 0),
            "pending_outer": (173, 216, 230),
            "unknown": (128, 128, 128),
        }
        for person_id, track in self.tracks.items():
            centroid = tuple(map(int, track["centroid"]))
            x1, y1, x2, y2 = track["bbox"]
            classification = self.track_data[person_id].get("classification", "unknown")
            color = colors.get(classification, (128, 128, 128))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.circle(image, centroid, 4, color, -1)
            trajectory: Deque[Tuple[int, int]] = self.track_data[person_id]["trajectory"]
            if len(trajectory) > 1:
                pts = np.array([tuple(map(int, point)) for point in trajectory], np.int32)
                cv2.polylines(image, [pts], False, color, 1)
            label = f"ID:{person_id} ({classification})"
            cv2.putText(
                image,
                label,
                (centroid[0] + 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        return image

    def _check_trajectory_intersection(self, trajectory: List[Tuple], line_points: List[Tuple]) -> bool:
        if len(trajectory) < 2 or len(line_points) < 2:
            return False
        line_start = np.array(line_points[0])
        line_end = np.array(line_points[1])
        for i in range(len(trajectory) - 1):
            traj_start = np.array(trajectory[i])
            traj_end = np.array(trajectory[i + 1])
            if self._line_intersection(traj_start, traj_end, line_start, line_end):
                return True
        return False

    @staticmethod
    def _line_intersection(line1_start, line1_end, line2_start, line2_end) -> bool:
        def cross(a, b):
            return a[0] * b[1] - a[1] * b[0]

        r = line1_end - line1_start
        s = line2_end - line2_start
        cross_rs = cross(r, s)
        if cross_rs == 0:
            return False
        t = cross(line2_start - line1_start, s) / cross_rs
        u = cross(line2_start - line1_start, r) / cross_rs
        return 0 <= t <= 1 and 0 <= u <= 1

