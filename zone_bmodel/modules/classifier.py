from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .tracker import PersonTracker


class MovementClassifier:
    """
    Zone-based double validation system for entry/exit counting.
    Uses Zone Sequence (A→B or B→A) + Vector Validation (inward/outward direction).
    """

    def __init__(self, roi_config: Dict):
        self.roi_config = roi_config or {"zones": []}
        self.zone_a: Optional[np.ndarray] = None  # Outer zone
        self.zone_b: Optional[np.ndarray] = None  # Inner zone
        self.zone_a_color: Tuple[int, int, int] = (173, 216, 230)  # Light blue
        self.zone_b_color: Tuple[int, int, int] = (255, 165, 0)  # Orange
        self._process_roi_config()
        
        # Vector analysis parameters
        self.VECTOR_HISTORY_SIZE = 10  # Number of points to analyze for vector direction
        self.MIN_VECTOR_MAGNITUDE = 5.0  # Minimum movement to consider vector valid

    def _process_roi_config(self) -> None:
        """Extract Zone A (outer) and Zone B (inner) from config."""
        zones = self.roi_config.get("zones", [])
        for zone in zones:
            zone_type = zone.get("type", "").lower()
            points = zone.get("points", [])
            
            if not points:
                continue
                
            zone_array = np.array(points, dtype=np.int32)
            
            if zone_type == "outer":
                self.zone_a = zone_array
                if "color" in zone:
                    self.zone_a_color = tuple(zone["color"])
            elif zone_type == "inner":
                self.zone_b = zone_array
                if "color" in zone:
                    self.zone_b_color = tuple(zone["color"])

    def _point_in_zone(self, point: Tuple[float, float], zone: np.ndarray) -> bool:
        """Check if a point is inside a zone polygon."""
        if zone is None or len(zone) < 3:
            return False
        result = cv2.pointPolygonTest(zone, point, False)
        return result >= 0

    def _get_current_zone(self, point: Tuple[float, float]) -> Optional[str]:
        """Determine which zone the point is currently in."""
        in_zone_a = self._point_in_zone(point, self.zone_a) if self.zone_a is not None else False
        in_zone_b = self._point_in_zone(point, self.zone_b) if self.zone_b is not None else False
        
        if in_zone_b:
            return "B"
        elif in_zone_a:
            return "A"
        return None

    def _calculate_movement_vector(self, trajectory: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """
        Calculate movement vector from last N points.
        Returns (dx, dy) vector or None if insufficient data.
        """
        if len(trajectory) < 2:
            return None
        
        # Use last N points (or all if less than N)
        history_size = min(self.VECTOR_HISTORY_SIZE, len(trajectory))
        if history_size < 2:
            return None
        
        recent_points = trajectory[-history_size:]
        
        # Calculate average direction vector
        start_point = np.array(recent_points[0])
        end_point = np.array(recent_points[-1])
        
        vector = end_point - start_point
        magnitude = np.linalg.norm(vector)
        
        # Only return vector if movement is significant
        if magnitude < self.MIN_VECTOR_MAGNITUDE:
            return None
        
        return tuple(vector)

    def _is_inward_vector(self, vector: Tuple[float, float]) -> bool:
        """
        Check if vector points inward (toward camera/inside mall).
        Inward = positive Y direction (downward in image coordinates).
        """
        dx, dy = vector
        # Inward movement: primarily downward (positive dy)
        # We check if dy is positive and significant relative to dx
        return dy > abs(dx) * 0.5  # More downward than horizontal

    def _is_outward_vector(self, vector: Tuple[float, float]) -> bool:
        """
        Check if vector points outward (away from camera/outside mall).
        Outward = negative Y direction (upward in image coordinates).
        """
        dx, dy = vector
        # Outward movement: primarily upward (negative dy)
        # We check if dy is negative and significant relative to dx
        return dy < -abs(dx) * 0.5  # More upward than horizontal

    def _check_zone_sequence(self, zone_history: List[Optional[str]], current_zone: Optional[str]) -> Tuple[bool, bool]:
        """
        Check if zone sequence indicates entry (A→B) or exit (B→A).
        Entry: Currently in B and has been in A before
        Exit: Currently in A and has been in B before
        Returns (has_entry_sequence, has_exit_sequence)
        """
        if len(zone_history) < 2:
            return False, False
        
        # Entry sequence: Currently in Zone B and previously was in Zone A
        has_entry_sequence = False
        if current_zone == "B":
            # Check if person was in Zone A at any point before reaching Zone B
            for i in range(len(zone_history) - 1):
                if zone_history[i] == "A":
                    # Found A, now check if we transitioned to B
                    for j in range(i + 1, len(zone_history)):
                        if zone_history[j] == "B":
                            has_entry_sequence = True
                            break
                    if has_entry_sequence:
                        break
        
        # Exit sequence: Currently in Zone A and previously was in Zone B
        has_exit_sequence = False
        if current_zone == "A":
            # Check if person was in Zone B at any point before reaching Zone A
            for i in range(len(zone_history) - 1):
                if zone_history[i] == "B":
                    # Found B, now check if we transitioned to A
                    for j in range(i + 1, len(zone_history)):
                        if zone_history[j] == "A":
                            has_exit_sequence = True
                            break
                    if has_exit_sequence:
                        break
        
        return has_entry_sequence, has_exit_sequence

    def classify_person_movement(self, tracker: PersonTracker, person_id: int) -> str:
        """
        Double validation: Zone Sequence + Vector Direction.
        Entry: A→B sequence + inward vector
        Exit: B→A sequence + outward vector
        """
        track_data = tracker.get_track_data(person_id)
        if not track_data:
            return "unknown"
        
        trajectory = list(track_data.get("trajectory", []))
        if len(trajectory) < 2:
            return "unknown"
        
        current_point = trajectory[-1]
        current_zone = self._get_current_zone(current_point)
        
        # Get or initialize zone history
        zone_history = track_data.get("zone_history", [])
        zone_history.append(current_zone)
        # Keep only last 50 zone states to avoid memory issues
        if len(zone_history) > 50:
            zone_history = zone_history[-50:]
        track_data["zone_history"] = zone_history
        
        # Calculate movement vector
        movement_vector = self._calculate_movement_vector(trajectory)
        
        # Check zone sequence
        has_entry_sequence, has_exit_sequence = self._check_zone_sequence(zone_history, current_zone)
        
        # Double validation for ENTRY: A→B sequence + inward vector
        entry_validated = False
        if has_entry_sequence and movement_vector is not None:
            if self._is_inward_vector(movement_vector):
                entry_validated = True
        
        # Double validation for EXIT: B→A sequence + outward vector
        exit_validated = False
        if has_exit_sequence and movement_vector is not None:
            if self._is_outward_vector(movement_vector):
                exit_validated = True
        
        # Mark entry/exit events (only once per person)
        if entry_validated and not track_data.get("entry_crossed", False):
            track_data["entry_crossed"] = True
            track_data["entry_crossed_frame"] = len(trajectory) - 1
        
        if exit_validated and not track_data.get("exit_crossed", False):
            track_data["exit_crossed"] = True
            track_data["exit_crossed_frame"] = len(trajectory) - 1
        
        # Classification based on validated events
        entry_crossed = track_data.get("entry_crossed", False)
        exit_crossed = track_data.get("exit_crossed", False)
        
        if entry_crossed and exit_crossed:
            # Both entry and exit crossed - person completed a cycle
            return "exited"
        elif entry_crossed:
            # Entered but hasn't exited yet - currently inside
            return "inside"
        elif exit_crossed:
            # Exited without entering first - someone leaving
            return "exited"
        elif current_zone == "B":
            # In inner zone but hasn't validated entry yet
            return "pending_inner"
        elif current_zone == "A":
            # In outer zone
            return "pending_outer"
        else:
            return "unknown"

    def draw_classification_zones(self, image: np.ndarray) -> np.ndarray:
        """Draw Zone A and Zone B polygons on the frame - only thin outlines, no fill."""
        # Draw only thin outlines - no fill to keep video completely visible
        
        # Draw Zone A (outer) - Very thin outline only
        if self.zone_a is not None:
            # Draw very thin, semi-transparent outline (1 pixel, light color)
            cv2.polylines(image, [self.zone_a], True, (150, 150, 150), 1)
            # Add small label (optional, can be removed if too visible)
            if len(self.zone_a) > 0:
                centroid = np.mean(self.zone_a, axis=0).astype(int)
                cv2.putText(
                    image,
                    "A",
                    tuple(centroid),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (180, 180, 180),
                    1,
                )
        
        # Draw Zone B (inner) - Very thin outline only
        if self.zone_b is not None:
            # Draw very thin, semi-transparent outline (1 pixel, light color)
            cv2.polylines(image, [self.zone_b], True, (150, 150, 150), 1)
            # Add small label (optional, can be removed if too visible)
            if len(self.zone_b) > 0:
                centroid = np.mean(self.zone_b, axis=0).astype(int)
                cv2.putText(
                    image,
                    "B",
                    tuple(centroid),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (180, 180, 180),
                    1,
                )
        
        return image
