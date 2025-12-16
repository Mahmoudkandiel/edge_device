import os
from typing import Optional, Tuple

import cv2


def create_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def calculate_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def draw_text_with_background(
    image,
    text: str,
    position: Tuple[int, int],
    bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.6,
    thickness: int = 1,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x, y = position
    cv2.rectangle(image, (x, y - text_size[1] - 5), (x + text_size[0] + 5, y + 5), bg_color, -1)
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness)
    return text_size


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def get_video_info(video_path: str) -> Tuple[float, int, int, int]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, width, height, total_frames


def check_camera_availability(camera_index: int = 0) -> bool:
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        cap.release()
        return True
    return False


class FPSCounter:
    def __init__(self) -> None:
        self.times = []

    def update(self) -> None:
        self.times.append(cv2.getTickCount())
        if len(self.times) > 100:
            self.times.pop(0)

    def get_fps(self) -> float:
        if len(self.times) < 2:
            return 0.0
        ticks = self.times[-1] - self.times[0]
        freq = cv2.getTickFrequency()
        return (len(self.times) - 1) * freq / ticks if ticks else 0.0


def read_first_frame(video_path: str):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def capture_camera_frame(camera_index: int = 0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

