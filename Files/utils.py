import cv2
import os


def ensure_output_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)


def open_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    return cap


def get_video_writer(output_path: str, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {output_path}")
    return writer


def to_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)