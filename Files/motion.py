import cv2
import numpy as np


def flow_to_magnitude(flow: np.ndarray) -> np.ndarray:
    fx = flow[..., 0]
    fy = flow[..., 1]
    magnitude = np.sqrt(fx**2 + fy**2)
    return magnitude


def motion_mask_from_magnitude(magnitude: np.ndarray, threshold: float) -> np.ndarray:
    mask = (magnitude > threshold).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    return mask


def extract_bounding_boxes(mask: np.ndarray, min_contour_area: int):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_contour_area:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))

    return boxes


def points_to_box(points, padding=10):
    if len(points) == 0:
        return None

    pts = np.array(points, dtype=np.int32)
    x_min = np.min(pts[:, 0]) - padding
    y_min = np.min(pts[:, 1]) - padding
    x_max = np.max(pts[:, 0]) + padding
    y_max = np.max(pts[:, 1]) + padding

    x_min = max(0, x_min)
    y_min = max(0, y_min)

    return (x_min, y_min, x_max - x_min, y_max - y_min)