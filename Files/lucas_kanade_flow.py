import math

import cv2
import numpy as np
from motion import points_to_box


def _cluster_points(points, cluster_distance):
    """Greedy centroid-based clustering — groups nearby points into separate clusters."""
    if not points:
        return []

    clusters = []
    assigned = [False] * len(points)

    for i, pt in enumerate(points):
        if assigned[i]:
            continue
        cluster = [pt]
        assigned[i] = True

        changed = True
        while changed:
            changed = False
            cx = sum(p[0] for p in cluster) / len(cluster)
            cy = sum(p[1] for p in cluster) / len(cluster)
            for j, other_pt in enumerate(points):
                if assigned[j]:
                    continue
                if math.sqrt((other_pt[0] - cx) ** 2 + (other_pt[1] - cy) ** 2) <= cluster_distance:
                    cluster.append(other_pt)
                    assigned[j] = True
                    changed = True

        clusters.append(cluster)

    return clusters


class LucasKanadeProcessor:
    def __init__(self, config):
        self.config = config
        self.prev_points = None
        self.frame_index = 0

    def _detect_features(self, gray):
        return cv2.goodFeaturesToTrack(
            gray,
            mask=None,
            maxCorners=self.config.max_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance,
            blockSize=self.config.block_size,
        )

    def process_frame(self, prev_gray, gray, frame):
        annotated = frame.copy()
        self.frame_index += 1

        if self.prev_points is None or self.frame_index % self.config.feature_reinit_interval == 0:
            self.prev_points = self._detect_features(prev_gray)

        boxes = []

        if self.prev_points is not None and len(self.prev_points) > 0:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                self.prev_points,
                None,
                winSize=self.config.win_size,
                maxLevel=self.config.max_level,
            )

            if next_points is not None and status is not None:
                good_new = next_points[status.flatten() == 1]
                good_old = self.prev_points[status.flatten() == 1]

                moving_points = []

                for new_pt, old_pt in zip(good_new, good_old):
                    a, b = new_pt.ravel()
                    c, d = old_pt.ravel()

                    motion = np.sqrt((a - c) ** 2 + (b - d) ** 2)

                    if motion >= self.config.point_motion_threshold:
                        moving_points.append((int(a), int(b)))
                        cv2.circle(annotated, (int(a), int(b)), 3, (0, 0, 255), -1)
                        cv2.line(
                            annotated,
                            (int(c), int(d)),
                            (int(a), int(b)),
                            (255, 0, 0),
                            1,
                        )

                for cluster in _cluster_points(moving_points, self.config.cluster_distance):
                    if len(cluster) >= self.config.min_points_for_box:
                        box = points_to_box(cluster, padding=self.config.box_padding)
                        if box is not None:
                            boxes.append(box)

                self.prev_points = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None
            else:
                self.prev_points = None

        cv2.putText(
            annotated,
            "Method: Lucas-Kanade",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        for (x, y, w, h) in boxes:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)

        return annotated, boxes