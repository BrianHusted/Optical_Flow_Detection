import cv2
import numpy as np

from motion import extract_bounding_boxes, points_to_box


def _distance_sq(p1, p2):
    dx = float(p1[0]) - float(p2[0])
    dy = float(p1[1]) - float(p2[1])
    return dx * dx + dy * dy


def _cluster_points(points, cluster_distance):
    if not points:
        return []

    dist_sq_thresh = cluster_distance * cluster_distance
    n = len(points)
    visited = [False] * n
    clusters = []

    for i in range(n):
        if visited[i]:
            continue

        queue = [i]
        visited[i] = True
        cluster = []

        while queue:
            idx = queue.pop()
            cluster.append(points[idx])

            for j in range(n):
                if visited[j]:
                    continue
                if _distance_sq(points[idx], points[j]) <= dist_sq_thresh:
                    visited[j] = True
                    queue.append(j)

        clusters.append(cluster)

    return clusters


class LucasKanadeProcessor:
    def __init__(self, config):
        self.config = config
        self.prev_points = None
        self.frame_index = 0
        self.last_boxes = []

        self.lk_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            getattr(config, "lk_criteria_count", 10),
            getattr(config, "lk_criteria_eps", 0.03),
        )

        self.fb_max_error = getattr(config, "fb_max_error", 2.0)
        self.lk_max_error = getattr(config, "lk_max_error", 30.0)
        self.max_point_motion = getattr(config, "max_point_motion", 100.0)
        self.min_tracked_points = getattr(config, "min_tracked_points", 25)
        self.roi_padding = getattr(config, "roi_padding", 30)
        self.draw_tracks = getattr(config, "draw_tracks", True)

        # Motion-mask defaults
        self.motion_diff_threshold = getattr(config, "motion_diff_threshold", 12)
        self.motion_min_contour_area = getattr(config, "motion_min_contour_area", 250)
        self.motion_open_kernel = getattr(config, "motion_open_kernel", 3)
        self.motion_close_kernel = getattr(config, "motion_close_kernel", 9)
        self.motion_dilate_kernel = getattr(config, "motion_dilate_kernel", 5)
        self.min_box_support_points = getattr(config, "min_box_support_points", 2)

    def _expand_box(self, box, width, height, padding):
        x, y, w, h = box
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width - 1, x + w + padding)
        y2 = min(height - 1, y + h + padding)
        return x1, y1, x2, y2

    def _pad_box(self, box, width, height, padding):
        x, y, w, h = box
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width - 1, x + w + padding)
        y2 = min(height - 1, y + h + padding)
        return (x1, y1, x2 - x1, y2 - y1)

    def _build_roi_mask(self, gray, boxes):
        if not boxes:
            return None

        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        for box in boxes:
            x1, y1, x2, y2 = self._expand_box(box, w, h, self.roi_padding)
            mask[y1:y2 + 1, x1:x2 + 1] = 255

        return mask

    def _make_motion_mask(self, prev_gray, gray):
        prev_blur = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        diff = cv2.absdiff(prev_blur, gray_blur)
        _, mask = cv2.threshold(
            diff,
            self.motion_diff_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        open_k = np.ones((self.motion_open_kernel, self.motion_open_kernel), np.uint8)
        close_k = np.ones((self.motion_close_kernel, self.motion_close_kernel), np.uint8)
        dilate_k = np.ones((self.motion_dilate_kernel, self.motion_dilate_kernel), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
        mask = cv2.dilate(mask, dilate_k, iterations=1)

        return mask

    def _detect_features(self, gray, mask=None):
        return cv2.goodFeaturesToTrack(
            gray,
            mask=mask,
            maxCorners=self.config.max_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance,
            blockSize=self.config.block_size,
        )

    def _detect_new_features(self, gray, motion_mask, existing_points=None, boxes=None):
        h, w = gray.shape

        roi_mask = self._build_roi_mask(gray, boxes)
        if roi_mask is None:
            mask = motion_mask.copy()
        else:
            mask = cv2.bitwise_or(motion_mask, roi_mask)

        if mask is None or np.count_nonzero(mask) == 0:
            mask = np.full((h, w), 255, dtype=np.uint8)

        if existing_points is not None and len(existing_points) > 0:
            for pt in existing_points.reshape(-1, 2):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(mask, (x, y), self.config.min_distance, 0, -1)

        return self._detect_features(gray, mask=mask)

    def _merge_points(self, base_points, new_points):
        if base_points is None or len(base_points) == 0:
            if new_points is None:
                return None
            return new_points.reshape(-1, 1, 2).astype(np.float32)

        if new_points is None or len(new_points) == 0:
            return base_points.reshape(-1, 1, 2).astype(np.float32)

        merged = np.vstack([base_points.reshape(-1, 2), new_points.reshape(-1, 2)])

        if len(merged) > self.config.max_corners:
            merged = merged[: self.config.max_corners]

        return merged.reshape(-1, 1, 2).astype(np.float32)

    def _filter_tracks(self, prev_gray, gray, next_points, status, err):
        if next_points is None or status is None:
            return None, None, None

        status = status.reshape(-1).astype(bool)
        if not np.any(status):
            return None, None, None

        forward_new = next_points[status].reshape(-1, 1, 2)
        forward_old = self.prev_points[status].reshape(-1, 1, 2)
        forward_err = err[status].reshape(-1) if err is not None else None

        back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray,
            prev_gray,
            forward_new,
            None,
            winSize=self.config.win_size,
            maxLevel=self.config.max_level,
            criteria=self.lk_criteria,
        )

        if back_points is None or back_status is None:
            return None, None, None

        back_status = back_status.reshape(-1).astype(bool)
        if not np.any(back_status):
            return None, None, None

        forward_new = forward_new[back_status]
        forward_old = forward_old[back_status]
        if forward_err is not None:
            forward_err = forward_err[back_status]
        back_points = back_points[back_status]

        fb_error = np.linalg.norm(
            forward_old.reshape(-1, 2) - back_points.reshape(-1, 2),
            axis=1,
        )
        motion = np.linalg.norm(
            forward_new.reshape(-1, 2) - forward_old.reshape(-1, 2),
            axis=1,
        )

        valid = fb_error <= self.fb_max_error
        valid &= motion <= self.max_point_motion

        if forward_err is not None:
            valid &= forward_err <= self.lk_max_error

        if not np.any(valid):
            return None, None, None

        good_new = forward_new[valid].reshape(-1, 2)
        good_old = forward_old[valid].reshape(-1, 2)
        good_motion = motion[valid]

        return good_new, good_old, good_motion

    def _point_in_box(self, point, box):
        px, py = point
        x, y, w, h = box
        return (x <= px <= x + w) and (y <= py <= y + h)

    def process_frame(self, prev_gray, gray, frame):
        annotated = frame.copy()
        self.frame_index += 1
        boxes = []

        motion_mask = self._make_motion_mask(prev_gray, gray)

        # Bootstrap / refresh tracking points from actual motion regions in previous frame
        need_reset = (
            self.prev_points is None
            or len(self.prev_points) == 0
            or self.frame_index % self.config.feature_reinit_interval == 0
        )
        if need_reset:
            prev_roi = self._build_roi_mask(prev_gray, self.last_boxes)
            detect_mask = motion_mask.copy()
            if prev_roi is not None:
                detect_mask = cv2.bitwise_or(detect_mask, prev_roi)

            if np.count_nonzero(detect_mask) == 0:
                detect_mask = None

            self.prev_points = self._detect_features(prev_gray, mask=detect_mask)

        good_new = None
        good_old = None
        good_motion = None

        if self.prev_points is not None and len(self.prev_points) > 0:
            next_points, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray,
                gray,
                self.prev_points,
                None,
                winSize=self.config.win_size,
                maxLevel=self.config.max_level,
                criteria=self.lk_criteria,
            )

            good_new, good_old, good_motion = self._filter_tracks(
                prev_gray, gray, next_points, status, err
            )

        moving_points = []

        if good_new is not None and good_old is not None and good_motion is not None:
            h, w = gray.shape

            for new_pt, old_pt, motion_mag in zip(good_new, good_old, good_motion):
                a, b = new_pt
                c, d = old_pt

                xi = int(np.clip(round(a), 0, w - 1))
                yi = int(np.clip(round(b), 0, h - 1))

                # Count only motion-supported LK points
                if motion_mag >= self.config.point_motion_threshold and motion_mask[yi, xi] > 0:
                    moving_points.append((xi, yi))

                    if self.draw_tracks:
                        cv2.circle(annotated, (xi, yi), 3, (0, 0, 255), -1)
                        cv2.line(
                            annotated,
                            (int(c), int(d)),
                            (xi, yi),
                            (255, 0, 0),
                            1,
                        )

        # Primary boxes come from motion regions, validated by LK support points
        candidate_boxes = extract_bounding_boxes(
            motion_mask,
            min_contour_area=self.motion_min_contour_area,
        )

        for box in candidate_boxes:
            support = sum(1 for pt in moving_points if self._point_in_box(pt, box))
            if support >= self.min_box_support_points:
                boxes.append(self._pad_box(box, gray.shape[1], gray.shape[0], self.config.box_padding))

        # Fallback: if motion mask is weak but LK still has grouped moving points
        if not boxes and moving_points:
            for cluster in _cluster_points(moving_points, self.config.cluster_distance):
                if len(cluster) >= self.config.min_points_for_box:
                    box = points_to_box(cluster, padding=self.config.box_padding)
                    if box is not None:
                        boxes.append(box)

        # Keep only currently tracked points for next frame
        current_points = (
            good_new.reshape(-1, 1, 2).astype(np.float32)
            if good_new is not None and len(good_new) > 0
            else None
        )

        # Replenish using current-frame motion evidence
        need_replenish = (
            current_points is None
            or len(current_points) < self.min_tracked_points
        )
        if need_replenish:
            replenish_boxes = boxes if boxes else self.last_boxes
            new_features = self._detect_new_features(
                gray,
                motion_mask=motion_mask,
                existing_points=current_points,
                boxes=replenish_boxes,
            )
            current_points = self._merge_points(current_points, new_features)

        self.prev_points = current_points
        self.last_boxes = boxes

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