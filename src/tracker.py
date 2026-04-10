import math


class CentroidTracker:
    def __init__(self, max_distance=50.0, max_missed_frames=10):
        self.next_id = 0
        self.objects = {}  # id -> {"centroid": (x, y), "box": (x, y, w, h), "missed": int}
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames

    def _centroid(self, box):
        x, y, w, h = box
        return (x + w // 2, y + h // 2)

    def _distance(self, c1, c2):
        return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)

    def update(self, boxes):
        new_centroids = [self._centroid(box) for box in boxes]

        matched_ids = set()
        matched_boxes = set()

        object_ids = list(self.objects.keys())

        for obj_id in object_ids:
            old_centroid = self.objects[obj_id]["centroid"]

            best_idx = None
            best_dist = float("inf")

            for i, centroid in enumerate(new_centroids):
                if i in matched_boxes:
                    continue
                dist = self._distance(old_centroid, centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx is not None and best_dist <= self.max_distance:
                self.objects[obj_id]["centroid"] = new_centroids[best_idx]
                self.objects[obj_id]["box"] = boxes[best_idx]
                self.objects[obj_id]["missed"] = 0
                matched_ids.add(obj_id)
                matched_boxes.add(best_idx)
            else:
                self.objects[obj_id]["missed"] += 1

        for obj_id in list(self.objects.keys()):
            if self.objects[obj_id]["missed"] > self.max_missed_frames:
                del self.objects[obj_id]

        for i, box in enumerate(boxes):
            if i not in matched_boxes:
                self.objects[self.next_id] = {
                    "centroid": new_centroids[i],
                    "box": box,
                    "missed": 0,
                }
                self.next_id += 1

        return self.objects