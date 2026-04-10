import csv
import os
import time


class MethodMetrics:
    def __init__(self, method_name):
        self.method_name = method_name
        self.frame_times = []
        self.object_counts = []
        self._track_lifetimes = {}
        self._active_ids = {}
        self._frame_idx = 0
        self._frame_start = None

    def start_frame(self):
        self._frame_start = time.perf_counter()

    def end_frame(self, tracked_objects):
        self.frame_times.append(time.perf_counter() - self._frame_start)
        self.object_counts.append(len(tracked_objects))

        current_ids = set(tracked_objects.keys())

        for obj_id in current_ids:
            if obj_id not in self._active_ids:
                self._active_ids[obj_id] = self._frame_idx

        for obj_id in list(self._active_ids):
            if obj_id not in current_ids:
                lifetime = self._frame_idx - self._active_ids.pop(obj_id)
                self._track_lifetimes[obj_id] = lifetime

        self._frame_idx += 1

    def finalize(self):
        for obj_id, start in self._active_ids.items():
            self._track_lifetimes[obj_id] = self._frame_idx - start

    def avg_fps(self):
        if not self.frame_times:
            return 0.0
        return 1.0 / (sum(self.frame_times) / len(self.frame_times))

    def avg_object_count(self):
        if not self.object_counts:
            return 0.0
        return sum(self.object_counts) / len(self.object_counts)

    def avg_track_lifetime(self):
        if not self._track_lifetimes:
            return 0.0
        return sum(self._track_lifetimes.values()) / len(self._track_lifetimes)

    def summary(self):
        return {
            "method": self.method_name,
            "avg_fps": round(self.avg_fps(), 2),
            "avg_objects_per_frame": round(self.avg_object_count(), 2),
            "total_unique_tracks": len(self._track_lifetimes),
            "avg_track_lifetime_frames": round(self.avg_track_lifetime(), 2),
            "total_frames": len(self.frame_times),
        }

    def save_csv(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "processing_time_s", "object_count"])
            for i, (t, c) in enumerate(zip(self.frame_times, self.object_counts)):
                writer.writerow([i, round(t, 6), c])
