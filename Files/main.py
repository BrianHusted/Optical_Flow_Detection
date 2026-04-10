import argparse
import cv2

from config import VideoConfig, FarnebackConfig, LucasKanadeConfig, TrackerConfig
from metrics import MethodMetrics
from utils import ensure_output_dirs, open_video, get_video_writer, to_gray
from farneback_flow import process_farneback_frame
from lucas_kanade_flow import LucasKanadeProcessor
from tracker import CentroidTracker


def draw_tracked_objects(frame, tracked_objects, color=(255, 255, 255)):
    for obj_id, data in tracked_objects.items():
        x, y, w, h = data["box"]
        cx, cy = data["centroid"]

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (cx, cy), 4, color, -1)
        cv2.putText(
            frame,
            f"ID {obj_id}",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )


def run_farneback(video_cfg, farneback_cfg, tracker_cfg):
    cap = open_video(video_cfg.input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = get_video_writer(video_cfg.farneback_output, fps, width, height)

    tracker = CentroidTracker(
        max_distance=tracker_cfg.max_distance,
        max_missed_frames=tracker_cfg.max_missed_frames,
    )
    met = MethodMetrics("Farneback")

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame.")

    prev_gray = to_gray(prev_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = to_gray(frame)
        met.start_frame()
        annotated, boxes, _, _ = process_farneback_frame(prev_gray, gray, frame, farneback_cfg)
        tracked = tracker.update(boxes)
        met.end_frame(tracked)
        draw_tracked_objects(annotated, tracked, color=(255, 255, 255))

        writer.write(annotated)
        prev_gray = gray

    cap.release()
    writer.release()
    met.finalize()
    met.save_csv("outputs/metrics/farneback_metrics.csv")
    print(f"Saved Farneback output to: {video_cfg.farneback_output}")
    return met.summary()


def run_lk(video_cfg, lk_cfg, tracker_cfg):
    cap = open_video(video_cfg.input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = get_video_writer(video_cfg.lk_output, fps, width, height)

    tracker = CentroidTracker(
        max_distance=tracker_cfg.max_distance,
        max_missed_frames=tracker_cfg.max_missed_frames,
    )
    met = MethodMetrics("Lucas-Kanade")

    lk_processor = LucasKanadeProcessor(lk_cfg)

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame.")

    prev_gray = to_gray(prev_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = to_gray(frame)
        met.start_frame()
        annotated, boxes = lk_processor.process_frame(prev_gray, gray, frame)
        tracked = tracker.update(boxes)
        met.end_frame(tracked)
        draw_tracked_objects(annotated, tracked, color=(255, 255, 255))

        writer.write(annotated)
        prev_gray = gray

    cap.release()
    writer.release()
    met.finalize()
    met.save_csv("outputs/metrics/lk_metrics.csv")
    print(f"Saved Lucas-Kanade output to: {video_cfg.lk_output}")
    return met.summary()


def print_comparison(summaries):
    metrics = [
        ("avg_fps",                    "Avg FPS"),
        ("avg_objects_per_frame",      "Avg Objects/Frame"),
        ("total_unique_tracks",        "Total Unique Tracks"),
        ("avg_track_lifetime_frames",  "Avg Track Lifetime (frames)"),
        ("total_frames",               "Total Frames Processed"),
    ]
    col = 32
    print("\n" + "=" * (col + 18 * len(summaries)))
    print("METHOD COMPARISON SUMMARY")
    print("=" * (col + 18 * len(summaries)))
    print(f"{'Metric':<{col}}" + "".join(f"{s['method']:>18}" for s in summaries))
    print("-" * (col + 18 * len(summaries)))
    for key, label in metrics:
        print(f"{label:<{col}}" + "".join(f"{s[key]:>18}" for s in summaries))
    print("=" * (col + 18 * len(summaries)) + "\n")
    print("Per-frame CSVs saved to: outputs/metrics/")


def main():
    parser = argparse.ArgumentParser(description="Optical flow motion detection and tracking")
    parser.add_argument(
        "--method",
        choices=["farneback", "lk", "both"],
        default="both",
        help="Method to run",
    )
    args = parser.parse_args()

    ensure_output_dirs()

    video_cfg = VideoConfig()
    farneback_cfg = FarnebackConfig()
    lk_cfg = LucasKanadeConfig()
    tracker_cfg = TrackerConfig()

    summaries = []

    if args.method in ["farneback", "both"]:
        summaries.append(run_farneback(video_cfg, farneback_cfg, tracker_cfg))

    if args.method in ["lk", "both"]:
        summaries.append(run_lk(video_cfg, lk_cfg, tracker_cfg))

    if len(summaries) > 1:
        print_comparison(summaries)


if __name__ == "__main__":
    main()