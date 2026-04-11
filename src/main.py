import argparse
import csv
import cv2
import os
import time

from tqdm import tqdm


from config import VideoConfig, FarnebackConfig, LucasKanadeConfig, TrackerConfig
from utils import ensure_output_dirs, open_video, get_video_writer, to_gray
from farneback_flow import process_farneback_frame
from lucas_kanade_flow import LucasKanadeProcessor
from tracker import CentroidTracker


RESULTS_CSV_PATH = "outputs/metrics/results.csv"


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


def ensure_results_csv():
    os.makedirs("outputs/metrics", exist_ok=True)

    if not os.path.exists(RESULTS_CSV_PATH):
        with open(RESULTS_CSV_PATH, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "video_name",
                "condition",
                "method",
                "output_video",
                "total_frames",
                "frames_with_detection",
                "percent_frames_with_detection",
                "total_boxes",
                "avg_boxes_per_frame",
                "avg_tracked_objects_per_frame",
                "max_tracked_objects_in_frame",
                "total_processing_time_sec",
                "avg_time_per_frame_sec",
                "fps_estimate",
                "manual_tracking_rating",
                "manual_noise_rating",
                "notes",
            ])


def append_results_row(row):
    with open(RESULTS_CSV_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def summarize_and_save_results(
    video_name,
    condition,
    method,
    output_video,
    total_frames,
    frames_with_detection,
    total_boxes,
    tracked_objects_total,
    max_tracked_objects_in_frame,
    total_processing_time,
    manual_tracking_rating,
    manual_noise_rating,
    notes,
):
    percent_frames_with_detection = (
        (frames_with_detection / total_frames) * 100.0 if total_frames > 0 else 0.0
    )
    avg_boxes_per_frame = total_boxes / total_frames if total_frames > 0 else 0.0
    avg_tracked_objects_per_frame = (
        tracked_objects_total / total_frames if total_frames > 0 else 0.0
    )
    avg_time_per_frame = (
        total_processing_time / total_frames if total_frames > 0 else 0.0
    )
    fps_estimate = total_frames / total_processing_time if total_processing_time > 0 else 0.0

    append_results_row([
        video_name,
        condition,
        method,
        output_video,
        total_frames,
        frames_with_detection,
        f"{percent_frames_with_detection:.2f}",
        total_boxes,
        f"{avg_boxes_per_frame:.4f}",
        f"{avg_tracked_objects_per_frame:.4f}",
        max_tracked_objects_in_frame,
        f"{total_processing_time:.4f}",
        f"{avg_time_per_frame:.6f}",
        f"{fps_estimate:.4f}",
        manual_tracking_rating,
        manual_noise_rating,
        notes,
    ])

    print("\n--- Run Summary ---")
    print(f"Video: {video_name}")
    print(f"Condition: {condition}")
    print(f"Method: {method}")
    print(f"Output video: {output_video}")
    print(f"Total frames processed: {total_frames}")
    print(f"Frames with detection: {frames_with_detection}")
    print(f"Percent frames with detection: {percent_frames_with_detection:.2f}%")
    print(f"Total boxes: {total_boxes}")
    print(f"Average boxes per frame: {avg_boxes_per_frame:.4f}")
    print(f"Average tracked objects per frame: {avg_tracked_objects_per_frame:.4f}")
    print(f"Max tracked objects in a frame: {max_tracked_objects_in_frame}")
    print(f"Total processing time: {total_processing_time:.4f} sec")
    print(f"Average time per frame: {avg_time_per_frame:.6f} sec")
    print(f"Estimated FPS: {fps_estimate:.4f}")
    print(f"Saved metrics row to: {RESULTS_CSV_PATH}")


def run_farneback(
    video_cfg,
    farneback_cfg,
    tracker_cfg,
    condition,
    manual_tracking_rating,
    manual_noise_rating,
    notes,
):
    cap = open_video(video_cfg.input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = get_video_writer(video_cfg.farneback_output, fps, width, height)

    tracker = CentroidTracker(
        max_distance=tracker_cfg.max_distance,
        max_missed_frames=tracker_cfg.max_missed_frames,
    )

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame.")

    prev_gray = to_gray(prev_frame)

    total_frames = 0
    frames_with_detection = 0
    total_boxes = 0
    tracked_objects_total = 0
    max_tracked_objects_in_frame = 0
    total_processing_time = 0.0

    video_label = os.path.basename(video_cfg.input_video)
    pbar = tqdm(total=total_frame_count - 1, desc=f"Farneback | {video_label}", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.perf_counter()

        gray = to_gray(frame)
        annotated, boxes, _, _ = process_farneback_frame(prev_gray, gray, frame, farneback_cfg)
        tracked = tracker.update(boxes)
        draw_tracked_objects(annotated, tracked, color=(255, 255, 255))
        writer.write(annotated)

        frame_end = time.perf_counter()

        total_frames += 1
        total_processing_time += (frame_end - frame_start)
        total_boxes += len(boxes)
        tracked_count = len(tracked)
        tracked_objects_total += tracked_count
        max_tracked_objects_in_frame = max(max_tracked_objects_in_frame, tracked_count)

        if len(boxes) > 0:
            frames_with_detection += 1

        elapsed = frame_end - frame_start
        pbar.set_postfix(fps=f"{1/elapsed:.1f}" if elapsed > 0 else "?", objects=tracked_count)
        pbar.update(1)

        prev_gray = gray

    pbar.close()

    cap.release()
    writer.release()

    summarize_and_save_results(
        video_name=os.path.basename(video_cfg.input_video),
        condition=condition,
        method="Farneback",
        output_video=video_cfg.farneback_output,
        total_frames=total_frames,
        frames_with_detection=frames_with_detection,
        total_boxes=total_boxes,
        tracked_objects_total=tracked_objects_total,
        max_tracked_objects_in_frame=max_tracked_objects_in_frame,
        total_processing_time=total_processing_time,
        manual_tracking_rating=manual_tracking_rating,
        manual_noise_rating=manual_noise_rating,
        notes=notes,
    )

    print(f"Saved Farneback output to: {video_cfg.farneback_output}")


def run_lk(
    video_cfg,
    lk_cfg,
    tracker_cfg,
    condition,
    manual_tracking_rating,
    manual_noise_rating,
    notes,
):
    cap = open_video(video_cfg.input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = get_video_writer(video_cfg.lk_output, fps, width, height)

    tracker = CentroidTracker(
        max_distance=tracker_cfg.max_distance,
        max_missed_frames=tracker_cfg.max_missed_frames,
    )

    lk_processor = LucasKanadeProcessor(lk_cfg)

    ret, prev_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame.")

    prev_gray = to_gray(prev_frame)

    total_frames = 0
    frames_with_detection = 0
    total_boxes = 0
    tracked_objects_total = 0
    max_tracked_objects_in_frame = 0
    total_processing_time = 0.0

    video_label = os.path.basename(video_cfg.input_video)
    pbar = tqdm(total=total_frame_count - 1, desc=f"Lucas-Kanade | {video_label}", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.perf_counter()

        gray = to_gray(frame)
        annotated, boxes = lk_processor.process_frame(prev_gray, gray, frame)
        tracked = tracker.update(boxes)
        draw_tracked_objects(annotated, tracked, color=(255, 255, 255))
        writer.write(annotated)

        frame_end = time.perf_counter()

        total_frames += 1
        total_processing_time += (frame_end - frame_start)
        total_boxes += len(boxes)
        tracked_count = len(tracked)
        tracked_objects_total += tracked_count
        max_tracked_objects_in_frame = max(max_tracked_objects_in_frame, tracked_count)

        if len(boxes) > 0:
            frames_with_detection += 1

        elapsed = frame_end - frame_start
        pbar.set_postfix(fps=f"{1/elapsed:.1f}" if elapsed > 0 else "?", objects=tracked_count)
        pbar.update(1)

        prev_gray = gray

    pbar.close()

    cap.release()
    writer.release()

    summarize_and_save_results(
        video_name=os.path.basename(video_cfg.input_video),
        condition=condition,
        method="Lucas-Kanade",
        output_video=video_cfg.lk_output,
        total_frames=total_frames,
        frames_with_detection=frames_with_detection,
        total_boxes=total_boxes,
        tracked_objects_total=tracked_objects_total,
        max_tracked_objects_in_frame=max_tracked_objects_in_frame,
        total_processing_time=total_processing_time,
        manual_tracking_rating=manual_tracking_rating,
        manual_noise_rating=manual_noise_rating,
        notes=notes,
    )

    print(f"Saved Lucas-Kanade output to: {video_cfg.lk_output}")


def main():
    parser = argparse.ArgumentParser(description="Optical flow motion detection and tracking")
    parser.add_argument(
        "--method",
        choices=["farneback", "lk", "both"],
        default="both",
        help="Method to run",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="unspecified",
        help="Scenario label, e.g. static_one_person, fast_motion, camera_motion",
    )
    parser.add_argument(
        "--tracking-rating",
        type=str,
        default="",
        help="Manual rating you add after reviewing output, e.g. good/fair/poor",
    )
    parser.add_argument(
        "--noise-rating",
        type=str,
        default="",
        help="Manual noise rating, e.g. low/medium/high",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Extra notes for this run",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/videos/HD/person_walking_HD.mp4",
        help="Path to input video, e.g. data/videos/HD/multiple_cars_HD.mp4",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    ensure_results_csv()

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    video_cfg = VideoConfig(
        input_video=args.video,
        farneback_output=f"outputs/{video_name}_farneback.mp4",
        lk_output=f"outputs/{video_name}_lk.mp4",
    )
    farneback_cfg = FarnebackConfig()
    lk_cfg = LucasKanadeConfig()
    tracker_cfg = TrackerConfig()

    if args.method in ["farneback", "both"]:
        run_farneback(
            video_cfg=video_cfg,
            farneback_cfg=farneback_cfg,
            tracker_cfg=tracker_cfg,
            condition=args.condition,
            manual_tracking_rating=args.tracking_rating,
            manual_noise_rating=args.noise_rating,
            notes=args.notes,
        )

    if args.method in ["lk", "both"]:
        run_lk(
            video_cfg=video_cfg,
            lk_cfg=lk_cfg,
            tracker_cfg=tracker_cfg,
            condition=args.condition,
            manual_tracking_rating=args.tracking_rating,
            manual_noise_rating=args.noise_rating,
            notes=args.notes,
        )


if __name__ == "__main__":
    main()