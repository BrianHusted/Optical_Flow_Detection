import subprocess
import sys
import os

VIDEOS = [
    ("data/videos/HD/person_walking_HD.mp4",        "person_walking"),
    ("data/videos/HD/car_moving_away_HD.mp4",        "car_moving_away"),
    ("data/videos/HD/car_moving_diagonally_HD.mp4",  "car_moving_diagonal"),
    ("data/videos/HD/multiple_cars_HD.mp4",          "multiple_cars"),
]


def main():
    missing = [p for p, _ in VIDEOS if not os.path.exists(p)]
    if missing:
        print("ERROR: Missing video files:")
        for p in missing:
            print(f"  {p}")
        sys.exit(1)

    total = len(VIDEOS)
    for idx, (video_path, condition) in enumerate(VIDEOS, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{total}] {os.path.basename(video_path)}  |  condition: {condition}")
        print("="*60)

        for method in ["farneback", "lk"]:
            print(f"\n  -> method: {method}")
            subprocess.run(
                [
                    sys.executable, "src/main.py",
                    "--video",     video_path,
                    "--method",    method,
                    "--condition", condition,
                ],
                check=True,
            )

    print("\n" + "="*60)
    print("All runs complete. Results saved to outputs/metrics/results.csv")
    print("Run  python plot_results.py  to generate comparison charts.")
    print("="*60)


if __name__ == "__main__":
    main()
