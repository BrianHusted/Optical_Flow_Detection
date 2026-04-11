# Computer Vision Project - Motion Detection and Object Tracking in Video
Using Optical Flow Techniques

This project compares two optical flow methods for motion detection and tracking in video:

- Lucas-Kanade
- Farneback

## Requirements

- Python 3.8+
- opencv-python
- numpy
- matplotlib
- tqdm

## Setup

1. Clone the repository and navigate to the project folder:

```bash
git clone https://github.com/BrianHusted/Optical_Flow_Detection.git
cd Optical_Flow_Detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your video files in the following folder structure:

```
data/
└── videos/
    └── HD/
        ├── person_walking_HD.mp4
        ├── car_moving_away_HD.mp4
        ├── car_moving_diagonally_HD.mp4
        ├── multiple_cars_HD.mp4
        └── fireworks_moving_camera.mp4
```

The dataset can be downloaded from the link below.

**Video Dataset:** https://drive.google.com/drive/u/2/folders/1A244gWzA6GVo5W3Ff_avtFlKfWQt8qGZ

## Running the Project

### Run all videos with both methods

```bash
python run_all.py
```

This processes all five videos using both Farneback and Lucas-Kanade and saves results to `outputs/metrics/results.csv`.

### Run a single video

```bash
python src/main.py --video data/videos/HD/person_walking_HD.mp4 --method both
```

**Options:**

| Flag | Values | Description |
|---|---|---|
| `--video` | path to video file | Input video to process |
| `--method` | `farneback`, `lk`, `both` | Which method(s) to run (default: `both`) |
| `--condition` | any string | Label for this run in the results CSV |

### Generate comparison charts

After running, generate the comparison plots with:

```bash
python plot_results.py
```

This reads `outputs/metrics/results.csv` and saves a chart to `outputs/metrics/comparison_plots.png`.

## Output

All outputs are saved to the `outputs/` folder:

```
outputs/
├── <video_name>_farneback.mp4
├── <video_name>_lk.mp4
└── metrics/
    ├── results.csv
    └── comparison_plots.png
```

Pre-generated output videos and results are also available in the Google Drive link above.
