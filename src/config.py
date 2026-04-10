from dataclasses import dataclass


@dataclass
class VideoConfig:
    input_video: str = "data/videos/HD/person_walking_HD.mp4"
    farneback_output: str = "outputs/farneback_output.mp4"
    lk_output: str = "outputs/lk_output.mp4"


@dataclass
class FarnebackConfig:
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0

    motion_threshold: float = 3.0
    min_contour_area: int = 1200


@dataclass
class LucasKanadeConfig:
    max_corners: int = 200
    quality_level: float = 0.3
    min_distance: int = 7
    block_size: int = 7

    win_size: tuple = (15, 15)
    max_level: int = 2

    feature_reinit_interval: int = 10
    point_motion_threshold: float = 2.0
    min_points_for_box: int = 6
    box_padding: int = 10
    cluster_distance: int = 50


@dataclass
class TrackerConfig:
    max_distance: float = 50.0
    max_missed_frames: int = 10