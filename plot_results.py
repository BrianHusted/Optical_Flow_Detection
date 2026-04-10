"""
Reads outputs/metrics/results.csv and produces a 4-panel comparison figure
plus a printed summary table.

Usage:
    python plot_results.py
"""

import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH    = "outputs/metrics/results.csv"
OUTPUT_PATH = "outputs/metrics/comparison_plots.png"

# Friendly x-axis labels for each condition
CONDITION_LABELS = {
    "person_walking":    "Person\nWalking",
    "car_moving_away":   "Car Moving\nAway",
    "car_moving_diagonal": "Car Moving\nDiagonally",
    "multiple_cars":     "Multiple\nCars",
}

METHOD_COLORS = {
    "Farneback":    "#2196F3",   # blue
    "Lucas-Kanade": "#FF9800",   # orange
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results():
    if not os.path.exists(CSV_PATH):
        sys.exit(f"ERROR: {CSV_PATH} not found. Run  python run_all.py  first.")

    rows = []
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    if not rows:
        sys.exit("ERROR: results.csv is empty.")
    return rows


def dedup_latest(rows):
    """Keep the last recorded entry per (condition, method) to avoid duplicates."""
    seen = {}
    for row in rows:
        key = (row["condition"], row["method"])
        seen[key] = row
    return list(seen.values())


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def grouped_bar(ax, conditions, method_data, ylabel, title, fmt=".1f", multiplier=1.0):
    """
    method_data: dict  method_name -> list of values (same order as conditions)
    """
    x = np.arange(len(conditions))
    n = len(method_data)
    width = 0.35
    offsets = np.linspace(-(n - 1) * width / 2, (n - 1) * width / 2, n)

    for offset, (method, values) in zip(offsets, method_data.items()):
        bars = ax.bar(
            x + offset,
            [v * multiplier for v in values],
            width,
            label=method,
            color=METHOD_COLORS.get(method, "#999"),
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * ax.get_ylim()[1],
                    f"{val * multiplier:{fmt}}",
                    ha="center", va="bottom", fontsize=7.5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONDITION_LABELS.get(c, c) for c in conditions],
        fontsize=9,
    )
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rows = dedup_latest(load_results())

    # Collect all conditions present, preserving a sensible order
    order = ["person_walking", "car_moving_away", "car_moving_diagonal", "multiple_cars"]
    conditions = [c for c in order if any(r["condition"] == c for r in rows)]
    # append any unexpected conditions that appeared
    for r in rows:
        if r["condition"] not in conditions:
            conditions.append(r["condition"])

    methods = sorted({r["method"] for r in rows})

    def values_for(metric, cast=float):
        """Return {method: [value_per_condition]} dict."""
        result = {}
        for method in methods:
            vals = []
            for cond in conditions:
                match = next(
                    (r for r in rows if r["condition"] == cond and r["method"] == method),
                    None,
                )
                vals.append(cast(match[metric]) if match and match[metric] else 0.0)
            result[method] = vals
        return result

    fps_data      = values_for("fps_estimate")
    detect_data   = values_for("percent_frames_with_detection")
    objects_data  = values_for("avg_tracked_objects_per_frame")
    time_data     = values_for("avg_time_per_frame_sec")

    # ---- Figure ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        "Optical Flow Method Comparison — Farneback vs Lucas-Kanade",
        fontsize=13, fontweight="bold", y=1.01,
    )

    grouped_bar(axes[0, 0], conditions, fps_data,
                "Frames / second", "Processing Speed (FPS)", fmt=".1f")

    grouped_bar(axes[0, 1], conditions, detect_data,
                "% frames", "Detection Rate (% Frames with Motion)", fmt=".1f")

    grouped_bar(axes[1, 0], conditions, objects_data,
                "Avg objects / frame", "Tracking Density (Avg Objects per Frame)", fmt=".2f")

    grouped_bar(axes[1, 1], conditions, time_data,
                "Time (ms)", "Avg Processing Time per Frame",
                fmt=".1f", multiplier=1000.0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {OUTPUT_PATH}")
    plt.show()

    # ---- Summary table ----
    col_w = [22, 18, 8, 10, 10, 12]
    headers = ["Condition", "Method", "FPS", "Detect %", "Avg Obj", "ms/frame"]
    sep = "  ".join("-" * w for w in col_w)

    print("\n" + "=" * sum(col_w + [2] * len(col_w)))
    print("SUMMARY TABLE")
    print("=" * sum(col_w + [2] * len(col_w)))
    print("  ".join(h.ljust(w) for h, w in zip(headers, col_w)))
    print(sep)

    for cond in conditions:
        for method in methods:
            match = next(
                (r for r in rows if r["condition"] == cond and r["method"] == method),
                None,
            )
            if not match:
                continue
            fps   = float(match["fps_estimate"])
            det   = float(match["percent_frames_with_detection"])
            obj   = float(match["avg_tracked_objects_per_frame"])
            ms    = float(match["avg_time_per_frame_sec"]) * 1000
            row_vals = [
                CONDITION_LABELS.get(cond, cond).replace("\n", " "),
                method,
                f"{fps:.1f}",
                f"{det:.1f}",
                f"{obj:.2f}",
                f"{ms:.1f}",
            ]
            print("  ".join(v.ljust(w) for v, w in zip(row_vals, col_w)))
        print(sep)


if __name__ == "__main__":
    main()
