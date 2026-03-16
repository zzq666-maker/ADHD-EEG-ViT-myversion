import argparse
from pathlib import Path

import numpy as np
import scipy.io


def load_mat_signal(path: Path):
    mat = scipy.io.loadmat(str(path))
    if path.stem not in mat:
        raise KeyError(f"Variable '{path.stem}' not found in {path}")
    return mat[path.stem]


def summarize_signal(arr: np.ndarray):
    channel_means = np.mean(arr, axis=0)
    channel_stds = np.std(arr, axis=0)
    timepoint_mean_across_channels = np.mean(arr, axis=1)

    summary = {
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "overall_min": float(np.min(arr)),
        "overall_max": float(np.max(arr)),
        "overall_mean": float(np.mean(arr)),
        "overall_std": float(np.std(arr)),
        "allclose_to_integer": bool(np.allclose(arr, np.round(arr))),
        "channel_mean_abs_avg": float(np.mean(np.abs(channel_means))),
        "channel_std_avg": float(np.mean(channel_stds)),
        "channel_std_min": float(np.min(channel_stds)),
        "channel_std_max": float(np.max(channel_stds)),
        "timepoint_mean_abs_avg": float(np.mean(np.abs(timepoint_mean_across_channels))),
        "first_channel_mean": float(channel_means[0]),
        "first_channel_std": float(channel_stds[0]),
        "first_row_first5": arr[0, :5].tolist(),
    }
    return summary


def heuristic_judgement(summary: dict):
    notes = []

    if summary["allclose_to_integer"]:
        notes.append("Data values look integer-like, which is more consistent with raw exported EEG than z-scored data.")
    else:
        notes.append("Data values are not integer-like, which may indicate floating-point preprocessing or original float export.")

    if summary["channel_mean_abs_avg"] < 1e-3:
        notes.append("Per-channel mean is extremely close to 0, which is consistent with per-channel centering/z-score.")
    else:
        notes.append("Per-channel mean is not near 0, so the file does not look already centered per channel.")

    if 0.8 <= summary["channel_std_avg"] <= 1.2:
        notes.append("Average per-channel std is near 1, which is consistent with z-score normalization.")
    else:
        notes.append("Average per-channel std is not near 1, so the file does not look already z-scored.")

    if summary["timepoint_mean_abs_avg"] < 1e-3:
        notes.append("Mean across channels at each time point is near 0, which is consistent with common-average referencing.")
    else:
        notes.append("Mean across channels at each time point is not near 0, so common-average reference is not obvious.")

    return notes


def main():
    parser = argparse.ArgumentParser(description="Inspect raw EEG .mat files for signs of preprocessing.")
    parser.add_argument(
        "files",
        nargs="*",
        default=[
            "data_raw/ADHD_part1/v1p.mat",
            "data_raw/ADHD_part2/v177.mat",
            "data_raw/Control_part1/v41p.mat",
            "data_raw/Control_part2/v117.mat",
        ],
        help="Paths to .mat files to inspect.",
    )
    args = parser.parse_args()

    for file_str in args.files:
        path = Path(file_str)
        arr = load_mat_signal(path)
        summary = summarize_signal(arr)
        notes = heuristic_judgement(summary)

        print(f"\n=== {path} ===")
        for k, v in summary.items():
            print(f"{k}: {v}")
        print("notes:")
        for note in notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
