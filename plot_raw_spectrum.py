import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.signal import welch


FS = 128
DEFAULT_MAX_FREQ = 64.0
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]


def load_mat_signal(path: Path) -> np.ndarray:
    mat = scipy.io.loadmat(str(path))
    if path.stem in mat:
        signal = mat[path.stem]
    else:
        valid_keys = [key for key in mat.keys() if not key.startswith("__")]
        if not valid_keys:
            raise KeyError(f"No EEG array found in {path}")
        signal = mat[valid_keys[0]]

    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError(f"Expected 2D EEG array, got shape={signal.shape}")
    return signal


def parse_path_list(raw_value: str):
    parts = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not parts:
        raise ValueError("Expected at least one path")
    return [Path(item) for item in parts]


def collect_mat_files(paths):
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.mat")))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"Path not found: {path}")
    if not files:
        raise ValueError("No .mat files were found in the provided paths")
    return files


def resolve_channel_index(channel: str, n_channels: int) -> int:
    if channel.isdigit():
        index = int(channel)
        if not 0 <= index < n_channels:
            raise ValueError(f"Channel index out of range: {index}, n_channels={n_channels}")
        return index

    lower_names = [name.lower() for name in CHANNEL_NAMES[:n_channels]]
    if channel.lower() not in lower_names:
        raise ValueError(f"Unknown channel name: {channel}")
    return lower_names.index(channel.lower())


def compute_psd(signal: np.ndarray, fs: int, nperseg: int):
    nperseg = min(nperseg, signal.shape[0])
    if nperseg < 32:
        raise ValueError(f"Signal too short for PSD estimation: {signal.shape[0]} samples")

    freqs, psd = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
        axis=0,
    )
    return freqs, psd


def plot_channel_psd(ax, freqs, psd, channel_label: str, max_freq: float):
    mask = freqs <= max_freq
    ax.plot(freqs[mask], psd[mask], linewidth=1.5)
    ax.set_title(channel_label)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.grid(True, alpha=0.3)


def plot_compare_psd(ax, freqs_a, psd_a, label_a: str, freqs_b, psd_b, label_b: str, max_freq: float):
    mask_a = freqs_a <= max_freq
    mask_b = freqs_b <= max_freq
    ax.plot(freqs_a[mask_a], psd_a[mask_a], linewidth=1.8, label=label_a)
    ax.plot(freqs_b[mask_b], psd_b[mask_b], linewidth=1.8, label=label_b)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.grid(True, alpha=0.3)
    ax.legend()


def compute_group_mean_psd(file_paths, fs: int, nperseg: int, channel: str | None):
    group_psd = []
    used_files = []

    for file_path in file_paths:
        signal = load_mat_signal(file_path)
        freqs, psd = compute_psd(signal, fs=fs, nperseg=nperseg)
        if channel is None:
            curve = np.mean(psd, axis=1)
        else:
            channel_index = resolve_channel_index(channel, signal.shape[1])
            curve = psd[:, channel_index]
        group_psd.append(curve)
        used_files.append(file_path)

    return freqs, np.mean(np.stack(group_psd, axis=0), axis=0), used_files


def main():
    parser = argparse.ArgumentParser(description="Plot PSD spectrum from raw EEG .mat data.")
    parser.add_argument("file", help="Path to a .mat EEG file.")
    parser.add_argument("--compare-file", default=None, help="Optional second .mat file for side-by-side PSD comparison.")
    parser.add_argument(
        "--group-a",
        default=None,
        help="Comma-separated files or directories for group A average PSD, e.g. ADHD_part1,ADHD_part2.",
    )
    parser.add_argument(
        "--group-b",
        default=None,
        help="Comma-separated files or directories for group B average PSD, e.g. Control_part1,Control_part2.",
    )
    parser.add_argument("--group-a-label", default="Group A", help="Legend label for group A.")
    parser.add_argument("--group-b-label", default="Group B", help="Legend label for group B.")
    parser.add_argument("--fs", type=int, default=FS, help="Sampling rate in Hz.")
    parser.add_argument("--channel", default=None, help="Single channel name or 0-based index, e.g. Fp1 or 0.")
    parser.add_argument(
        "--mode",
        choices=["single", "mean", "all"],
        default="mean",
        help="single: one channel; mean: average PSD across channels; all: one subplot per channel.",
    )
    parser.add_argument("--nperseg", type=int, default=256, help="Welch segment length.")
    parser.add_argument(
        "--max-freq",
        type=float,
        default=DEFAULT_MAX_FREQ,
        help="Maximum frequency to display. Use 64 to inspect possible 50 Hz line noise.",
    )
    parser.add_argument("--save", default=None, help="Optional output image path.")
    args = parser.parse_args()

    if (args.group_a is None) != (args.group_b is None):
        raise ValueError("--group-a and --group-b must be provided together")

    if args.group_a is not None:
        group_a_files = collect_mat_files(parse_path_list(args.group_a))
        group_b_files = collect_mat_files(parse_path_list(args.group_b))
        group_channel = args.channel if args.mode == "single" else None
        freqs_a, mean_psd_a, used_a = compute_group_mean_psd(
            group_a_files,
            fs=args.fs,
            nperseg=args.nperseg,
            channel=group_channel,
        )
        freqs_b, mean_psd_b, used_b = compute_group_mean_psd(
            group_b_files,
            fs=args.fs,
            nperseg=args.nperseg,
            channel=group_channel,
        )

        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        label_suffix = f" ({len(used_a)} subjects)" if len(used_a) > 1 else ""
        compare_suffix = f" ({len(used_b)} subjects)" if len(used_b) > 1 else ""
        plot_compare_psd(
            ax,
            freqs_a,
            mean_psd_a,
            f"{args.group_a_label}{label_suffix}",
            freqs_b,
            mean_psd_b,
            f"{args.group_b_label}{compare_suffix}",
            args.max_freq,
        )
        if args.mode == "single":
            if args.channel is None:
                raise ValueError("--channel is required when --mode single is used")
            ax.set_title(f"Group PSD Comparison | {args.channel}")
        else:
            ax.set_title("Group PSD Comparison | Channel Mean")
        fig.tight_layout()
    else:
        file_path = Path(args.file)
        signal = load_mat_signal(file_path)
        n_timepoints, n_channels = signal.shape

        freqs, psd = compute_psd(signal, fs=args.fs, nperseg=args.nperseg)
        compare_path = Path(args.compare_file) if args.compare_file is not None else None
        compare_signal = None
        compare_freqs = None
        compare_psd = None
        compare_n_channels = None

        if compare_path is not None:
            compare_signal = load_mat_signal(compare_path)
            _, compare_n_channels = compare_signal.shape
            compare_freqs, compare_psd = compute_psd(compare_signal, fs=args.fs, nperseg=args.nperseg)

        if compare_path is not None:
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            if args.mode == "single":
                if args.channel is None:
                    raise ValueError("--channel is required when --mode single is used")
                channel_index_a = resolve_channel_index(args.channel, n_channels)
                channel_index_b = resolve_channel_index(args.channel, compare_n_channels)
                channel_label = CHANNEL_NAMES[channel_index_a] if channel_index_a < len(CHANNEL_NAMES) else f"Ch{channel_index_a}"
                plot_compare_psd(
                    ax,
                    freqs,
                    psd[:, channel_index_a],
                    f"{file_path.stem} | {channel_label}",
                    compare_freqs,
                    compare_psd[:, channel_index_b],
                    f"{compare_path.stem} | {channel_label}",
                    args.max_freq,
                )
                ax.set_title(f"PSD Comparison | {channel_label}")
            else:
                mean_psd = np.mean(psd, axis=1)
                compare_mean_psd = np.mean(compare_psd, axis=1)
                plot_compare_psd(
                    ax,
                    freqs,
                    mean_psd,
                    f"{file_path.stem} | Mean PSD",
                    compare_freqs,
                    compare_mean_psd,
                    f"{compare_path.stem} | Mean PSD",
                    args.max_freq,
                )
                ax.set_title("PSD Comparison | Channel Mean")
            fig.tight_layout()
        elif args.mode == "single":
            if args.channel is None:
                raise ValueError("--channel is required when --mode single is used")
            channel_index = resolve_channel_index(args.channel, n_channels)
            channel_label = CHANNEL_NAMES[channel_index] if channel_index < len(CHANNEL_NAMES) else f"Ch{channel_index}"
            fig, ax = plt.subplots(figsize=(8, 4.5))
            plot_channel_psd(ax, freqs, psd[:, channel_index], f"{file_path.stem} | {channel_label}", args.max_freq)
            fig.tight_layout()
        elif args.mode == "mean":
            mean_psd = np.mean(psd, axis=1)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            plot_channel_psd(ax, freqs, mean_psd, f"{file_path.stem} | Mean PSD ({n_channels} ch)", args.max_freq)
            fig.tight_layout()
        else:
            ncols = 2
            nrows = int(np.ceil(n_channels / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows), squeeze=False)
            flat_axes = axes.ravel()
            for channel_index in range(n_channels):
                channel_label = CHANNEL_NAMES[channel_index] if channel_index < len(CHANNEL_NAMES) else f"Ch{channel_index}"
                plot_channel_psd(
                    flat_axes[channel_index],
                    freqs,
                    psd[:, channel_index],
                    f"{channel_label}",
                    args.max_freq,
                )
            for extra_ax in flat_axes[n_channels:]:
                extra_ax.axis("off")
            fig.suptitle(f"{file_path.stem} | All-channel PSD | shape={n_timepoints}x{n_channels}", y=0.995)
            fig.tight_layout()

    if args.save is not None:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
