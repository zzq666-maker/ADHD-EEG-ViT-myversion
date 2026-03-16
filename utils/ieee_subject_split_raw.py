import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
import torch
from sklearn.model_selection import train_test_split


WINDOW_SIZE = 9250


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    label: int
    file_path: Path


def load_subject_signal(file_path: Path) -> np.ndarray:
    mat = scipy.io.loadmat(str(file_path))
    signal = mat[file_path.stem]
    return signal.astype(np.float32)


def split_into_windows(signal: np.ndarray, window_size: int = WINDOW_SIZE):
    if signal.shape[0] < window_size:
        return []

    windows = []
    for start in range(0, signal.shape[0] - window_size + 1, window_size):
        window = signal[start : start + window_size]
        windows.append(window.T.astype(np.float32))
    return windows


def collect_subjects(raw_dir: Path):
    group_dirs = {
        "ADHD_part1": 1,
        "ADHD_part2": 1,
        "Control_part1": 0,
        "Control_part2": 0,
    }
    subjects = []
    for directory_name, label in group_dirs.items():
        directory = raw_dir / directory_name
        for file_path in sorted(directory.glob("*.mat")):
            subject_id = file_path.stem
            subjects.append(SubjectRecord(subject_id=subject_id, label=label, file_path=file_path))
    return subjects


def stratified_subject_split(subjects, test_ratio: float, val_ratio: float, seed: int):
    labels = [subject.label for subject in subjects]

    train_val_subjects, test_subjects = train_test_split(
        subjects,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )

    train_val_labels = [subject.label for subject in train_val_subjects]
    val_ratio_in_trainval = val_ratio / (1.0 - test_ratio)
    train_subjects, val_subjects = train_test_split(
        train_val_subjects,
        test_size=val_ratio_in_trainval,
        random_state=seed,
        shuffle=True,
        stratify=train_val_labels,
    )
    return train_subjects, val_subjects, test_subjects


def build_split(subjects, split_name: str, window_size: int):
    data = []
    labels = []
    subject_ids = []
    source_files = []
    dropped_subjects = []

    for subject in subjects:
        windows = split_into_windows(load_subject_signal(subject.file_path), window_size=window_size)
        if not windows:
            dropped_subjects.append(subject.subject_id)
            continue

        for window in windows:
            data.append(torch.from_numpy(window))
            labels.append(subject.label)
            subject_ids.append(subject.subject_id)
            source_files.append(subject.file_path.name)

    if not data:
        raise ValueError(f"No windows were generated for split '{split_name}'.")

    dataset = {
        "data": torch.stack(data, dim=0),
        "label": torch.tensor(labels, dtype=torch.long),
        "subject_id": subject_ids,
        "source_file": source_files,
        "split": split_name,
    }
    return dataset, dropped_subjects


def save_split(dataset: dict, output_path: Path):
    torch.save(dataset, output_path)


def summarize_split(subjects, dataset, dropped_subjects):
    unique_subjects = sorted(set(dataset["subject_id"]))
    label_tensor = dataset["label"]
    return {
        "subjects": len(subjects),
        "subjects_used": len(unique_subjects),
        "subjects_dropped_short_signal": dropped_subjects,
        "windows": int(dataset["data"].shape[0]),
        "adhd_windows": int((label_tensor == 1).sum().item()),
        "control_windows": int((label_tensor == 0).sum().item()),
        "shape": list(dataset["data"].shape),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create strict subject-level IEEE ADHD EEG splits without any preprocessing."
    )
    parser.add_argument("--raw-dir", default="data_raw", help="Directory containing raw IEEE .mat files.")
    parser.add_argument(
        "--output-dir",
        default="IEEE_subject_split_raw",
        help="Directory to store subject-level train/val/test .pt files.",
    )
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE, help="Non-overlapping EEG window size.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Subject-level test ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Subject-level validation ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = collect_subjects(raw_dir)
    train_subjects, val_subjects, test_subjects = stratified_subject_split(
        subjects,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_set, train_dropped = build_split(train_subjects, "train", args.window_size)
    val_set, val_dropped = build_split(val_subjects, "val", args.window_size)
    test_set, test_dropped = build_split(test_subjects, "test", args.window_size)

    save_split(train_set, output_dir / "train.pt")
    save_split(val_set, output_dir / "val.pt")
    save_split(test_set, output_dir / "test.pt")

    metadata = {
        "name": "IEEE subject-level raw EEG data for ADHD / Control children",
        "description": "Subject-level split performed before windowing. No preprocessing is applied.",
        "license": "CC BY 4.0",
        "window_size": args.window_size,
        "seed": args.seed,
        "split_strategy": "subject_level_stratified_split_then_windowing",
        "preprocessing": "none",
        "train": summarize_split(train_subjects, train_set, train_dropped),
        "val": summarize_split(val_subjects, val_set, val_dropped),
        "test": summarize_split(test_subjects, test_set, test_dropped),
        "subject_counts": {
            "total": len(subjects),
            "adhd": sum(subject.label == 1 for subject in subjects),
            "control": sum(subject.label == 0 for subject in subjects),
        },
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
