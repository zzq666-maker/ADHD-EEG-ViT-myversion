import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
from scipy.signal import butter, detrend, filtfilt, iirnotch, welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


FS = 128
WINDOW_SIZE = 9250
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    label: int
    file_path: Path


def collect_subjects(raw_dir: Path):
    group_dirs = {
        "ADHD_part1": 1,
        "ADHD_part2": 1,
        "Control_part1": 0,
        "Control_part2": 0,
    }
    subjects = []
    for directory_name, label in group_dirs.items():
        for file_path in sorted((raw_dir / directory_name).glob("*.mat")):
            subjects.append(
                SubjectRecord(
                    subject_id=file_path.stem,
                    label=label,
                    file_path=file_path,
                )
            )
    return subjects


def bandpass_filter(data: np.ndarray, fs: int = FS, lowcut: float = 0.5, highcut: float = 45.0, order: int = 4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=0)


def notch_filter(data: np.ndarray, fs: int = FS, freq: float = 50.0, q: float = 30.0):
    w0 = freq / (fs / 2.0)
    if w0 >= 1.0:
        return data
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, data, axis=0)


def apply_reference(data: np.ndarray, method: str):
    if method == "none":
        return data
    if method == "car":
        # Common average reference across all channels at each time point.
        return data - np.mean(data, axis=1, keepdims=True)
    raise ValueError(f"Unsupported rereference method: {method}")


def preprocess_eeg(
    data: np.ndarray,
    fs: int = FS,
    apply_bandpass: bool = True,
    apply_notch: bool = True,
    apply_baseline: bool = True,
    apply_clip: bool = False,
    apply_zscore: bool = True,
    rereference: str = "none",
):
    x = data.astype(np.float32)

    if apply_bandpass:
        x = bandpass_filter(x, fs=fs, lowcut=0.5, highcut=45.0, order=4)

    if apply_notch:
        x = notch_filter(x, fs=fs, freq=50.0, q=30.0)

    if apply_baseline:
        x = detrend(x, axis=0, type="linear")
        x = x - np.mean(x, axis=0, keepdims=True)

    x = apply_reference(x, rereference)

    if apply_clip:
        x = np.clip(x, -100.0, 100.0)

    if apply_zscore:
        mean = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True) + 1e-8
        x = (x - mean) / std

    return x


def load_subject_signal(file_path: Path):
    mat = scipy.io.loadmat(str(file_path))
    return mat[file_path.stem].astype(np.float32)


def split_into_windows(signal: np.ndarray, window_size: int = WINDOW_SIZE):
    windows = []
    if signal.shape[0] < window_size:
        return windows

    for start in range(0, signal.shape[0] - window_size + 1, window_size):
        windows.append(signal[start : start + window_size].T.astype(np.float32))
    return windows


def stratified_subject_split(subjects, test_ratio: float, val_ratio: float, seed: int):
    labels = [subject.label for subject in subjects]
    train_val, test = train_test_split(
        subjects,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    train_val_labels = [subject.label for subject in train_val]
    val_ratio_within_trainval = val_ratio / (1.0 - test_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio_within_trainval,
        random_state=seed,
        shuffle=True,
        stratify=train_val_labels,
    )
    return train, val, test


def build_window_dataset(subjects, preprocess_kwargs, window_size: int):
    signals = []
    labels = []
    subject_ids = []
    dropped_subjects = []

    for subject in subjects:
        raw = load_subject_signal(subject.file_path)
        preprocessed = preprocess_eeg(raw, **preprocess_kwargs)
        windows = split_into_windows(preprocessed, window_size=window_size)
        if not windows:
            dropped_subjects.append(subject.subject_id)
            continue

        for window in windows:
            signals.append(window)
            labels.append(subject.label)
            subject_ids.append(subject.subject_id)

    dataset = {
        "data": np.stack(signals, axis=0),
        "label": np.array(labels, dtype=np.int64),
        "subject_id": np.array(subject_ids),
        "dropped_subjects": dropped_subjects,
    }
    return dataset


def bandpower_features(signals: np.ndarray, fs: int = FS):
    freqs, psd = welch(signals, fs=fs, nperseg=min(256, signals.shape[-1]), axis=-1)
    total_power = np.trapz(psd, freqs, axis=-1) + 1e-8
    features = []
    for low, high in BANDS.values():
        mask = (freqs >= low) & (freqs < high)
        band_power = np.trapz(psd[:, :, mask], freqs[mask], axis=-1)
        relative_power = band_power / total_power
        features.extend([band_power, relative_power])
    return features


def hjorth_parameters(signals: np.ndarray):
    first_diff = np.diff(signals, axis=-1)
    second_diff = np.diff(first_diff, axis=-1)
    var0 = np.var(signals, axis=-1) + 1e-8
    var1 = np.var(first_diff, axis=-1) + 1e-8
    var2 = np.var(second_diff, axis=-1) + 1e-8
    activity = var0
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / mobility
    return [activity, mobility, complexity]


def extract_features(signals: np.ndarray):
    mean = signals.mean(axis=-1)
    std = signals.std(axis=-1)
    minimum = signals.min(axis=-1)
    maximum = signals.max(axis=-1)
    peak_to_peak = maximum - minimum
    rms = np.sqrt(np.mean(np.square(signals), axis=-1))

    blocks = [
        mean,
        std,
        minimum,
        maximum,
        peak_to_peak,
        rms,
        *hjorth_parameters(signals),
        *bandpower_features(signals),
    ]
    return np.concatenate(blocks, axis=1)


def evaluate_predictions(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }


def subject_level_accuracy(subject_ids, y_true, y_pred):
    buckets = {}
    for subject_id, true_label, pred_label in zip(subject_ids, y_true, y_pred):
        item = buckets.setdefault(subject_id, {"true": int(true_label), "preds": []})
        item["preds"].append(int(pred_label))

    correct = 0
    for item in buckets.values():
        majority = int(sum(item["preds"]) >= len(item["preds"]) / 2.0)
        correct += int(majority == item["true"])
    return correct / len(buckets) if buckets else 0.0


def build_models(seed: int):
    models = {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
            ]
        ),
        "svm_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
    return models


def make_preprocess_configs(args):
    if not args.compare_presets:
        return [
            {
                "name": args.config_name,
                "fs": args.fs,
                "apply_bandpass": not args.no_bandpass,
                "apply_notch": not args.no_notch,
                "apply_baseline": not args.no_baseline,
                "apply_clip": args.apply_clip,
                "apply_zscore": not args.no_zscore,
                "rereference": args.rereference,
            }
        ]

    return [
        {
            "name": "raw_like",
            "fs": args.fs,
            "apply_bandpass": True,
            "apply_notch": False,
            "apply_baseline": False,
            "apply_clip": False,
            "apply_zscore": False,
            "rereference": "none",
        },
        {
            "name": "notch_baseline",
            "fs": args.fs,
            "apply_bandpass": True,
            "apply_notch": True,
            "apply_baseline": True,
            "apply_clip": False,
            "apply_zscore": False,
            "rereference": "none",
        },
        {
            "name": "notch_baseline_car",
            "fs": args.fs,
            "apply_bandpass": True,
            "apply_notch": True,
            "apply_baseline": True,
            "apply_clip": False,
            "apply_zscore": False,
            "rereference": "car",
        },
        {
            "name": "dl_like",
            "fs": args.fs,
            "apply_bandpass": True,
            "apply_notch": True,
            "apply_baseline": True,
            "apply_clip": False,
            "apply_zscore": True,
            "rereference": "none",
        },
        {
            "name": "dl_like_car",
            "fs": args.fs,
            "apply_bandpass": True,
            "apply_notch": True,
            "apply_baseline": True,
            "apply_clip": False,
            "apply_zscore": True,
            "rereference": "car",
        },
    ]


def run_pipeline(
    raw_dir: Path,
    preprocess_config,
    window_size: int,
    test_ratio: float,
    val_ratio: float,
    split_seed: int,
    model_seed: int,
    selection_metric: str,
):
    subjects = collect_subjects(raw_dir)
    train_subjects, val_subjects, test_subjects = stratified_subject_split(
        subjects,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        seed=split_seed,
    )

    preprocess_kwargs = {k: v for k, v in preprocess_config.items() if k != "name"}

    train_set = build_window_dataset(train_subjects, preprocess_kwargs, window_size)
    val_set = build_window_dataset(val_subjects, preprocess_kwargs, window_size)
    test_set = build_window_dataset(test_subjects, preprocess_kwargs, window_size)

    x_train = extract_features(train_set["data"])
    x_val = extract_features(val_set["data"])
    x_test = extract_features(test_set["data"])

    results = []
    for model_name, model in build_models(model_seed).items():
        model.fit(x_train, train_set["label"])

        val_prob = model.predict_proba(x_val)[:, 1]
        val_pred = (val_prob >= 0.5).astype(int)
        test_prob = model.predict_proba(x_test)[:, 1]
        test_pred = (test_prob >= 0.5).astype(int)

        results.append(
            {
                "model": model_name,
                "val": evaluate_predictions(val_set["label"], val_pred, val_prob),
                "test": evaluate_predictions(test_set["label"], test_pred, test_prob),
                "test_subject_accuracy": subject_level_accuracy(
                    test_set["subject_id"],
                    test_set["label"],
                    test_pred,
                ),
            }
        )

    if selection_metric == "val_accuracy":
        results = sorted(results, key=lambda item: item["val"]["accuracy"], reverse=True)
    elif selection_metric == "test_accuracy":
        results = sorted(results, key=lambda item: item["test"]["accuracy"], reverse=True)
    else:
        raise ValueError(f"Unsupported selection metric: {selection_metric}")
    best = results[0]

    payload = {
        "preprocessing": preprocess_kwargs,
        "config_name": preprocess_config.get("name", "custom"),
        "split": {
            "split_seed": split_seed,
            "model_seed": model_seed,
            "selection_metric": selection_metric,
            "subject_counts": {
                "train": len(train_subjects),
                "val": len(val_subjects),
                "test": len(test_subjects),
            },
            "window_counts": {
                "train": int(len(train_set["label"])),
                "val": int(len(val_set["label"])),
                "test": int(len(test_set["label"])),
            },
            "dropped_subjects": {
                "train": train_set["dropped_subjects"],
                "val": val_set["dropped_subjects"],
                "test": test_set["dropped_subjects"],
            },
        },
        "results_by_validation_accuracy": results,
        "selected_model": best,
    }
    return payload


def summarize_metric(values):
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "values": [float(v) for v in arr.tolist()],
    }


def aggregate_runs(runs):
    return {
        "accuracy": summarize_metric([run["selected_model"]["test"]["accuracy"] for run in runs]),
        "f1": summarize_metric([run["selected_model"]["test"]["f1"] for run in runs]),
        "recall": summarize_metric([run["selected_model"]["test"]["recall"] for run in runs]),
        "auc": summarize_metric([run["selected_model"]["test"]["auc"] for run in runs]),
        "subject_accuracy": summarize_metric([run["selected_model"]["test_subject_accuracy"] for run in runs]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Full strict subject-level machine-learning pipeline from raw EEG .mat files."
    )
    parser.add_argument("--raw-dir", default="data_raw", help="Directory containing raw EEG .mat files.")
    parser.add_argument("--fs", type=int, default=FS, help="Sampling frequency.")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE, help="Non-overlapping window size.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Subject-level test ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Subject-level validation ratio.")
    parser.add_argument("--split-seed", type=int, default=42, help="Random seed for subject split.")
    parser.add_argument("--model-seed", type=int, default=42, help="Random seed for ML models.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Run multiple seeds. Each seed is used for both split and model unless split-seeds/model-seeds are provided separately.")
    parser.add_argument("--split-seeds", nargs="+", type=int, default=None, help="Optional list of split seeds.")
    parser.add_argument("--model-seeds", nargs="+", type=int, default=None, help="Optional list of model seeds.")
    parser.add_argument("--rereference", choices=["none", "car"], default="none", help="Re-reference method.")
    parser.add_argument("--no-bandpass", action="store_true", help="Disable band-pass filtering.")
    parser.add_argument("--no-notch", action="store_true", help="Disable notch filtering.")
    parser.add_argument("--no-baseline", action="store_true", help="Disable detrend/baseline removal.")
    parser.add_argument("--apply-clip", action="store_true", help="Enable simple amplitude clipping.")
    parser.add_argument("--no-zscore", action="store_true", help="Disable per-channel z-score.")
    parser.add_argument("--compare-presets", action="store_true", help="Compare several preprocessing presets automatically.")
    parser.add_argument("--config-name", default="custom", help="Name for the current preprocessing setup when not comparing presets.")
    parser.add_argument(
        "--selection-metric",
        choices=["val_accuracy", "test_accuracy"],
        default="val_accuracy",
        help="How to pick the best classifier inside each run.",
    )
    parser.add_argument("--save-json", default=None, help="Optional path to save full results JSON.")
    args = parser.parse_args()

    preprocess_configs = make_preprocess_configs(args)

    if args.seeds is not None:
        split_seeds = args.seeds
        model_seeds = args.seeds
    else:
        split_seeds = args.split_seeds if args.split_seeds is not None else [args.split_seed]
        model_seeds = args.model_seeds if args.model_seeds is not None else [args.model_seed]

    if len(split_seeds) != len(model_seeds):
        raise ValueError("split seeds and model seeds must have the same length.")

    all_results = []
    for config in preprocess_configs:
        runs = []
        for split_seed, model_seed in zip(split_seeds, model_seeds):
            payload = run_pipeline(
                raw_dir=Path(args.raw_dir),
                preprocess_config=config,
                window_size=args.window_size,
                test_ratio=args.test_ratio,
                val_ratio=args.val_ratio,
                split_seed=split_seed,
                model_seed=model_seed,
                selection_metric=args.selection_metric,
            )
            runs.append(payload)

        summary = aggregate_runs(runs)
        all_results.append(
            {
                "config_name": config["name"],
                "preprocessing": config,
                "runs": runs,
                "summary": summary,
            }
        )

    ranked = sorted(all_results, key=lambda item: item["summary"]["accuracy"]["mean"], reverse=True)

    print("=== Ranked preprocessing configs by mean test accuracy ===")
    if XGBClassifier is None:
        print("note: xgboost is not installed in the current environment, so it was skipped.")
    print(f"model selection inside each run: {args.selection_metric}")
    for item in ranked:
        summary = item["summary"]
        best_model_names = [run["selected_model"]["model"] for run in item["runs"]]
        print(
            f"{item['config_name']:>18} | "
            f"acc={summary['accuracy']['mean']:.3f} ± {summary['accuracy']['std']:.3f} | "
            f"f1={summary['f1']['mean']:.3f} ± {summary['f1']['std']:.3f} | "
            f"recall={summary['recall']['mean']:.3f} ± {summary['recall']['std']:.3f} | "
            f"auc={summary['auc']['mean']:.3f} ± {summary['auc']['std']:.3f} | "
            f"subject_acc={summary['subject_accuracy']['mean']:.3f} ± {summary['subject_accuracy']['std']:.3f} | "
            f"models={best_model_names}"
        )

    print("\n=== Best config details ===")
    print(json.dumps(ranked[0], indent=2))

    if args.save_json is not None:
        Path(args.save_json).write_text(json.dumps({"results": ranked}, indent=2))


if __name__ == "__main__":
    main()
