import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import butter, detrend, filtfilt, iirnotch, welch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


FS = 128
CHANNEL_NAMES = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]
REGIONS = {
    "frontal": ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "central": ["C3", "C4", "Cz"],
    "parietal": ["P3", "P4", "P7", "P8", "Pz"],
    "occipital": ["O1", "O2"],
    "temporal": ["T7", "T8"],
}
ASYM_PAIRS = [
    ("Fp1", "Fp2"),
    ("F3", "F4"),
    ("F7", "F8"),
    ("C3", "C4"),
    ("P3", "P4"),
    ("P7", "P8"),
    ("O1", "O2"),
    ("T7", "T8"),
]
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
    label_name: str
    file_path: Path


def collect_subjects(raw_dir: Path):
    mapping = {
        "ADHD_part1": (1, "ADHD"),
        "ADHD_part2": (1, "ADHD"),
        "Control_part1": (0, "Control"),
        "Control_part2": (0, "Control"),
    }
    subjects = []
    for folder_name, (label, label_name) in mapping.items():
        for file_path in sorted((raw_dir / folder_name).glob("*.mat")):
            subjects.append(
                SubjectRecord(
                    subject_id=file_path.stem,
                    label=label,
                    label_name=label_name,
                    file_path=file_path,
                )
            )
    if not subjects:
        raise ValueError(f"No .mat subjects found under {raw_dir}")
    return subjects


def load_subject_signal(file_path: Path):
    mat = scipy.io.loadmat(str(file_path))
    if file_path.stem in mat:
        signal = mat[file_path.stem]
    else:
        valid_keys = [key for key in mat.keys() if not key.startswith("__")]
        if not valid_keys:
            raise KeyError(f"No EEG variable found in {file_path}")
        signal = mat[valid_keys[0]]
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError(f"Expected 2D EEG matrix, got shape={signal.shape} for {file_path}")
    return signal


def bandpass_filter(data: np.ndarray, fs: int = FS, lowcut: float = 0.5, highcut: float = 45.0, order: int = 4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=0)


def notch_filter(data: np.ndarray, fs: int = FS, freq: float = 50.0, q: float = 30.0):
    w0 = freq / (fs / 2.0)
    if w0 >= 1.0:
        return data
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, data, axis=0)


def preprocess_signal(signal: np.ndarray, fs: int, apply_notch_filter: bool):
    x = signal.astype(np.float32)
    x = bandpass_filter(x, fs=fs, lowcut=0.5, highcut=45.0, order=4)
    if apply_notch_filter:
        x = notch_filter(x, fs=fs, freq=50.0, q=30.0)
    x = detrend(x, axis=0, type="linear")
    x = x - np.mean(x, axis=0, keepdims=True)
    return x


def sliding_windows(signal_1d: np.ndarray, fs: int, window_sec: float, step_sec: float):
    window_size = int(fs * window_sec)
    step_size = int(fs * step_sec)
    if signal_1d.shape[0] < window_size:
        return []
    return [
        signal_1d[start : start + window_size]
        for start in range(0, signal_1d.shape[0] - window_size + 1, step_size)
    ]


def bandpower(freqs, psd, band):
    low, high = band
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def hjorth_parameters(x: np.ndarray):
    first_diff = np.diff(x)
    second_diff = np.diff(first_diff)
    var0 = np.var(x) + 1e-8
    var1 = np.var(first_diff) + 1e-8
    var2 = np.var(second_diff) + 1e-8
    activity = var0
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / max(mobility, 1e-8)
    return activity, mobility, complexity


def extract_window_features(window: np.ndarray, fs: int):
    x = np.asarray(window, dtype=np.float64)
    x = x - np.mean(x)
    nperseg = min(256, x.shape[0])
    if nperseg < 32:
        return None

    feats = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "rms": float(np.sqrt(np.mean(np.square(x)))),
        "ptp": float(np.ptp(x)),
        "skew": float(skew(x, bias=False)),
        "kurtosis": float(kurtosis(x, fisher=True, bias=False)),
        "line_length": float(np.sum(np.abs(np.diff(x)))),
        "zero_crossing_rate": float(np.mean(np.diff(np.signbit(x)).astype(np.float64))),
    }

    activity, mobility, complexity = hjorth_parameters(x)
    feats["hjorth_activity"] = float(activity)
    feats["hjorth_mobility"] = float(mobility)
    feats["hjorth_complexity"] = float(complexity)

    freqs, psd = welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )

    total_mask = (freqs >= 0.5) & (freqs <= 45.0)
    total_power = float(np.trapz(psd[total_mask], freqs[total_mask])) if np.any(total_mask) else 1e-12
    total_power = max(total_power, 1e-12)
    band_values = {}
    for band_name, band_range in BANDS.items():
        power = bandpower(freqs, psd, band_range)
        band_values[band_name] = power
        feats[f"{band_name}_abs"] = power
        feats[f"{band_name}_rel"] = power / total_power

    feats["theta_beta_ratio"] = band_values["theta"] / max(band_values["beta"], 1e-12)
    feats["theta_alpha_ratio"] = band_values["theta"] / max(band_values["alpha"], 1e-12)
    feats["beta_alpha_ratio"] = band_values["beta"] / max(band_values["alpha"], 1e-12)

    for hz in range(2, 31, 2):
        index = int(np.argmin(np.abs(freqs - hz)))
        feats[f"psd_{hz}hz"] = float(psd[index])

    return feats


def summarize_window_features(feature_list, prefix: str):
    if not feature_list:
        return {}
    summary = {}
    keys = list(feature_list[0].keys())
    for key in keys:
        values = np.array([item[key] for item in feature_list], dtype=np.float64)
        summary[f"{prefix}_{key}_mean"] = float(np.mean(values))
        summary[f"{prefix}_{key}_std"] = float(np.std(values))
    return summary


def extract_subject_features(subject: SubjectRecord, fs: int, window_sec: float, step_sec: float, apply_notch_filter: bool):
    eeg = preprocess_signal(load_subject_signal(subject.file_path), fs=fs, apply_notch_filter=apply_notch_filter)
    row = {
        "subject": subject.subject_id,
        "label": subject.label,
        "label_name": subject.label_name,
        "file_path": str(subject.file_path),
    }

    channel_cache = {}
    for ch_idx in range(eeg.shape[1]):
        channel_name = CHANNEL_NAMES[ch_idx] if ch_idx < len(CHANNEL_NAMES) else f"Ch{ch_idx+1}"
        windows = sliding_windows(eeg[:, ch_idx], fs=fs, window_sec=window_sec, step_sec=step_sec)
        feature_list = []
        for window in windows:
            feats = extract_window_features(window, fs=fs)
            if feats is not None:
                feature_list.append(feats)
        if feature_list:
            summary = summarize_window_features(feature_list, channel_name)
            row.update(summary)
            channel_cache[channel_name] = summary

    if not channel_cache:
        raise ValueError(f"No valid features extracted for subject {subject.subject_id}")

    region_metrics = [
        "theta_rel_mean",
        "alpha_rel_mean",
        "beta_rel_mean",
        "theta_beta_ratio_mean",
        "hjorth_mobility_mean",
        "hjorth_complexity_mean",
        "std_mean",
    ]
    for region_name, region_channels in REGIONS.items():
        valid = [ch for ch in region_channels if ch in channel_cache]
        if not valid:
            continue
        for metric in region_metrics:
            values = [row[f"{ch}_{metric}"] for ch in valid if f"{ch}_{metric}" in row]
            if values:
                row[f"{region_name}_{metric}"] = float(np.mean(values))

    asym_metrics = [
        "theta_rel_mean",
        "alpha_rel_mean",
        "beta_rel_mean",
        "theta_beta_ratio_mean",
        "std_mean",
        "hjorth_mobility_mean",
    ]
    for left, right in ASYM_PAIRS:
        for metric in asym_metrics:
            left_key = f"{left}_{metric}"
            right_key = f"{right}_{metric}"
            if left_key in row and right_key in row:
                row[f"{left}_{right}_{metric}_diff"] = row[left_key] - row[right_key]

    return row


def build_feature_dataframe(subjects, fs: int, window_sec: float, step_sec: float, apply_notch_filter: bool):
    rows = []
    total = len(subjects)
    for index, subject in enumerate(subjects, start=1):
        rows.append(
            extract_subject_features(
                subject,
                fs=fs,
                window_sec=window_sec,
                step_sec=step_sec,
                apply_notch_filter=apply_notch_filter,
            )
        )
        if index == 1 or index % 10 == 0 or index == total:
            print(f"Feature extraction {index}/{total}: {subject.subject_id}")
    df = pd.DataFrame(rows).fillna(0.0)
    return df


def build_models(seed: int):
    return {
        "logreg": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed)),
                ]
            ),
            {
                "clf__C": [0.1, 1.0, 10.0],
            },
        ),
        "svm_rbf": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
                ]
            ),
            {
                "clf__C": [0.5, 1.0, 2.0, 5.0],
                "clf__gamma": ["scale", 0.01, 0.001],
            },
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
            {
                "n_estimators": [300, 500],
                "max_depth": [None, 8, 16],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
    }


def get_positive_scores(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-raw_scores))
    return None


def evaluate_binary_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auc": float(roc_auc_score(y_true, y_prob)) if y_prob is not None and len(np.unique(y_true)) > 1 else None,
    }
    return metrics


def aggregate_metric_table(df: pd.DataFrame, prefix: str):
    columns = [col for col in df.columns if col.startswith(prefix)]
    output = {}
    for col in columns:
        output[f"mean_{col}"] = float(df[col].mean())
        output[f"std_{col}"] = float(df[col].std())
    return output


def categorize_feature(feature_name: str):
    if "_diff" in feature_name:
        return "asymmetry"
    if any(feature_name.startswith(f"{region}_") for region in REGIONS):
        return "region"
    if "hjorth_" in feature_name:
        return "hjorth"
    if any(token in feature_name for token in ["theta_beta_ratio", "theta_alpha_ratio", "beta_alpha_ratio"]):
        return "band_ratio"
    if "_abs_" in feature_name or "_rel_" in feature_name:
        return "frequency_band"
    if "psd_" in feature_name:
        return "psd_bin"
    if any(token in feature_name for token in ["mean_", "std_", "rms_", "ptp_", "skew_", "kurtosis_", "line_length_", "zero_crossing_rate_"]):
        return "time_domain"
    return "other"


def build_feature_catalog(feature_cols):
    rows = []
    for feature_name in feature_cols:
        rows.append(
            {
                "feature_name": feature_name,
                "feature_group": categorize_feature(feature_name),
            }
        )
    return pd.DataFrame(rows)


def compute_feature_screening_table(feature_df: pd.DataFrame, feature_cols):
    x = feature_df[feature_cols].fillna(0.0)
    y = feature_df["label"].values
    adhd_mask = feature_df["label"] == 1
    control_mask = feature_df["label"] == 0
    f_scores, p_values = f_classif(x, y)
    rows = []
    for idx, feature_name in enumerate(feature_cols):
        adhd_mean = float(x.loc[adhd_mask, feature_name].mean())
        control_mean = float(x.loc[control_mask, feature_name].mean())
        effect = adhd_mean - control_mean
        rows.append(
            {
                "feature_name": feature_name,
                "feature_group": categorize_feature(feature_name),
                "f_score": float(f_scores[idx]) if np.isfinite(f_scores[idx]) else 0.0,
                "p_value": float(p_values[idx]) if np.isfinite(p_values[idx]) else 1.0,
                "adhd_mean": adhd_mean,
                "control_mean": control_mean,
                "mean_difference": effect,
                "abs_mean_difference": abs(effect),
            }
        )
    screening_df = pd.DataFrame(rows).sort_values(
        by=["f_score", "abs_mean_difference"],
        ascending=[False, False],
    )
    return screening_df


def compute_tsne_embedding(x: np.ndarray, seed: int, perplexity: float):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    n_components = min(50, x_scaled.shape[1], max(2, x_scaled.shape[0] - 1))
    if x_scaled.shape[1] > n_components:
        x_for_tsne = PCA(n_components=n_components, random_state=seed).fit_transform(x_scaled)
    else:
        x_for_tsne = x_scaled

    effective_perplexity = min(perplexity, max(5.0, x.shape[0] - 1.0))
    if effective_perplexity >= x.shape[0]:
        effective_perplexity = max(2.0, x.shape[0] / 3.0)

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    return tsne.fit_transform(x_for_tsne)


def save_tsne_plot(tsne_df: pd.DataFrame, title: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    label_styles = {
        "ADHD": {"color": "#d55e00", "marker": "o"},
        "Control": {"color": "#0072b2", "marker": "s"},
    }
    for label_name, style in label_styles.items():
        subset = tsne_df[tsne_df["label_name"] == label_name]
        if subset.empty:
            continue
        ax.scatter(
            subset["tsne_1"],
            subset["tsne_2"],
            label=label_name,
            c=style["color"],
            marker=style["marker"],
            s=55,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.4,
        )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_tsne_analysis(feature_df: pd.DataFrame, feature_cols, output_dir: Path, seed: int, perplexity: float):
    tsne_dir = output_dir / "tsne_analysis"
    tsne_dir.mkdir(parents=True, exist_ok=True)

    feature_catalog = build_feature_catalog(feature_cols)
    feature_catalog.to_csv(tsne_dir / "feature_catalog.csv", index=False, encoding="utf-8-sig")

    screening_df = compute_feature_screening_table(feature_df, feature_cols)
    screening_df.to_csv(tsne_dir / "feature_screening_scores.csv", index=False, encoding="utf-8-sig")
    screening_df.head(100).to_csv(tsne_dir / "feature_screening_top100.csv", index=False, encoding="utf-8-sig")

    base_info = feature_df[["subject", "label", "label_name", "file_path"]].copy()

    print("Running t-SNE on all features...")
    all_embedding = compute_tsne_embedding(feature_df[feature_cols].fillna(0.0).values, seed=seed, perplexity=perplexity)
    all_tsne_df = base_info.copy()
    all_tsne_df["tsne_1"] = all_embedding[:, 0]
    all_tsne_df["tsne_2"] = all_embedding[:, 1]
    all_tsne_df.to_csv(tsne_dir / "tsne_all_features.csv", index=False, encoding="utf-8-sig")
    save_tsne_plot(all_tsne_df, "t-SNE of All Features", tsne_dir / "tsne_all_features.png")

    for feature_group in sorted(feature_catalog["feature_group"].unique()):
        group_features = feature_catalog.loc[feature_catalog["feature_group"] == feature_group, "feature_name"].tolist()
        if len(group_features) < 2:
            continue
        print(f"Running t-SNE for feature group: {feature_group} ({len(group_features)} features)")
        group_embedding = compute_tsne_embedding(
            feature_df[group_features].fillna(0.0).values,
            seed=seed,
            perplexity=perplexity,
        )
        group_tsne_df = base_info.copy()
        group_tsne_df["tsne_1"] = group_embedding[:, 0]
        group_tsne_df["tsne_2"] = group_embedding[:, 1]
        group_tsne_df.to_csv(tsne_dir / f"tsne_{feature_group}.csv", index=False, encoding="utf-8-sig")
        save_tsne_plot(
            group_tsne_df,
            f"t-SNE of {feature_group} Features",
            tsne_dir / f"tsne_{feature_group}.png",
        )

    group_counts = feature_catalog["feature_group"].value_counts().to_dict()
    summary = {
        "n_subjects": int(feature_df.shape[0]),
        "n_features": int(len(feature_cols)),
        "tsne_perplexity": float(min(perplexity, max(5.0, feature_df.shape[0] - 1.0))),
        "feature_group_counts": group_counts,
    }
    with open(tsne_dir / "tsne_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Standard EEG ML baseline with subject-level train/val/test evaluation.")
    parser.add_argument("--raw-dir", default="data_raw", help="Directory containing ADHD_part*/Control_part* .mat files.")
    parser.add_argument("--output-dir", default="ml_standard_results", help="Directory to store features and evaluation outputs.")
    parser.add_argument("--fs", type=int, default=FS, help="Sampling rate.")
    parser.add_argument("--window-sec", type=float, default=4.0, help="Sliding window size in seconds.")
    parser.add_argument("--step-sec", type=float, default=2.0, help="Sliding window step in seconds.")
    parser.add_argument("--repeats", type=int, default=10, help="Number of repeated subject-level experiments.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Independent test ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Final overall validation ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--no-notch", action="store_true", help="Disable 50 Hz notch filter.")
    parser.add_argument("--run-tsne", action="store_true", help="Export t-SNE plots and feature screening tables for manual feature selection.")
    parser.add_argument("--tsne-only", action="store_true", help="Only export feature tables and t-SNE outputs, skip classifier training.")
    parser.add_argument("--tsne-perplexity", type=float, default=20.0, help="Perplexity used by t-SNE analysis.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = collect_subjects(raw_dir)
    feature_df = build_feature_dataframe(
        subjects,
        fs=args.fs,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        apply_notch_filter=not args.no_notch,
    )
    feature_df.to_csv(output_dir / "all_subject_features.csv", index=False, encoding="utf-8-sig")

    exclude_cols = ["subject", "label", "label_name", "file_path"]
    feature_cols = [col for col in feature_df.columns if col not in exclude_cols]

    tsne_summary = None
    if args.run_tsne:
        tsne_summary = export_tsne_analysis(
            feature_df,
            feature_cols,
            output_dir=output_dir,
            seed=args.seed,
            perplexity=args.tsne_perplexity,
        )
        print("t-SNE analysis exported to:", output_dir / "tsne_analysis")
        if args.tsne_only:
            metadata = {
                "raw_dir": str(raw_dir),
                "output_dir": str(output_dir),
                "fs": args.fs,
                "window_sec": args.window_sec,
                "step_sec": args.step_sec,
                "run_tsne": True,
                "tsne_only": True,
                "tsne_summary": tsne_summary,
            }
            with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print("t-SNE only mode finished. Skipping classifier training.")
            return

    all_model_results = []
    selected_results = []
    best_selected = None

    val_ratio_within_trainval = args.val_ratio / (1.0 - args.test_ratio)

    for repeat_idx in range(args.repeats):
        seed = args.seed + repeat_idx
        train_val_df, test_df = train_test_split(
            feature_df,
            test_size=args.test_ratio,
            stratify=feature_df["label"],
            random_state=seed,
        )
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_within_trainval,
            stratify=train_val_df["label"],
            random_state=seed,
        )

        x_train = train_df[feature_cols].fillna(0.0)
        y_train = train_df["label"].values
        x_val = val_df[feature_cols].fillna(0.0)
        y_val = val_df["label"].values
        x_test = test_df[feature_cols].fillna(0.0)
        y_test = test_df["label"].values

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        repeat_best = None

        for model_name, (model, param_grid) in build_models(seed).items():
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring="balanced_accuracy",
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            search.fit(x_train, y_train)
            best_model = search.best_estimator_

            y_val_pred = best_model.predict(x_val)
            y_test_pred = best_model.predict(x_test)
            y_val_prob = get_positive_scores(best_model, x_val)
            y_test_prob = get_positive_scores(best_model, x_test)
            val_metrics = evaluate_binary_metrics(y_val, y_val_pred, y_val_prob)
            test_metrics = evaluate_binary_metrics(y_test, y_test_pred, y_test_prob)

            result = {
                "repeat": repeat_idx + 1,
                "seed": seed,
                "model_name": model_name,
                "cv_balanced_accuracy": float(search.best_score_),
                "best_params": search.best_params_,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            all_model_results.append(result)

            if repeat_best is None or result["val_balanced_accuracy"] > repeat_best["val_balanced_accuracy"]:
                repeat_best = {
                    **result,
                    "best_estimator": best_model,
                }

        selected = {k: v for k, v in repeat_best.items() if k != "best_estimator"}
        selected_results.append(selected)
        if best_selected is None or repeat_best["val_balanced_accuracy"] > best_selected["val_balanced_accuracy"]:
            best_selected = repeat_best

        print(
            f"Repeat {repeat_idx + 1}/{args.repeats} | "
            f"selected={repeat_best['model_name']} | "
            f"val_bal_acc={repeat_best['val_balanced_accuracy']:.4f} | "
            f"test_bal_acc={repeat_best['test_balanced_accuracy']:.4f}"
        )

    all_results_df = pd.DataFrame(all_model_results)
    selected_df = pd.DataFrame(selected_results)
    model_summary_df = (
        all_results_df.groupby("model_name")
        .agg(
            mean_val_balanced_accuracy=("val_balanced_accuracy", "mean"),
            std_val_balanced_accuracy=("val_balanced_accuracy", "std"),
            mean_test_balanced_accuracy=("test_balanced_accuracy", "mean"),
            std_test_balanced_accuracy=("test_balanced_accuracy", "std"),
            mean_test_accuracy=("test_accuracy", "mean"),
            std_test_accuracy=("test_accuracy", "std"),
            mean_test_f1=("test_f1", "mean"),
            std_test_f1=("test_f1", "std"),
            mean_test_auc=("test_auc", "mean"),
            std_test_auc=("test_auc", "std"),
        )
        .reset_index()
    )

    selected_summary = {
        "repeats": int(args.repeats),
        "selection_metric": "validation balanced accuracy",
        "selected_model_counts": selected_df["model_name"].value_counts().to_dict(),
        **aggregate_metric_table(selected_df, "test_"),
    }

    all_results_df.to_csv(output_dir / "all_model_results.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(output_dir / "selected_model_results.csv", index=False, encoding="utf-8-sig")
    model_summary_df.to_csv(output_dir / "model_summary.csv", index=False, encoding="utf-8-sig")
    joblib.dump(best_selected["best_estimator"], output_dir / "best_selected_model.pkl")

    with open(output_dir / "selected_summary.json", "w", encoding="utf-8") as f:
        json.dump(selected_summary, f, ensure_ascii=False, indent=2)

    metadata = {
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "fs": args.fs,
        "window_sec": args.window_sec,
        "step_sec": args.step_sec,
        "repeats": args.repeats,
        "test_ratio": args.test_ratio,
        "val_ratio": args.val_ratio,
        "notch_filter": not args.no_notch,
        "run_tsne": bool(args.run_tsne),
        "feature_groups": [
            "time_domain_statistics",
            "hjorth_parameters",
            "band_power_and_relative_power",
            "fine_grained_psd_bins",
            "region_averages",
            "left_right_asymmetry",
        ],
        "tsne_summary": tsne_summary,
        "selection_rule": "Within each repeat, choose the model with the highest validation balanced accuracy and report test performance.",
        "best_selected_model": {
            "repeat": int(best_selected["repeat"]),
            "model_name": best_selected["model_name"],
            "val_balanced_accuracy": float(best_selected["val_balanced_accuracy"]),
            "test_balanced_accuracy": float(best_selected["test_balanced_accuracy"]),
            "best_params": best_selected["best_params"],
        },
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n=== Selected-model generalized performance ===")
    print(json.dumps(selected_summary, ensure_ascii=False, indent=2))
    print("\nResults saved to:", output_dir)


if __name__ == "__main__":
    main()
