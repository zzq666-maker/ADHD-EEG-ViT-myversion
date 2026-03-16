import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


FS = 128
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def load_split(path: Path):
    data = torch.load(path, weights_only=True)
    return {
        "data": data["data"].float().numpy(),
        "label": data["label"].long().numpy(),
        "subject_id": np.array(data.get("subject_id", [])),
    }


def bandpower_features(signal: np.ndarray, fs: int = FS):
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, signal.shape[-1]), axis=-1)
    total_power = np.trapz(psd, freqs, axis=-1) + 1e-8

    features = []
    for low, high in BANDS.values():
        mask = (freqs >= low) & (freqs < high)
        band_power = np.trapz(psd[:, :, mask], freqs[mask], axis=-1)
        relative_power = band_power / total_power
        features.extend([band_power, relative_power])
    return features


def hjorth_parameters(signal: np.ndarray):
    first_diff = np.diff(signal, axis=-1)
    second_diff = np.diff(first_diff, axis=-1)

    var0 = np.var(signal, axis=-1) + 1e-8
    var1 = np.var(first_diff, axis=-1) + 1e-8
    var2 = np.var(second_diff, axis=-1) + 1e-8

    activity = var0
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / mobility
    return [activity, mobility, complexity]


def extract_features_batch(signals: np.ndarray):
    signal_mean = signals.mean(axis=-1)
    signal_std = signals.std(axis=-1)
    signal_min = signals.min(axis=-1)
    signal_max = signals.max(axis=-1)
    signal_ptp = signal_max - signal_min
    signal_rms = np.sqrt(np.mean(np.square(signals), axis=-1))

    feature_blocks = [
        signal_mean,
        signal_std,
        signal_min,
        signal_max,
        signal_ptp,
        signal_rms,
        *hjorth_parameters(signals),
        *bandpower_features(signals),
    ]
    return np.concatenate(feature_blocks, axis=1)


def evaluate_predictions(y_true, y_pred, y_prob):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }
    return metrics


def subject_level_accuracy(subject_ids, y_true, y_pred):
    if len(subject_ids) == 0:
        return None

    subject_votes = {}
    for subject_id, true_label, pred_label in zip(subject_ids, y_true, y_pred):
        bucket = subject_votes.setdefault(subject_id, {"true": true_label, "preds": []})
        bucket["preds"].append(pred_label)

    correct = 0
    total = 0
    for subject_info in subject_votes.values():
        pred_sum = sum(subject_info["preds"])
        pred_majority = int(pred_sum >= (len(subject_info["preds"]) / 2.0))
        correct += int(pred_majority == subject_info["true"])
        total += 1
    return correct / total if total else None


def build_models(seed: int):
    return {
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
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def run_baseline(data_dir: Path, seed: int):
    train_split = load_split(data_dir / "train.pt")
    val_split = load_split(data_dir / "val.pt")
    test_split = load_split(data_dir / "test.pt")

    x_train = extract_features_batch(train_split["data"])
    x_val = extract_features_batch(val_split["data"])
    x_test = extract_features_batch(test_split["data"])

    y_train = train_split["label"]
    y_val = val_split["label"]
    y_test = test_split["label"]

    models = build_models(seed)
    results = []

    for model_name, model in models.items():
        model.fit(x_train, y_train)

        val_prob = model.predict_proba(x_val)[:, 1]
        val_pred = (val_prob >= 0.5).astype(int)
        test_prob = model.predict_proba(x_test)[:, 1]
        test_pred = (test_prob >= 0.5).astype(int)

        val_metrics = evaluate_predictions(y_val, val_pred, val_prob)
        test_metrics = evaluate_predictions(y_test, test_pred, test_prob)
        test_subject_acc = subject_level_accuracy(
            test_split["subject_id"],
            y_test,
            test_pred,
        )

        result = {
            "model": model_name,
            "val": val_metrics,
            "test": test_metrics,
            "test_subject_accuracy": test_subject_acc,
        }
        results.append(result)

    results_by_val = sorted(results, key=lambda item: item["val"]["accuracy"], reverse=True)
    results_by_test = sorted(results, key=lambda item: item["test"]["accuracy"], reverse=True)
    return results_by_val, results_by_test


def main():
    parser = argparse.ArgumentParser(description="Machine-learning baseline on subject-level EEG splits.")
    parser.add_argument("--data-dir", default="IEEE_subject_split", help="Directory containing train.pt / val.pt / test.pt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for classifiers.")
    parser.add_argument("--save-json", default=None, help="Optional path to save the full result table as JSON.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_by_val, results_by_test = run_baseline(data_dir, args.seed)

    print("=== Validation ranking (fair model selection) ===")
    for item in results_by_val:
        print(
            f"{item['model']:>14} | "
            f"val_acc={item['val']['accuracy']:.3f} | "
            f"test_acc={item['test']['accuracy']:.3f} | "
            f"test_f1={item['test']['f1']:.3f} | "
            f"test_recall={item['test']['recall']:.3f} | "
            f"test_auc={item['test']['auc']:.3f} | "
            f"test_subject_acc={item['test_subject_accuracy']:.3f}"
        )

    print("\n=== Highest observed test accuracy among tried ML models ===")
    best_test = results_by_test[0]
    print(json.dumps(best_test, indent=2))

    if args.save_json is not None:
        save_path = Path(args.save_json)
        save_path.write_text(
            json.dumps(
                {
                    "seed": args.seed,
                    "results_by_validation_accuracy": results_by_val,
                    "results_by_test_accuracy": results_by_test,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
