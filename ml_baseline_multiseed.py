import argparse
import json
from pathlib import Path

import numpy as np

from ml_baseline import run_baseline


METRIC_NAMES = ["accuracy", "f1", "recall", "auc", "subject_accuracy"]


def summarize(values):
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "values": [float(v) for v in arr.tolist()],
    }


def main():
    parser = argparse.ArgumentParser(description="Run ML baselines across multiple random seeds and report mean/std.")
    parser.add_argument("--data-dir", default="IEEE_subject_split", help="Directory containing train.pt / val.pt / test.pt.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 2024, 3407, 777],
        help="Random seeds to evaluate.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["val_accuracy", "test_accuracy"],
        default="val_accuracy",
        help="How to pick one model per seed before aggregation.",
    )
    parser.add_argument("--save-json", default=None, help="Optional path to save the full result table as JSON.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    per_seed = []

    for seed in args.seeds:
        results_by_val, results_by_test = run_baseline(data_dir, seed)
        selected = results_by_val[0] if args.selection_metric == "val_accuracy" else results_by_test[0]
        per_seed.append(
            {
                "seed": seed,
                "selected_model": selected["model"],
                "selection_metric": args.selection_metric,
                "val": selected["val"],
                "test": selected["test"],
                "test_subject_accuracy": selected["test_subject_accuracy"],
                "all_models_by_validation_accuracy": results_by_val,
                "all_models_by_test_accuracy": results_by_test,
            }
        )

    summary = {
        "accuracy": summarize([item["test"]["accuracy"] for item in per_seed]),
        "f1": summarize([item["test"]["f1"] for item in per_seed]),
        "recall": summarize([item["test"]["recall"] for item in per_seed]),
        "auc": summarize([item["test"]["auc"] for item in per_seed]),
        "subject_accuracy": summarize([item["test_subject_accuracy"] for item in per_seed]),
    }

    print("=== Per-seed selected result ===")
    for item in per_seed:
        print(
            f"seed={item['seed']:>4} | "
            f"model={item['selected_model']:>14} | "
            f"test_acc={item['test']['accuracy']:.3f} | "
            f"test_f1={item['test']['f1']:.3f} | "
            f"test_recall={item['test']['recall']:.3f} | "
            f"test_auc={item['test']['auc']:.3f} | "
            f"test_subject_acc={item['test_subject_accuracy']:.3f}"
        )

    print("\n=== Mean ± Std across seeds ===")
    print(f"Accuracy: {summary['accuracy']['mean']:.3f} ± {summary['accuracy']['std']:.3f}")
    print(f"F1-score: {summary['f1']['mean']:.3f} ± {summary['f1']['std']:.3f}")
    print(f"Recall: {summary['recall']['mean']:.3f} ± {summary['recall']['std']:.3f}")
    print(f"AUC: {summary['auc']['mean']:.3f} ± {summary['auc']['std']:.3f}")
    print(
        "Subject-accuracy: "
        f"{summary['subject_accuracy']['mean']:.3f} ± {summary['subject_accuracy']['std']:.3f}"
    )

    payload = {
        "data_dir": str(data_dir),
        "selection_metric": args.selection_metric,
        "seeds": args.seeds,
        "per_seed": per_seed,
        "summary": summary,
    }

    if args.save_json is not None:
        Path(args.save_json).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
