import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.transformer import ViTransformer
from models.vit_config import get_default_vit_config
from utils import EEGDataset, WarmupScheduler, device, evaluate, fix_random_seed, train


def build_candidates(preset: str = "coarse"):
    base = get_default_vit_config()
    coarse = [
        {
            "name": "small_reg_tf32",
            "train": {
                "epochs": 20,
                "lr": 3e-4,
                "weight_decay": 5e-3,
                "patience": 6,
                "batch_size": 8,
                "grad_step": 2,
                "warmup_steps": 10,
                "lr_decay_factor": 0.5,
            },
            "model": {
                **deepcopy(base),
                "embed_dim": 32,
                "num_blocks": 1,
                "block_hidden_dim": 64,
                "fc_hidden_dim": 16,
                "dropout_p": 0.4,
                "shallow_hidden_dim": 64,
                "shallow_dropout_p": 0.2,
                "tf_dropout_p": 0.5,
                "stft_n_fft": 32,
                "stft_hop_length": 16,
            },
        },
        {
            "name": "small_reg_tf64",
            "train": {
                "epochs": 20,
                "lr": 3e-4,
                "weight_decay": 5e-3,
                "patience": 6,
                "batch_size": 8,
                "grad_step": 2,
                "warmup_steps": 10,
                "lr_decay_factor": 0.5,
            },
            "model": {
                **deepcopy(base),
                "embed_dim": 32,
                "num_blocks": 1,
                "block_hidden_dim": 64,
                "fc_hidden_dim": 16,
                "dropout_p": 0.4,
                "shallow_hidden_dim": 64,
                "shallow_dropout_p": 0.2,
                "tf_dropout_p": 0.4,
                "stft_n_fft": 64,
                "stft_hop_length": 32,
            },
        },
        {
            "name": "mid_reg_tf32",
            "train": {
                "epochs": 20,
                "lr": 3e-4,
                "weight_decay": 5e-3,
                "patience": 6,
                "batch_size": 8,
                "grad_step": 2,
                "warmup_steps": 10,
                "lr_decay_factor": 0.5,
            },
            "model": {
                **deepcopy(base),
                "embed_dim": 48,
                "num_blocks": 2,
                "block_hidden_dim": 96,
                "fc_hidden_dim": 24,
                "dropout_p": 0.4,
                "shallow_hidden_dim": 96,
                "shallow_dropout_p": 0.2,
                "tf_dropout_p": 0.5,
                "stft_n_fft": 32,
                "stft_hop_length": 16,
            },
        },
        {
            "name": "small_stronger_reg",
            "train": {
                "epochs": 24,
                "lr": 2e-4,
                "weight_decay": 1e-2,
                "patience": 8,
                "batch_size": 8,
                "grad_step": 2,
                "warmup_steps": 10,
                "lr_decay_factor": 0.5,
            },
            "model": {
                **deepcopy(base),
                "embed_dim": 32,
                "num_blocks": 1,
                "block_hidden_dim": 64,
                "fc_hidden_dim": 16,
                "dropout_p": 0.5,
                "shallow_hidden_dim": 64,
                "shallow_dropout_p": 0.3,
                "tf_dropout_p": 0.6,
                "stft_n_fft": 32,
                "stft_hop_length": 16,
            },
        },
    ]

    fine = [
        {
            "name": "fine_base",
            "train": {
                "epochs": 12,
                "lr": 3e-4,
                "weight_decay": 5e-3,
                "patience": 4,
                "batch_size": 8,
                "grad_step": 2,
                "warmup_steps": 10,
                "lr_decay_factor": 0.5,
            },
            "model": {**deepcopy(base)},
        },
        {
            "name": "fine_tf06",
            "train": {
                "epochs": 12,
                "lr": 3e-4,
                "weight_decay": 5e-3,
                "patience": 4,
                "batch_size": 8,
                "grad_step": 2,
                "warmup_steps": 10,
                "lr_decay_factor": 0.5,
            },
            "model": {**deepcopy(base), "tf_dropout_p": 0.6},
        },
        {
            "name": "fine_lr2e4",
            "train": {
                "epochs": 12,
                "lr": 2e-4,
                "weight_decay": 5e-3,
                "patience": 4,
                "batch_size": 8,
                "grad_step": 2,
                "warmup_steps": 10,
                "lr_decay_factor": 0.5,
            },
            "model": {**deepcopy(base)},
        },
        {
            "name": "fine_drop045",
            "train": {
                "epochs": 12,
                "lr": 3e-4,
                "weight_decay": 5e-3,
                "patience": 4,
                "batch_size": 8,
                "grad_step": 2,
                "warmup_steps": 10,
                "lr_decay_factor": 0.5,
            },
            "model": {**deepcopy(base), "dropout_p": 0.45},
        },
        {
            "name": "fine_tf04",
            "train": {
                "epochs": 12,
                "lr": 3e-4,
                "weight_decay": 5e-3,
                "patience": 4,
                "batch_size": 8,
                "grad_step": 2,
                "warmup_steps": 10,
                "lr_decay_factor": 0.5,
            },
            "model": {**deepcopy(base), "tf_dropout_p": 0.4},
        },
        {
            "name": "fine_drop035",
            "train": {
                "epochs": 12,
                "lr": 3e-4,
                "weight_decay": 5e-3,
                "patience": 4,
                "batch_size": 8,
                "grad_step": 2,
                "warmup_steps": 10,
                "lr_decay_factor": 0.5,
            },
            "model": {**deepcopy(base), "dropout_p": 0.35},
        },
    ]

    if preset == "fine":
        return fine
    return coarse


def rank_key(item):
    metrics = item["val_metrics"]
    return (
        metrics["auc"],
        metrics["f1-score"],
        metrics["accuracy"],
        -item["checkpoint_epoch"],
    )


def main():
    parser = argparse.ArgumentParser(description="Bounded hyperparameter tuning for the dual-branch EEG ViTransformer.")
    parser.add_argument("--data-dir", default="IEEE_subject_split", help="Directory containing train.pt / val.pt / test.pt.")
    parser.add_argument("--output-dir", default="tuning_runs", help="Directory to store sweep artifacts.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", choices=["coarse", "fine"], default="coarse")
    parser.add_argument("--max-experiments", type=int, default=None, help="Optional cap on number of candidates to run.")
    parser.add_argument("--run-test-on-best", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--override-epochs", type=int, default=None, help="Override all candidate epochs for short screening runs.")
    parser.add_argument("--override-patience", type=int, default=None, help="Override all candidate patience values.")
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    fix_random_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = EEGDataset(str(data_dir / "train.pt"))
    val_dataset = EEGDataset(str(data_dir / "val.pt"))
    test_dataset = EEGDataset(str(data_dir / "test.pt"))

    current_device = device(force_cuda=False)
    criterion = torch.nn.CrossEntropyLoss()

    candidates = build_candidates(args.preset)
    if args.max_experiments is not None:
        candidates = candidates[: args.max_experiments]

    results = []

    for idx, candidate in enumerate(candidates, start=1):
        fix_random_seed(args.seed)
        run_name = f"{idx:02d}_{candidate['name']}"
        model_path = output_dir / f"{run_name}.pt"
        train_cfg = candidate["train"]
        model_cfg = candidate["model"]
        if args.override_epochs is not None:
            train_cfg = {**train_cfg, "epochs": args.override_epochs}
        if args.override_patience is not None:
            train_cfg = {**train_cfg, "patience": args.override_patience}

        print(f"\n===== Running {run_name} =====")
        print(json.dumps({"train": train_cfg, "model": model_cfg}, indent=2))

        model = ViTransformer(**model_cfg)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
        )
        scheduler = WarmupScheduler(
            optimizer,
            lr=train_cfg["lr"],
            warmup_steps=train_cfg["warmup_steps"],
            decay_factor=train_cfg["lr_decay_factor"],
        )

        checkpoint_epoch = train(
            model=model,
            device=current_device,
            model_path=str(model_path),
            optimizer=optimizer,
            criterion=criterion,
            epochs=train_cfg["epochs"],
            train_loader=DataLoader(
                train_dataset,
                batch_size=train_cfg["batch_size"],
                shuffle=True,
            ),
            val_loader=DataLoader(
                val_dataset,
                batch_size=train_cfg["batch_size"],
                shuffle=False,
            ),
            gradient_step=train_cfg["grad_step"],
            patience=train_cfg["patience"],
            enable_fp16=False,
            scheduler=scheduler,
        )

        best_model = ViTransformer(**model_cfg)
        best_model.load_state_dict(
            torch.load(model_path, map_location=current_device, weights_only=True)
        )

        val_metrics = evaluate(
            best_model,
            current_device,
            DataLoader(val_dataset, batch_size=train_cfg["batch_size"], shuffle=False),
            enable_fp16=False,
            verbose=False,
        )

        result = {
            "name": candidate["name"],
            "run_name": run_name,
            "checkpoint_epoch": checkpoint_epoch,
            "model_path": str(model_path),
            "train": train_cfg,
            "model": model_cfg,
            "val_metrics": val_metrics,
        }
        results.append(result)

        (output_dir / f"{run_name}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"run_name": run_name, "val_metrics": val_metrics}, indent=2))

    ranked = sorted(results, key=rank_key, reverse=True)
    summary = {
        "selection_rule": "highest validation AUC, then F1, then accuracy",
        "seed": args.seed,
        "results": ranked,
    }

    if args.run_test_on_best and ranked:
        best = ranked[0]
        best_model = ViTransformer(**best["model"])
        best_model.load_state_dict(
            torch.load(best["model_path"], map_location=current_device, weights_only=True)
        )
        test_metrics = evaluate(
            best_model,
            current_device,
            DataLoader(test_dataset, batch_size=best["train"]["batch_size"], shuffle=False),
            enable_fp16=False,
            verbose=False,
        )
        summary["best_test_metrics"] = test_metrics

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\n===== Ranked Results =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
