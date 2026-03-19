import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.transformer import ViTransformer
from models.vit_config import get_default_vit_config
from utils import EEGDataset, WarmupScheduler, device, evaluate, fix_random_seed, train


ABLATION_MODES = [
    "temporal_only",
    "temporal_plus_tf",
    "full_fusion",
]


def run_single_ablation(
    mode: str,
    train_dataset,
    val_dataset,
    test_dataset,
    current_device,
    output_dir: Path,
):
    train_cfg = {
        "epochs": 20,
        "lr": 3e-4,
        "weight_decay": 5e-3,
        "patience": 6,
        "batch_size": 8,
        "grad_step": 2,
        "warmup_steps": 10,
        "lr_decay_factor": 0.5,
    }

    model_cfg = get_default_vit_config()
    model_cfg.update(
        {
            "input_channel": 19,
            "seq_length": 9250,
            "num_classes": 2,
            "ablation_mode": mode,
        }
    )

    model_path = output_dir / f"{mode}.pt"
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
        criterion=torch.nn.CrossEntropyLoss(),
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
    test_metrics = evaluate(
        best_model,
        current_device,
        DataLoader(test_dataset, batch_size=train_cfg["batch_size"], shuffle=False),
        enable_fp16=False,
        verbose=False,
    )

    result = {
        "mode": mode,
        "checkpoint_epoch": checkpoint_epoch,
        "model_path": str(model_path),
        "train_config": train_cfg,
        "model_config": model_cfg,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    (output_dir / f"{mode}.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    fix_random_seed(42)

    data_dir = Path("IEEE_subject_split")
    output_dir = Path("ablation_runs")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = EEGDataset(str(data_dir / "train.pt"))
    val_dataset = EEGDataset(str(data_dir / "val.pt"))
    test_dataset = EEGDataset(str(data_dir / "test.pt"))
    current_device = device(force_cuda=False)

    results = []
    for mode in ABLATION_MODES:
        print(f"\n===== Running ablation: {mode} =====")
        result = run_single_ablation(
            mode=mode,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            current_device=current_device,
            output_dir=output_dir,
        )
        print(json.dumps(result, indent=2))
        results.append(result)

    summary = {
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\n===== Ablation Summary =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
