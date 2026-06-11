from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda import amp

from snn_minimal.data import (
    build_train_val_dataloaders,
    get_dataset_spec,
    infer_pin_memory,
)
from snn_minimal.engine import load_checkpoint, run_epoch, save_checkpoint
from snn_minimal.models import ConvSNN
from snn_minimal.plotting import setup_recorder


def parse_args():
    parser = argparse.ArgumentParser(description="Train a minimal Conv-SNN baseline")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./runs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--neuron", choices=["lif", "if"], default="lif")
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(args, model):
    if args.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    return torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    dataset_spec = get_dataset_spec(args.dataset)

    if args.resume:
        # Keep writing into the original run folder when resuming.
        run_dir = Path(args.resume).resolve().parent
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.output_dir) / f"{args.dataset}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    recorder = setup_recorder(run_dir, resume=bool(args.resume))

    train_loader, val_loader = build_train_val_dataloaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=infer_pin_memory(device),
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
    )

    model = ConvSNN(
        dataset=args.dataset,
        timesteps=args.timesteps,
        channels=args.channels,
        hidden_dim=args.hidden_dim,
        neuron_type=args.neuron,
        tau=args.tau,
        dropout=args.dropout,
    ).to(device)

    optimizer = build_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = None
    if args.amp and device.type == "cuda":
        scaler = amp.GradScaler()

    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        checkpoint = load_checkpoint(
            Path(args.resume),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location="cpu",
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_acc = float(checkpoint.get("best_acc", 0.0))

    print(f"device={device}")
    print(model)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            num_classes=dataset_spec.num_classes,
            optimizer=optimizer,
            scaler=scaler,
            limit_batches=args.limit_train_batches,
            desc=f"train {epoch}",
        )
        scheduler.step()

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=dataset_spec.num_classes,
            optimizer=None,
            scaler=None,
            limit_batches=args.limit_val_batches,
            desc=f"val {epoch}",
        )

        latest_path = run_dir / "checkpoint_latest.pt"
        save_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_acc=max(best_acc, val_metrics.acc),
            args=vars(args),
        )

        if val_metrics.acc >= best_acc:
            best_acc = val_metrics.acc
            save_checkpoint(
                path=run_dir / "checkpoint_best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_acc=best_acc,
                args=vars(args),
            )

        elapsed = time.time() - epoch_start
        recorder.record_epoch(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_acc=best_acc,
            elapsed=elapsed,
            lr=optimizer.param_groups[0]["lr"],
        )

    recorder.finalize()


if __name__ == "__main__":
    main()
