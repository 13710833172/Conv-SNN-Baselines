from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from snn_minimal.data import build_test_dataloader, get_dataset_spec, infer_pin_memory
from snn_minimal.engine import load_checkpoint, run_epoch
from snn_minimal.models import ConvSNN


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate best checkpoint on test set")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory path")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (default: <run-dir>/checkpoint_best.pt)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root. Default uses args.json value.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _read_args_json(run_dir: Path) -> dict:
    args_path = run_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"args.json not found in run dir: {run_dir}")
    with args_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    cli_args = parse_args()

    run_dir = Path(cli_args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    checkpoint_path = (
        Path(cli_args.checkpoint).resolve()
        if cli_args.checkpoint
        else run_dir / "checkpoint_best.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    train_args = _read_args_json(run_dir)
    dataset = train_args["dataset"]
    data_root = cli_args.data_root or train_args.get("data_root", "./data")
    batch_size = cli_args.batch_size or int(train_args.get("batch_size", 64))
    num_workers = (
        cli_args.num_workers
        if cli_args.num_workers is not None
        else int(train_args.get("num_workers", 2))
    )
    device = resolve_device(cli_args.device)
    dataset_spec = get_dataset_spec(dataset)

    test_loader = build_test_dataloader(
        dataset_name=dataset,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=infer_pin_memory(device),
    )

    model = ConvSNN(
        dataset=dataset,
        timesteps=int(train_args.get("timesteps", 4)),
        channels=int(train_args.get("channels", 64)),
        hidden_dim=int(train_args.get("hidden_dim", 256)),
        neuron_type=str(train_args.get("neuron", "lif")),
        tau=float(train_args.get("tau", 2.0)),
        dropout=float(train_args.get("dropout", 0.25)),
    ).to(device)

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=None,
        scheduler=None,
        map_location="cpu",
    )
    ckpt_epoch = int(checkpoint.get("epoch", -1))
    ckpt_best_acc = float(checkpoint.get("best_acc", 0.0))

    print(f"device={device}")
    print(f"checkpoint={checkpoint_path}")
    print(f"checkpoint_epoch={ckpt_epoch} checkpoint_best_val_acc={ckpt_best_acc:.4f}")

    test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=dataset_spec.num_classes,
        optimizer=None,
        scaler=None,
        limit_batches=cli_args.limit_test_batches,
        desc="test",
    )

    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt_epoch,
        "checkpoint_best_val_acc": ckpt_best_acc,
        "test_loss": float(test_metrics.loss),
        "test_acc": float(test_metrics.acc),
        "test_samples": int(test_metrics.samples),
    }
    out_path = run_dir / "test_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(
        f"test_loss={test_metrics.loss:.4f} test_acc={test_metrics.acc:.4f} "
        f"samples={test_metrics.samples}"
    )
    print(f"saved test metrics: {out_path}")


if __name__ == "__main__":
    main()
