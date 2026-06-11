from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from torch import nn
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EpochMetrics:
    loss: float
    acc: float
    samples: int


def _one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(target, num_classes=num_classes).float()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[amp.GradScaler] = None,
    limit_batches: Optional[int] = None,
    desc: str = "train",
) -> EpochMetrics:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    iterator = tqdm(loader, desc=desc, leave=False)
    amp_enabled = scaler is not None and device.type == "cuda"
    context = torch.enable_grad if is_train else torch.no_grad

    with context():
        for step, (images, targets) in enumerate(iterator):
            if limit_batches is not None and step >= limit_batches:
                break

            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            target_one_hot = _one_hot(targets, num_classes)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with amp.autocast(enabled=amp_enabled):
                outputs = model(images)
                loss = F.mse_loss(outputs, target_one_hot)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            # Spiking states must not leak across mini-batches.
            functional.reset_net(model)

            batch_size = targets.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == targets).sum().item()

            iterator.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{total_correct / max(total_samples, 1):.4f}",
            )

    return EpochMetrics(
        loss=total_loss / max(total_samples, 1),
        acc=total_correct / max(total_samples, 1),
        samples=total_samples,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_acc: float,
    args: dict,
):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "args": args,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    map_location: str = "cpu",
):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint
