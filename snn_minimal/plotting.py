from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from snn_minimal.engine import EpochMetrics

LOG_PATTERN = re.compile(
    r"epoch=(?P<epoch>\d+)\s+"
    r"train_loss=(?P<train_loss>[0-9]*\.?[0-9]+)\s+"
    r"train_acc=(?P<train_acc>[0-9]*\.?[0-9]+)\s+"
    r"val_loss=(?P<val_loss>[0-9]*\.?[0-9]+)\s+"
    r"val_acc=(?P<val_acc>[0-9]*\.?[0-9]+)\s+"
    r"best_val_acc=(?P<best_val_acc>[0-9]*\.?[0-9]+)\s+"
    r"time=(?P<time>[0-9]*\.?[0-9]+)s\s+"
    r"lr=(?P<lr>[0-9]*\.?[0-9]+)"
)


def empty_history() -> dict[str, list[float] | list[int]]:
    return {"epochs": [], "train_loss": [], "val_acc": []}


def _parse_history_from_log(path: Path) -> dict[str, list[float] | list[int]]:
    history = empty_history()
    if not path.exists():
        return history

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            match = LOG_PATTERN.match(line.strip())
            if match is None:
                continue
            history["epochs"].append(int(match.group("epoch")))
            history["train_loss"].append(float(match.group("train_loss")))
            history["val_acc"].append(float(match.group("val_acc")))
    return history


def _save_metrics_summary(path: Path, history: dict[str, list[float] | list[int]]) -> None:
    if not history["epochs"]:
        payload = {"final": None, "best": None}
    else:
        final_idx = len(history["epochs"]) - 1
        best_val = max(history["val_acc"])
        best_idx = max(
            idx for idx, val_acc in enumerate(history["val_acc"]) if val_acc == best_val
        )
        payload = {
            "final": {
                "epoch": int(history["epochs"][final_idx]),
                "train_loss": float(history["train_loss"][final_idx]),
                "val_acc": float(history["val_acc"][final_idx]),
            },
            "best": {
                "epoch": int(history["epochs"][best_idx]),
                "train_loss": float(history["train_loss"][best_idx]),
                "val_acc": float(history["val_acc"][best_idx]),
            },
        }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def plot_training_curves(
    history: dict[str, list[float] | list[int]], run_dir: Path
) -> tuple[Path, Path]:
    loss_path = run_dir / "train_loss.svg"
    acc_path = run_dir / "val_acc.svg"
    epochs = history["epochs"]
    if not epochs:
        return loss_path, acc_path

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_loss"], marker="o", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(loss_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["val_acc"], marker="o", linewidth=1.5, color="C1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Validation Accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(acc_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    return loss_path, acc_path


class TrainingRecorder:
    def __init__(self, run_dir: Path, resume: bool = False):
        self.run_dir = run_dir
        self.metrics_path = run_dir / "metrics.json"
        self.log_path = run_dir / "train.log"
        if resume:
            self.history = _parse_history_from_log(self.log_path)
        else:
            self.history = empty_history()
        log_mode = "a" if resume else "w"
        self._log_file: TextIO = self.log_path.open(log_mode, encoding="utf-8")
        _save_metrics_summary(self.metrics_path, self.history)

    def record_epoch(
        self,
        epoch: int,
        train_metrics: EpochMetrics,
        val_metrics: EpochMetrics,
        best_acc: float,
        elapsed: float,
        lr: float,
    ) -> str:
        line = (
            f"epoch={epoch} "
            f"train_loss={train_metrics.loss:.4f} train_acc={train_metrics.acc:.4f} "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.acc:.4f} "
            f"best_val_acc={best_acc:.4f} time={elapsed:.1f}s lr={lr:.6f}"
        )
        self._log_file.write(line + "\n")
        self._log_file.flush()

        self.history["epochs"].append(epoch)
        self.history["train_loss"].append(train_metrics.loss)
        self.history["val_acc"].append(val_metrics.acc)
        _save_metrics_summary(self.metrics_path, self.history)

        print(line)
        return line

    def finalize(self) -> tuple[Path, Path]:
        self._log_file.close()
        loss_path, acc_path = plot_training_curves(self.history, self.run_dir)
        if self.history["epochs"]:
            print(f"saved plots: {loss_path}, {acc_path}")
        print(f"training log: {self.log_path}")
        print(f"metrics: {self.metrics_path}")
        return loss_path, acc_path


def setup_recorder(run_dir: Path, resume: bool = False) -> TrainingRecorder:
    return TrainingRecorder(run_dir, resume=resume)
