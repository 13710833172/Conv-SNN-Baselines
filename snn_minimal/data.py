from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    in_channels: int
    image_size: int
    num_classes: int = 10


DATASET_SPECS = {
    "mnist": DatasetSpec(name="mnist", in_channels=1, image_size=28),
    "cifar10": DatasetSpec(name="cifar10", in_channels=3, image_size=32),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    key = name.lower()
    if key not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset: {name}")
    return DATASET_SPECS[key]


def build_transforms(dataset_name: str):
    dataset_name = dataset_name.lower()
    if dataset_name == "mnist":
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        return train_transform, test_transform

    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        return train_transform, test_transform

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_datasets(dataset_name: str, data_root: str):
    train_transform, test_transform = build_transforms(dataset_name)
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        train_set = datasets.MNIST(
            root=data_root,
            train=True,
            transform=train_transform,
            download=True,
        )
        test_set = datasets.MNIST(
            root=data_root,
            train=False,
            transform=test_transform,
            download=True,
        )
        return train_set, test_set

    if dataset_name == "cifar10":
        train_set = datasets.CIFAR10(
            root=data_root,
            train=True,
            transform=train_transform,
            download=True,
        )
        test_set = datasets.CIFAR10(
            root=data_root,
            train=False,
            transform=test_transform,
            download=True,
        )
        return train_set, test_set

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _split_train_val_indices(
    total_size: int, val_ratio: float, split_seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    val_size = max(1, int(total_size * val_ratio))
    train_size = total_size - val_size
    if train_size <= 0:
        raise ValueError(
            f"val_ratio={val_ratio} is too large for dataset size={total_size}"
        )
    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(total_size, generator=generator)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return train_indices, val_indices


def build_train_val_dataloaders(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    val_ratio: float = 0.1,
    split_seed: int = 42,
):
    dataset_name = dataset_name.lower()
    train_transform, test_transform = build_transforms(dataset_name)

    if dataset_name == "mnist":
        train_set_aug = datasets.MNIST(
            root=data_root,
            train=True,
            transform=train_transform,
            download=True,
        )
        train_set_eval = datasets.MNIST(
            root=data_root,
            train=True,
            transform=test_transform,
            download=True,
        )
    elif dataset_name == "cifar10":
        train_set_aug = datasets.CIFAR10(
            root=data_root,
            train=True,
            transform=train_transform,
            download=True,
        )
        train_set_eval = datasets.CIFAR10(
            root=data_root,
            train=True,
            transform=test_transform,
            download=True,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_indices, val_indices = _split_train_val_indices(
        total_size=len(train_set_aug), val_ratio=val_ratio, split_seed=split_seed
    )

    train_set = Subset(train_set_aug, train_indices.tolist())
    val_set = Subset(train_set_eval, val_indices.tolist())

    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=True,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )
    return train_loader, val_loader


def build_test_dataloader(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
):
    _, test_set = build_datasets(dataset_name, data_root)
    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    return DataLoader(
        test_set,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )


def infer_pin_memory(device: torch.device) -> bool:
    return device.type == "cuda"
