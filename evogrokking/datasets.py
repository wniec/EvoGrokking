"""Datasets for grokking experiments.

Three families of task are supported, all reduced to in-memory tensors so that a
whole evolutionary search (hundreds of short trainings) stays fast:

* ``modadd``        -- p-modular addition ``(a + b) mod p`` as a classification
  task over ``p`` classes, in the style of Power et al. 2022 and the reference
  project in ``low_dimensional_grokking``.
* ``mnist`` / ``fashionmnist`` -- the classic image benchmarks.  Grokking on
  these requires a *small* training subset together with weight decay (cf.
  Liu et al., "Omnigrok"), so we subsample the training set.

Every loader returns a :class:`Dataset` bundle of train/val tensors plus a
:class:`DatasetSpec` describing the shapes the model builder needs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

DATA_ROOT = os.environ.get(
    "EVOGROKKING_DATA", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
)


@dataclass
class DatasetSpec:
    """Everything the model builder needs to size its input and output layers."""

    name: str
    task: str  # "modular" or "image"
    input_dim: int  # flattened feature dim (image) or number of tokens (modular)
    num_classes: int
    vocab_size: int | None = None  # modulus p for modular tasks; None for images
    image_shape: tuple[int, int, int] | None = None  # (C, H, W) for image tasks


@dataclass
class Dataset:
    spec: DatasetSpec
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor

    def to(self, device: torch.device) -> "Dataset":
        return Dataset(
            self.spec,
            self.x_train.to(device),
            self.y_train.to(device),
            self.x_val.to(device),
            self.y_val.to(device),
        )


def modular_addition(p: int = 97, train_frac: float = 0.4, seed: int = 0) -> Dataset:
    """``(a + b) mod p`` over every ordered pair ``(a, b)``.

    ``train_frac`` controls the fraction of the ``p * p`` pairs used for training;
    small fractions with weight decay are what makes the task grok.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    y = (x[:, 0] + x[:, 1]) % p
    perm = torch.randperm(len(x), generator=g)
    x, y = x[perm], y[perm]
    n_train = int(train_frac * len(x))
    spec = DatasetSpec(
        name=f"modadd_p{p}", task="modular", input_dim=2, num_classes=p, vocab_size=p
    )
    return Dataset(spec, x[:n_train], y[:n_train], x[n_train:], y[n_train:])


def _image_dataset(
    name: str, torchvision_cls, train_size: int, val_size: int, seed: int
) -> Dataset:
    from torchvision import datasets as tvd  # imported lazily; heavy dependency

    train_raw = torchvision_cls(DATA_ROOT, train=True, download=True)
    test_raw = torchvision_cls(DATA_ROOT, train=False, download=True)

    def to_tensors(raw, size, gen):
        images = raw.data.float().div_(255.0).reshape(len(raw.data), -1)
        # Normalise to zero mean / unit-ish scale; large input scale + weight
        # decay is part of what makes small-data image tasks grok.
        images = (images - images.mean()) / (images.std() + 1e-6)
        labels = torch.as_tensor(raw.targets, dtype=torch.long)
        if size is not None and size < len(images):
            idx = torch.randperm(len(images), generator=gen)[:size]
            images, labels = images[idx], labels[idx]
        return images, labels

    g = torch.Generator().manual_seed(seed)
    x_train, y_train = to_tensors(train_raw, train_size, g)
    x_val, y_val = to_tensors(test_raw, val_size, g)

    h, w = int(train_raw.data.shape[1]), int(train_raw.data.shape[2])
    spec = DatasetSpec(
        name=name,
        task="image",
        input_dim=x_train.shape[1],
        num_classes=int(max(y_train.max(), y_val.max()) + 1),
        vocab_size=None,
        image_shape=(1, h, w),  # single-channel; the conv path reshapes to this
    )
    return Dataset(spec, x_train, y_train, x_val, y_val)


def mnist(train_size: int = 1000, val_size: int = 2000, seed: int = 0) -> Dataset:
    from torchvision.datasets import MNIST

    return _image_dataset("mnist", MNIST, train_size, val_size, seed)


def fashion_mnist(train_size: int = 1000, val_size: int = 2000, seed: int = 0) -> Dataset:
    from torchvision.datasets import FashionMNIST

    return _image_dataset("fashionmnist", FashionMNIST, train_size, val_size, seed)


def load(name: str, **kwargs) -> Dataset:
    """Dispatch by name: ``modadd`` / ``mnist`` / ``fashionmnist``."""
    name = name.lower()
    if name in ("modadd", "modular", "modular_addition"):
        return modular_addition(**kwargs)
    if name == "mnist":
        return mnist(**kwargs)
    if name in ("fashionmnist", "fashion_mnist", "fashion"):
        return fashion_mnist(**kwargs)
    raise ValueError(f"unknown dataset {name!r}")
