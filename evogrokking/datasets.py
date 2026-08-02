"""Datasets for grokking experiments.

Three families of task are supported, all reduced to in-memory tensors so that a
whole evolutionary search (hundreds of short trainings) stays fast:

* ``modadd``        -- p-modular addition ``(a + b) mod p`` as a classification
  task over ``p`` classes, in the style of Power et al. 2022.  The two operands
  are **one-hot encoded** and concatenated into a ``2p`` feature vector: the
  networks here are plain MLPs over a fixed input vector, with no embedding
  layer, so the tokens have to arrive already encoded.
* ``mnist`` / ``fashionmnist`` -- the image benchmarks, in the
  **distribution-shifted** form of Carvalho et al. (2025): each digit class is
  split into latent subclasses and a subset of them is under-sampled in the
  training set only, so train and test hold the same digits drawn from different
  distributions.  That shift is what induces grokking (see
  :mod:`evogrokking.subclasses`).  ``mnist_plain`` / ``fashionmnist_plain``
  give the classic un-shifted versions, which merely subsample the training set
  (cf. Liu et al., "Omnigrok").

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
    input_dim: int  # length of the flat input vector -> one input neuron each
    num_classes: int  # one output neuron each
    image_shape: tuple[int, int, int] | None = None  # (C, H, W); image tasks only


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

    Each pair ``(a, b)`` becomes a ``2p``-dimensional vector: ``a`` one-hot in the
    first ``p`` positions, ``b`` one-hot in the second.  There is no embedding
    layer to learn a denser code -- the model is a plain MLP over these features.
    """
    g = torch.Generator().manual_seed(seed)
    pairs = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    y = (pairs[:, 0] + pairs[:, 1]) % p
    perm = torch.randperm(len(pairs), generator=g)
    pairs, y = pairs[perm], y[perm]

    x = torch.zeros(len(pairs), 2 * p)
    rows = torch.arange(len(pairs))
    x[rows, pairs[:, 0]] = 1.0
    x[rows, p + pairs[:, 1]] = 1.0

    n_train = int(train_frac * len(x))
    spec = DatasetSpec(
        name=f"modadd_p{p}", task="modular", input_dim=2 * p, num_classes=p
    )
    return Dataset(spec, x[:n_train], y[:n_train], x[n_train:], y[n_train:])


def _normalised(raw) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten a torchvision image dataset to zero-mean / unit-scale vectors.

    A large input scale together with weight decay is part of what makes
    small-data image tasks grok (Liu et al., "Omnigrok").
    """
    images = raw.data.float().div(255.0).reshape(len(raw.data), -1)
    images = (images - images.mean()) / (images.std() + 1e-6)
    labels = torch.as_tensor(raw.targets, dtype=torch.long)
    return images, labels


def _image_dataset(
    name: str,
    torchvision_cls,
    train_size: int,
    val_size: int,
    seed: int,
    train_frac: float | None = None,
) -> Dataset:
    train_raw = torchvision_cls(DATA_ROOT, train=True, download=True)
    test_raw = torchvision_cls(DATA_ROOT, train=False, download=True)

    # ``train_frac`` (a fraction of the *full* training set) takes precedence over
    # an absolute ``train_size`` when supplied, so the flag means the same thing
    # here as it does for the modular task.
    if train_frac is not None:
        train_size = int(train_frac * len(train_raw.data))
        if train_size < 1:
            raise ValueError(
                f"train_frac={train_frac} selects {train_size} of "
                f"{len(train_raw.data)} training images; use a larger fraction."
            )

    def to_tensors(raw, size, gen):
        images, labels = _normalised(raw)
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
        image_shape=(1, h, w),  # single-channel; used by the subclass clustering
    )
    return Dataset(spec, x_train, y_train, x_val, y_val)


def _shifted_image_dataset(
    name: str,
    torchvision_cls,
    train_size: int,
    val_size: int,
    seed: int,
    train_frac: float | None = None,
    n_subclasses: int = 4,
    shifted_per_class: int = 1,
    shift_frac: float = 0.05,
    feature_epochs: int = 2,
) -> Dataset:
    """An image dataset with a train/test **distribution shift**, after Carvalho
    et al. (2025).

    The validation split is the untouched test set (the original distribution).
    The training split is drawn from the same digits, but ``shifted_per_class``
    latent subclasses of every class are under-sampled to a fraction
    ``shift_frac`` of the rest -- so the two splits differ in their distribution
    over the learned feature space while containing the same classes.
    ``shift_frac=1.0`` reproduces a balanced (unshifted) subsample;
    ``shift_frac=0.0`` removes those subclasses from training altogether.
    """
    from evogrokking import subclasses as sub

    train_raw = torchvision_cls(DATA_ROOT, train=True, download=True)
    test_raw = torchvision_cls(DATA_ROOT, train=False, download=True)

    full_train, y_full = _normalised(train_raw)
    if train_frac is not None:
        train_size = int(train_frac * len(full_train))
        if train_size < 1:
            raise ValueError(
                f"train_frac={train_frac} selects {train_size} of "
                f"{len(full_train)} training images; use a larger fraction."
            )

    h, w = int(train_raw.data.shape[1]), int(train_raw.data.shape[2])
    image_shape = (1, h, w)
    num_classes = int(y_full.max()) + 1

    cache = os.path.join(
        DATA_ROOT, f"{name}_subclasses_k{n_subclasses}_s{seed}.pt"
    )
    sub_ids = sub.subclass_ids(
        full_train,
        y_full,
        image_shape=image_shape,
        n_subclasses=n_subclasses,
        num_classes=num_classes,
        seed=seed,
        cache_path=cache,
        feature_epochs=feature_epochs,
    )
    idx = sub.subsample_shifted(
        sub_ids,
        total=train_size,
        n_subclasses=n_subclasses,
        num_classes=num_classes,
        shifted_per_class=shifted_per_class,
        frac=shift_frac,
        seed=seed,
    )
    x_train, y_train = full_train[idx], y_full[idx]

    g = torch.Generator().manual_seed(seed)
    x_val, y_val = _normalised(test_raw)
    if val_size is not None and val_size < len(x_val):
        vidx = torch.randperm(len(x_val), generator=g)[:val_size]
        x_val, y_val = x_val[vidx], y_val[vidx]

    spec = DatasetSpec(
        name=name,
        task="image",
        input_dim=x_train.shape[1],
        num_classes=int(max(y_train.max(), y_val.max()) + 1),
        image_shape=image_shape,
    )
    return Dataset(spec, x_train, y_train, x_val, y_val)


def mnist_plain(
    train_size: int = 1000,
    val_size: int = 2000,
    seed: int = 0,
    train_frac: float | None = None,
) -> Dataset:
    """Classic MNIST: a balanced random subsample of the training set."""
    from torchvision.datasets import MNIST

    return _image_dataset("mnist_plain", MNIST, train_size, val_size, seed, train_frac)


def fashion_mnist_plain(
    train_size: int = 1000,
    val_size: int = 2000,
    seed: int = 0,
    train_frac: float | None = None,
) -> Dataset:
    """Classic FashionMNIST: a balanced random subsample of the training set."""
    from torchvision.datasets import FashionMNIST

    return _image_dataset(
        "fashionmnist_plain", FashionMNIST, train_size, val_size, seed, train_frac
    )


def mnist(
    train_size: int = 1000,
    val_size: int = 2000,
    seed: int = 0,
    train_frac: float | None = None,
    n_subclasses: int = 4,
    shifted_per_class: int = 1,
    shift_frac: float = 0.05,
    feature_epochs: int = 2,
) -> Dataset:
    """MNIST with a shifted training class distribution (the default)."""
    from torchvision.datasets import MNIST

    return _shifted_image_dataset(
        "mnist",
        MNIST,
        train_size,
        val_size,
        seed,
        train_frac,
        n_subclasses,
        shifted_per_class,
        shift_frac,
        feature_epochs,
    )


def fashion_mnist(
    train_size: int = 1000,
    val_size: int = 2000,
    seed: int = 0,
    train_frac: float | None = None,
    n_subclasses: int = 4,
    shifted_per_class: int = 1,
    shift_frac: float = 0.05,
    feature_epochs: int = 2,
) -> Dataset:
    """FashionMNIST with a shifted training class distribution."""
    from torchvision.datasets import FashionMNIST

    return _shifted_image_dataset(
        "fashionmnist",
        FashionMNIST,
        train_size,
        val_size,
        seed,
        train_frac,
        n_subclasses,
        shifted_per_class,
        shift_frac,
        feature_epochs,
    )


#: Dataset name -> loader.  ``mnist``/``fashionmnist`` are the
#: distribution-shifted versions; ``*_plain`` are the classic ones.
LOADERS = {
    "modadd": modular_addition,
    "modular": modular_addition,
    "modular_addition": modular_addition,
    "mnist": mnist,
    "fashionmnist": fashion_mnist,
    "fashion_mnist": fashion_mnist,
    "fashion": fashion_mnist,
    "mnist_plain": mnist_plain,
    "fashionmnist_plain": fashion_mnist_plain,
    "fashion_plain": fashion_mnist_plain,
}

#: Names whose loader accepts the distribution-shift knobs.
SHIFTED = {"mnist", "fashionmnist", "fashion_mnist", "fashion"}


def is_image(name: str) -> bool:
    return name.lower() not in ("modadd", "modular", "modular_addition")


def load(name: str, **kwargs) -> Dataset:
    """Dispatch by name.

    ``modadd`` | ``mnist`` | ``fashionmnist`` (distribution-shifted) |
    ``mnist_plain`` | ``fashionmnist_plain`` (classic).
    """
    try:
        loader = LOADERS[name.lower()]
    except KeyError:
        raise ValueError(
            f"unknown dataset {name!r}; expected one of {sorted(LOADERS)}"
        ) from None
    return loader(**kwargs)
