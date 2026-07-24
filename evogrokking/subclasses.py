"""Inducing a train/test distribution shift in MNIST, after Carvalho et al. (2025).

*"Grokking Explained: A Statistical Phenomenon"* argues that grokking is driven by
a **distribution shift between training and test data**, and demonstrates it on
real data by giving MNIST a latent class hierarchy:

    "To create these shifts, we apply a clustering algorithm to each digit based
    on its representation in a learned feature space.  Specifically, we train a
    ResNet classifier on MNIST and use its latent representations as input for
    clustering. [...] The resulting training dataset contains the same digits as
    the test set but with a different distribution in their representations."

So each digit class is split into ``m`` **subclasses** by clustering its images
in a learned latent space, and the shift is created by *under-sampling* a chosen
subset of those subclasses in the **training** set only.  The test set keeps
MNIST's original distribution, so train and test contain the same digits drawn
from measurably different distributions over the latent space.

This module implements the three pieces:

1. :func:`latent_features` -- a small CNN trained briefly on the full training
   set; its penultimate layer is the learned feature space.  (The paper uses a
   ResNet; a small CNN is used here because the whole point of this project is to
   run *hundreds* of short trainings, and the clustering only needs features that
   are semantically organised, not state-of-the-art ones.  Results are cached, so
   this cost is paid once per dataset/seed.)
2. :func:`cluster_subclasses` -- k-means within each digit class, giving every
   training image a subclass id.
3. :func:`subsample_shifted` -- the paper's Equation 1, which turns a target
   training-set size, a set of subclasses to under-sample and a fraction ``f``
   into per-subclass sample counts.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# 1. A learned feature space
# --------------------------------------------------------------------------
class _FeatureCNN(nn.Module):
    """Small conv classifier; ``features`` is the penultimate representation."""

    def __init__(self, num_classes: int = 10, feat_dim: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, feat_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(feat_dim, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def latent_features(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    image_shape: tuple[int, int, int],
    num_classes: int = 10,
    epochs: int = 2,
    batch_size: int = 256,
    seed: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Train a small classifier and return its penultimate features.

    ``images`` is the flattened, normalised training set; the returned tensor is
    ``(len(images), feat_dim)`` on the CPU.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    model = _FeatureCNN(num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    imgs = images.view(len(images), *image_shape)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(imgs), generator=g)
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = imgs[idx].to(device), labels[idx].to(device)
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()

    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            feats.append(model.features(imgs[i : i + batch_size].to(device)).cpu())
    return torch.cat(feats)


# --------------------------------------------------------------------------
# 2. Clustering each class into subclasses
# --------------------------------------------------------------------------
def _kmeans(
    x: torch.Tensor, k: int, *, iters: int = 25, seed: int = 0
) -> torch.Tensor:
    """Lloyd's algorithm with k-means++ seeding; returns a cluster id per row.

    Implemented here rather than pulled in from scikit-learn to keep the
    dependency footprint of the project unchanged.
    """
    n = len(x)
    if n <= k:
        return torch.arange(n) % k

    g = torch.Generator().manual_seed(seed)
    # k-means++ seeding: each new centre is drawn with probability proportional
    # to its squared distance from the nearest centre chosen so far.
    centres = [x[torch.randint(n, (1,), generator=g).item()]]
    for _ in range(1, k):
        d2 = torch.cdist(x, torch.stack(centres)).min(dim=1).values.pow(2)
        total = d2.sum()
        if total <= 0:
            centres.append(x[torch.randint(n, (1,), generator=g).item()])
        else:
            probs = d2 / total
            centres.append(x[torch.multinomial(probs, 1, generator=g).item()])
    c = torch.stack(centres)

    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        new_assign = torch.cdist(x, c).argmin(dim=1)
        if torch.equal(new_assign, assign):
            break
        assign = new_assign
        for j in range(k):
            members = x[assign == j]
            if len(members):
                c[j] = members.mean(dim=0)
    return assign


def cluster_subclasses(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_subclasses: int,
    num_classes: int = 10,
    seed: int = 0,
) -> torch.Tensor:
    """Cluster within each class; return a subclass id per sample.

    Subclass ids are ``class * n_subclasses + cluster``, so they identify both the
    digit and which sub-population of that digit the sample belongs to.
    """
    sub = torch.zeros(len(labels), dtype=torch.long)
    for cls in range(num_classes):
        idx = (labels == cls).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        # Standardise per class so clustering reflects direction in feature space
        # rather than the class's overall activation magnitude.
        f = features[idx]
        f = (f - f.mean(dim=0)) / (f.std(dim=0) + 1e-6)
        assign = _kmeans(f, n_subclasses, seed=seed + cls)
        sub[idx] = cls * n_subclasses + assign
    return sub


def subclass_ids(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    image_shape: tuple[int, int, int],
    n_subclasses: int,
    num_classes: int = 10,
    seed: int = 0,
    cache_path: str | None = None,
    feature_epochs: int = 2,
    verbose: bool = True,
) -> torch.Tensor:
    """Subclass id per training sample, computed once and cached on disk."""
    if cache_path and os.path.exists(cache_path):
        cached = torch.load(cache_path)
        if len(cached) == len(labels):
            return cached
    if verbose:
        print(
            f"  building subclass structure ({n_subclasses} clusters/class) -- "
            "training the feature extractor once; this is cached."
        )
    feats = latent_features(
        images,
        labels,
        image_shape=image_shape,
        num_classes=num_classes,
        epochs=feature_epochs,
        seed=seed,
    )
    sub = cluster_subclasses(
        feats, labels, n_subclasses=n_subclasses, num_classes=num_classes, seed=seed
    )
    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save(sub, cache_path)
    return sub


# --------------------------------------------------------------------------
# 3. The distribution shift (paper Equation 1)
# --------------------------------------------------------------------------
def equation_1(total: int, n_shifted: int, n_kept: int, frac: float) -> tuple[int, int]:
    """Per-subclass sample counts, Equation 1 of Carvalho et al. (2025)::

        s_s = ceil( f * gamma_D / (f * gamma_s + gamma_r) )
        s_r = floor(    gamma_D / (f * gamma_s + gamma_r) )

    ``total`` is the target training-set size ``gamma_D``, ``n_shifted`` the
    number of under-sampled subclasses ``gamma_s``, ``n_kept`` the number left at
    full strength ``gamma_r``, and ``frac`` the imbalance factor ``f`` in
    ``[0, 1]``: ``f = 1`` is no shift at all, ``f = 0`` removes the shifted
    subclasses entirely.

    Returns ``(s_s, s_r)`` -- how many samples to draw from each shifted and each
    kept subclass respectively.
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"frac must be in [0, 1], got {frac}")
    denom = frac * n_shifted + n_kept
    if denom <= 0:
        raise ValueError("no subclasses left to sample from")
    s_s = math.ceil(frac * total / denom)
    s_r = math.floor(total / denom)
    return s_s, s_r


def subsample_shifted(
    sub_ids: torch.Tensor,
    *,
    total: int,
    n_subclasses: int,
    num_classes: int = 10,
    shifted_per_class: int = 1,
    frac: float = 0.05,
    seed: int = 0,
) -> torch.Tensor:
    """Indices of a training subset whose subclass distribution is shifted.

    ``shifted_per_class`` subclasses of every digit are under-sampled to a
    fraction ``frac`` of the others' size, following :func:`equation_1`.  Which
    subclasses are picked is decided by ``seed``, so the shift is reproducible.
    """
    if not 0 <= shifted_per_class <= n_subclasses:
        raise ValueError(
            f"shifted_per_class must be in [0, {n_subclasses}], got {shifted_per_class}"
        )
    g = torch.Generator().manual_seed(seed)

    shifted: set[int] = set()
    for cls in range(num_classes):
        order = torch.randperm(n_subclasses, generator=g)[:shifted_per_class]
        shifted.update(int(cls * n_subclasses + c) for c in order)

    present = sorted({int(s) for s in sub_ids.unique()})
    n_shifted = sum(1 for s in present if s in shifted)
    n_kept = len(present) - n_shifted
    s_s, s_r = equation_1(total, n_shifted, n_kept, frac)

    picks: list[torch.Tensor] = []
    for s in present:
        idx = (sub_ids == s).nonzero(as_tuple=True)[0]
        take = min(s_s if s in shifted else s_r, len(idx))
        if take <= 0:
            continue
        picks.append(idx[torch.randperm(len(idx), generator=g)[:take]])
    if not picks:
        raise ValueError(
            "the requested shift selects no training samples at all; "
            "increase --train-size or --shift-frac"
        )
    return torch.cat(picks)
