"""Datasets: the one-hot modular task and the induced distribution shift."""

import torch

from evogrokking import datasets
from evogrokking.subclasses import _kmeans, equation_1, subsample_shifted


# --------------------------------------------------------------------------
# Registry / modular task
# --------------------------------------------------------------------------
def test_dataset_registry_exposes_shifted_and_plain():
    assert datasets.is_image("mnist") and not datasets.is_image("modadd")
    assert "mnist" in datasets.SHIFTED and "mnist_plain" not in datasets.SHIFTED
    try:
        datasets.load("nosuchdataset")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown dataset")


def test_modular_addition_is_one_hot_not_embedded():
    # No embedding layer exists any more, so the operands arrive one-hot: two hot
    # entries per row, in a 2p-wide vector.
    p = 7
    ds = datasets.modular_addition(p=p, train_frac=0.5)
    assert ds.spec.input_dim == 2 * p
    assert not hasattr(ds.spec, "vocab_size")
    assert ds.x_train.shape[1] == 2 * p
    assert torch.equal(ds.x_train.sum(dim=1), torch.full((len(ds.x_train),), 2.0))
    # The two hot positions decode back to a valid (a + b) mod p label.
    row, label = ds.x_train[0], ds.y_train[0].item()
    a = int(row[:p].argmax())
    b = int(row[p:].argmax())
    assert (a + b) % p == label


def test_modular_train_val_split_is_disjoint():
    ds = datasets.modular_addition(p=11, train_frac=0.4)
    total = 11 * 11
    assert len(ds.x_train) + len(ds.x_val) == total
    assert len(ds.x_train) == int(0.4 * total)


# --------------------------------------------------------------------------
# The distribution shift (Carvalho et al. 2025)
# --------------------------------------------------------------------------
def test_equation_1_matches_the_paper_worked_example():
    # Paper §4: 4 classes x 2 subclasses, gamma_D = 2000, f = 0.2, one subclass
    # per class subsampled -> gamma_s = 4, gamma_r = 4 -> s_s = 84, s_r = 416.
    s_s, s_r = equation_1(total=2000, n_shifted=4, n_kept=4, frac=0.2)
    assert (s_s, s_r) == (84, 416)
    assert 4 * s_s + 4 * s_r == 2000  # ...and the budget is respected


def test_equation_1_endpoints():
    s_s, s_r = equation_1(total=800, n_shifted=4, n_kept=4, frac=1.0)
    assert s_s == s_r == 100  # f = 1 is no shift at all
    s_s, s_r = equation_1(total=800, n_shifted=4, n_kept=4, frac=0.0)
    assert s_s == 0 and s_r == 200  # f = 0 removes them entirely


def test_equation_1_rejects_out_of_range_fraction():
    for bad in (-0.1, 1.5):
        try:
            equation_1(total=100, n_shifted=2, n_kept=2, frac=bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for frac={bad}")


def test_subsample_shifted_actually_shifts_the_distribution():
    n_subclasses, num_classes, per_sub = 4, 10, 500
    sub_ids = torch.arange(num_classes * n_subclasses).repeat_interleave(per_sub)

    idx = subsample_shifted(
        sub_ids, total=2000, n_subclasses=n_subclasses, num_classes=num_classes,
        shifted_per_class=1, frac=0.05, seed=0,
    )
    counts = torch.bincount(sub_ids[idx], minlength=num_classes * n_subclasses)
    assert (counts == counts.min()).sum().item() == num_classes
    assert counts.min() * 5 < counts.max()
    assert len(idx) <= 2000 * 1.1

    # Classes themselves stay balanced -- the shift is *within* class, which is
    # what makes train and test differ in representation but not in label prior.
    per_class = torch.bincount(sub_ids[idx] // n_subclasses, minlength=num_classes)
    assert per_class.max() - per_class.min() <= 2


def test_subsample_shifted_can_remove_a_subclass_entirely():
    sub_ids = torch.arange(8).repeat_interleave(100)
    idx = subsample_shifted(
        sub_ids, total=400, n_subclasses=2, num_classes=4,
        shifted_per_class=1, frac=0.0, seed=0,
    )
    counts = torch.bincount(sub_ids[idx], minlength=8)
    assert (counts == 0).sum().item() == 4


def test_subsample_shifted_is_reproducible_from_the_seed():
    sub_ids = torch.arange(8).repeat_interleave(100)
    kw = dict(total=400, n_subclasses=2, num_classes=4, shifted_per_class=1, frac=0.1)
    a = subsample_shifted(sub_ids, seed=3, **kw)
    b = subsample_shifted(sub_ids, seed=3, **kw)
    c = subsample_shifted(sub_ids, seed=4, **kw)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_kmeans_recovers_separated_clusters():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(50, 4, generator=g) + 10.0
    b = torch.randn(50, 4, generator=g) - 10.0
    assign = _kmeans(torch.cat([a, b]), k=2, seed=0)
    assert len(assign[:50].unique()) == 1 and len(assign[50:].unique()) == 1
    assert assign[0] != assign[50]
