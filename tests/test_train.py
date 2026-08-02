"""Training a genome: early stopping and seed reproducibility."""

import torch

from evogrokking import datasets
from evogrokking.genome import Genome
from evogrokking.train import EarlyStopping, train_and_evaluate

CPU = torch.device("cpu")


def _dataset():
    return datasets.modular_addition(p=11, train_frac=0.6)


def _genome(ds, hidden=16):
    return Genome.dense(ds.spec.input_dim, ds.spec.num_classes, hidden)


# --------------------------------------------------------------------------
# Early stopping
# --------------------------------------------------------------------------
def test_early_stopping_on_target_val_acc():
    es = EarlyStopping(target_val_acc=0.99)
    assert not es.update(0, val_loss=1.0, val_acc=0.5)
    assert es.update(10, val_loss=0.9, val_acc=0.995)
    assert es.stopped_epoch == 10


def test_early_stopping_on_patience():
    es = EarlyStopping(patience=3, min_delta=1e-3)
    assert not es.update(0, val_loss=1.0, val_acc=0.0)  # improves (from inf)
    assert not es.update(1, val_loss=1.0, val_acc=0.0)  # no improvement -> 1
    assert not es.update(2, val_loss=1.0, val_acc=0.0)  # -> 2
    assert es.update(3, val_loss=1.0, val_acc=0.0)  # -> 3 == patience, stop


def test_patience_tracks_best_so_far_not_last():
    # The long pre-grokking plateau must not trigger a stop: patience resets on
    # any new best, however late it arrives.
    es = EarlyStopping(patience=2, min_delta=1e-3)
    assert not es.update(0, val_loss=1.0, val_acc=0.0)
    assert not es.update(1, val_loss=1.0, val_acc=0.0)  # flat -> 1 bad step
    assert not es.update(2, val_loss=0.5, val_acc=0.0)  # new best -> reset
    assert not es.update(3, val_loss=0.5, val_acc=0.0)  # flat -> 1
    assert es.update(4, val_loss=0.5, val_acc=0.0)  # -> 2 == patience


def test_train_respects_early_stopping():
    ds = _dataset()
    es = EarlyStopping(target_val_acc=0.0)  # reached at the first evaluation
    result = train_and_evaluate(
        _genome(ds, 8), ds, epochs=500, device=CPU, early_stopping=es, seed=0
    )
    assert result.stopped_epoch == 0
    assert len(result.train_losses) == 1


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def test_train_runs_and_returns_metrics():
    ds = _dataset()
    result = train_and_evaluate(_genome(ds), ds, epochs=50, device=CPU, seed=0)
    assert len(result.train_losses) > 0
    assert result.metrics.final_train_acc >= 0.0
    assert result.n_params > 0
    assert result.seed == 0  # the training seed is recorded for reproduction


def test_same_seed_reproduces_run():
    # Two runs of the same genome with the same seed are bit-for-bit identical,
    # which is what lets `retrain` reproduce the search.
    ds = _dataset()
    genome = _genome(ds)
    a = train_and_evaluate(genome, ds, epochs=60, device=CPU, seed=123)
    b = train_and_evaluate(genome, ds, epochs=60, device=CPU, seed=123)
    assert a.train_losses == b.train_losses
    assert a.val_losses == b.val_losses

    # A longer retrain shares the identical prefix with the shorter search run.
    longer = train_and_evaluate(genome, ds, epochs=120, device=CPU, seed=123)
    assert longer.train_losses[: len(a.train_losses)] == a.train_losses

    # A different seed gives a different trajectory.
    other = train_and_evaluate(genome, ds, epochs=60, device=CPU, seed=7)
    assert other.train_losses != a.train_losses
