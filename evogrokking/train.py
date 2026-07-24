"""Train one network and measure how much it grokked.

This is the fitness evaluation at the heart of the evolutionary search: given a
genome and a dataset it builds the network, trains it full-batch with Adam/SGD
(on CUDA when available), records the train/val loss and accuracy curves, and
returns the :class:`~evogrokking.metrics.GrokkingMetrics` for the run.

Full-batch training keeps the loss curves smooth (so the grokking area is not
dominated by mini-batch noise) and is fast for the small datasets we use.

Two controls bound the run length:

* ``epochs``  -- the **maximum number of iterations** (a hard cap), and
* :class:`EarlyStopping` -- an optional, grokking-aware stop that ends a run once
  it has clearly finished: either the model has *fully grokked* (validation
  accuracy reaches ``target_val_acc``) or the validation loss has stopped
  improving for ``patience`` evaluations.  Tracking the *best-so-far* validation
  loss is what makes this safe for grokking: the long pre-grokking plateau, where
  validation loss is flat or even rising, does not count as "improvement lost"
  until after grokking has driven the loss to new minima and it finally levels
  off.
"""

from __future__ import annotations

import os

# Must be set before the first cuBLAS call for deterministic GEMMs on CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from dataclasses import dataclass

import torch
import torch.nn as nn

from evogrokking.datasets import Dataset
from evogrokking.genome import Genome
from evogrokking.hyperparams import Hyperparams
from evogrokking.metrics import GEN_THRESHOLD, GrokkingMetrics, grokking_metrics
from evogrokking.models import GrokNet, build_model, count_parameters


@dataclass
class EarlyStopping:
    """Grokking-aware early-stopping criterion.

    Evaluated once per recorded point (i.e. every ``eval_every`` epochs).  A run
    is stopped when *either*:

    * validation accuracy reaches ``target_val_acc`` -- the model has grokked, so
      further training only wastes compute; or
    * the best-so-far validation loss has not improved by at least ``min_delta``
      for ``patience`` consecutive evaluations.

    Set ``patience=None`` and ``target_val_acc=None`` to disable early stopping
    entirely (the run then always uses the full ``epochs`` budget).
    """

    patience: int | None = None
    min_delta: float = 1e-4
    target_val_acc: float | None = None

    best_val_loss: float = float("inf")
    _bad_steps: int = 0
    stopped_epoch: int | None = None

    def update(self, epoch: int, val_loss: float, val_acc: float) -> bool:
        """Record one evaluation; return ``True`` if training should stop."""
        if self.target_val_acc is not None and val_acc >= self.target_val_acc:
            self.stopped_epoch = epoch
            return True

        if self.patience is None:
            return False

        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self._bad_steps = 0
        else:
            self._bad_steps += 1
            if self._bad_steps >= self.patience:
                self.stopped_epoch = epoch
                return True
        return False


@dataclass
class TrainResult:
    metrics: GrokkingMetrics
    train_losses: list[float]
    val_losses: list[float]
    train_accs: list[float]
    val_accs: list[float]
    n_params: int
    stopped_epoch: int | None = None  # None if the full epoch budget was used
    seed: int | None = None  # the seed this run was trained with (for reproduction)
    model: object = None  # the trained module (only when return_model=True); not serialised

    def as_dict(self) -> dict:
        return {
            "n_params": self.n_params,
            "stopped_epoch": self.stopped_epoch,
            "seed": self.seed,
            **self.metrics.as_dict(),
        }


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    """Seed and force deterministic kernels so a run reproduces exactly -- even on
    CUDA, where e.g. embedding backward otherwise uses non-deterministic atomics.

    This is what lets ``retrain`` reproduce the search phase's best individual.
    ``warn_only=True`` keeps any op lacking a deterministic implementation from
    crashing the run (it falls back, at worst, to its non-deterministic path).
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _make_optimizer(hp: Hyperparams, model: nn.Module) -> torch.optim.Optimizer:
    """Build the optimiser from the run's fixed recipe (not from the genome)."""
    if hp.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay
        )
    if hp.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay
        )
    if hp.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=hp.lr,
            weight_decay=hp.weight_decay,
            momentum=0.9,
        )
    raise ValueError(f"unknown optimizer {hp.optimizer!r}")


@torch.no_grad()
def _evaluate(model: nn.Module, x, y, criterion) -> tuple[float, float]:
    logits = model(x)
    loss = criterion(logits, y).item()
    acc = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, acc


def train_and_evaluate(
    genome: Genome,
    dataset: Dataset,
    *,
    hp: Hyperparams | None = None,
    epochs: int = 2000,
    device: torch.device | None = None,
    eval_every: int = 1,
    log_every: int = 0,
    model: GrokNet | None = None,
    early_stopping: EarlyStopping | None = None,
    seed: int | None = None,
    gen_threshold: float = GEN_THRESHOLD,
    return_model: bool = False,
) -> TrainResult:
    """Train ``genome`` on ``dataset`` and return its grokking metrics.

    ``hp`` is the run's fixed training recipe; it defaults to the per-task
    defaults of :meth:`Hyperparams.for_task`.

    ``epochs`` is the maximum number of iterations.  ``eval_every`` sub-samples
    the loss curves to save time on long runs; the grokking area is computed on
    the recorded points.  Pass ``early_stopping`` to stop before ``epochs`` once
    the run has grokked or its validation loss has plateaued.  ``seed`` seeds the
    weight initialisation so a genome trains identically regardless of evaluation
    order -- important for reproducible *parallel* evolution and for ``retrain``.

    The model is built *after* seeding so its initialisation is reproducible; do
    not pass a pre-built ``model`` if you rely on that (pass ``return_model=True``
    to get the trained module back instead).
    """
    device = device or default_device()
    hp = hp or Hyperparams.for_task(dataset.spec.task)
    if seed is not None:
        _seed_everything(seed)
    dataset = dataset.to(device)
    model = (model or build_model(genome, dataset.spec, hp)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(hp, model)

    x_tr, y_tr = dataset.x_train, dataset.y_train
    x_va, y_va = dataset.x_val, dataset.y_val

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []
    stopped_epoch: int | None = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()

        if epoch % eval_every == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                tr_acc = (logits.argmax(dim=1) == y_tr).float().mean().item()
            va_loss, va_acc = _evaluate(model, x_va, y_va, criterion)
            train_losses.append(loss.item())
            val_losses.append(va_loss)
            train_accs.append(tr_acc)
            val_accs.append(va_acc)

            if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
                print(
                    f"  epoch {epoch:5d} | train {loss.item():8.4f}/{tr_acc:5.1%} "
                    f"| val {va_loss:8.4f}/{va_acc:5.1%}"
                )

            if early_stopping is not None and early_stopping.update(
                epoch, va_loss, va_acc
            ):
                stopped_epoch = epoch
                break

    metrics = grokking_metrics(
        train_losses, val_losses, train_accs, val_accs, gen_threshold=gen_threshold
    )
    return TrainResult(
        metrics=metrics,
        train_losses=train_losses,
        val_losses=val_losses,
        train_accs=train_accs,
        val_accs=val_accs,
        n_params=count_parameters(model),
        stopped_epoch=stopped_epoch,
        seed=seed,
        model=model if return_model else None,
    )
