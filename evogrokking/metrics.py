"""Quantifying grokking -- and scoring architectures that *avoid* it.

Grokking is *delayed generalisation*: the training loss collapses almost
immediately while the validation loss lingers high for a long time before it,
too, finally drops.  On a log-loss plot this shows up as a large, long-lived gap
between the two curves.

We measure the *scale* of grokking as the **area between the train and validation
log-loss curves**::

    grok_area = (1/T) * sum_t  max(0, log(val_loss_t) - log(train_loss_t))

The ``max(0, .)`` keeps the measure a "generalisation gap" (validation above
training) and the ``1/T`` normalisation makes runs of different length
comparable.  Alongside it we measure the **area between the train and validation
accuracy curves** (``acc_area``), which captures the same delayed-generalisation
gap but is *bounded* in ``[0, 1]`` and cannot be inflated by a diverging loss,
and ``gen_frac`` -- how far into the run validation accuracy first reached the
generalisation threshold.

The objective
-------------
The search **minimises** grokking: we are looking for architectures that
generalise *immediately* rather than after a long memorisation plateau -- while
still ending up accurate.  Those two halves pull against each other, and the
degenerate ways to "not grok" are exactly what the score has to exclude:

* a network that never learns anything has no train/val gap at all, and
* a network that memorises and never generalises has a permanent gap.

:meth:`GrokkingMetrics.score` therefore pays the anti-grokking reward only
through a sharp **accuracy gate** at ``gen_threshold``: below it the reward is
~0 and the score collapses to the (low) final accuracy, so neither degenerate
strategy pays.  Above it the individual collects credit for how *tightly*
validation tracked training (``1 - acc_area``) and how *early* it generalised
(``1 - gen_frac``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

# Accuracy at/above which a split is considered "generalised". 0.9 is right for
# modular addition (which groks to ~100%); lower it for harder image subsets.
GEN_THRESHOLD = 0.9

# Losses are clamped into ``[LOSS_MIN, LOSS_MAX]`` before the logarithm: the lower
# bound keeps a perfectly fit split from sending the measure to -inf, and the
# upper bound keeps a diverged (NaN/overflow) run from dominating the area.
LOSS_MIN = 1e-8
LOSS_MAX = 1e8


def _log_clamped(values: Sequence[float]) -> torch.Tensor:
    t = torch.as_tensor(values, dtype=torch.float64)
    # nan_to_num first so a diverged NaN loss becomes the finite upper bound
    # rather than poisoning the whole area.
    t = torch.nan_to_num(t, nan=LOSS_MAX, posinf=LOSS_MAX, neginf=LOSS_MIN)
    return torch.log(t.clamp(LOSS_MIN, LOSS_MAX))


@dataclass
class GrokkingMetrics:
    """Grokking metrics extracted from a single training run's curves."""

    grok_area: float
    """Area between the log-loss curves -- the literal grokking *scale*."""

    acc_area: float
    """Area between the train and validation *accuracy* curves, ``(1/T) * sum
    max(0, train_acc - val_acc)``.  A second, **bounded** ([0, 1]) view of the
    delayed-generalisation gap: unlike the loss area it cannot be inflated by a
    diverging loss, and it is measured in the units we ultimately care about."""

    gen_frac: float
    """How far into the run validation accuracy first reached ``gen_threshold``,
    normalised to ``[0, 1]``; ``1.0`` if it never got there.  This is the direct
    "how late was generalisation" measure the objective minimises."""

    grok_delay: float
    """Normalised lag between train and validation first reaching
    ``GEN_THRESHOLD``.  Zero if validation never gets there (i.e. overfitting) --
    a positive delay is the memorise-first-then-generalise signature."""

    val_loss_drop: float
    """How far validation log-loss fell from its peak, ``log(max) - log(final)``.
    ~0 for overfitting (val loss never recovers), large for grokking."""

    generalised: bool
    """Whether validation accuracy ever reached ``GEN_THRESHOLD``."""

    final_train_loss: float
    final_val_loss: float
    final_train_acc: float
    final_val_acc: float

    gen_threshold: float = GEN_THRESHOLD
    """The threshold these metrics were computed against.  :meth:`score` defaults
    to it, so a reported score always matches the run it came from."""

    def grok_magnitude(self, acc_weight: float = 5.0) -> float:
        """How much this run grokked: the loss area plus the (rescaled) accuracy
        area.  Reported for analysis; the objective uses the bounded terms
        directly so that a diverging loss cannot dominate it."""
        return self.grok_area + acc_weight * self.acc_area

    def score(
        self,
        acc_weight: float = 1.0,
        gap_weight: float = 1.0,
        speed_weight: float = 1.0,
        gen_threshold: float | None = None,
        sharpness: float = 30.0,
    ) -> float:
        """Composite fitness maximised by the search -- i.e. **minimal** grokking
        at **high** final accuracy.

        ::

            gate      = sigmoid(sharpness * (final_val_acc - gen_threshold))
            tightness = 1 - acc_area   # validation tracked training closely
            speed     = 1 - gen_frac   # generalisation happened early
            score     = acc_weight * final_val_acc
                      + gate * (gap_weight * tightness + speed_weight * speed)

        ``gate`` is what makes the two degenerate non-grokking strategies
        worthless: a network that never learns and a network that memorises
        forever both sit below ``gen_threshold``, score ~0 on the gate, and are
        left with only their (low) ``final_val_acc``.  An individual that reaches
        the threshold collects the anti-grokking reward in proportion to how
        small its train/validation gap was and how early it generalised.

        ``gen_threshold`` defaults to the one these metrics were computed with,
        so a score reported beside a run always uses that run's own gate.
        """
        thr = self.gen_threshold if gen_threshold is None else gen_threshold
        gate = 1.0 / (1.0 + math.exp(-sharpness * (self.final_val_acc - thr)))
        tightness = 1.0 - self.acc_area
        speed = 1.0 - self.gen_frac
        return acc_weight * self.final_val_acc + gate * (
            gap_weight * tightness + speed_weight * speed
        )

    def as_dict(self) -> dict:
        return {
            "grok_area": self.grok_area,
            "acc_area": self.acc_area,
            "gen_frac": self.gen_frac,
            "grok_delay": self.grok_delay,
            "val_loss_drop": self.val_loss_drop,
            "generalised": self.generalised,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "final_train_acc": self.final_train_acc,
            "final_val_acc": self.final_val_acc,
            "gen_threshold": self.gen_threshold,
            "grok_magnitude": self.grok_magnitude(),
            "score": self.score(),
        }


def _first_cross(accs: Sequence[float], threshold: float) -> int | None:
    for i, a in enumerate(accs):
        if a >= threshold:
            return i
    return None


def grokking_metrics(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    train_accs: Sequence[float],
    val_accs: Sequence[float],
    gen_threshold: float = GEN_THRESHOLD,
) -> GrokkingMetrics:
    """Compute :class:`GrokkingMetrics` from a run's per-epoch curves.

    Accuracies are expected as fractions in ``[0, 1]``.
    """
    if len(train_losses) == 0:
        raise ValueError("empty loss curves")

    log_train = _log_clamped(train_losses)
    log_val = _log_clamped(val_losses)
    gap = (log_val - log_train).clamp_min(0.0)

    # Trapezoidal area under the (non-negative) gap curve, normalised by length so
    # runs of different epoch counts are comparable.
    if gap.numel() > 1:
        area = torch.trapezoid(gap).item() / (gap.numel() - 1)
    else:
        area = gap.item()

    # Area between the accuracy curves (train above val): the same delayed gap,
    # but bounded in [0, 1] and immune to loss-scale artefacts.
    tr_acc = torch.as_tensor(train_accs, dtype=torch.float64)
    va_acc = torch.as_tensor(val_accs, dtype=torch.float64)
    acc_gap = (tr_acc - va_acc).clamp_min(0.0)
    if acc_gap.numel() > 1:
        acc_area = torch.trapezoid(acc_gap).item() / (acc_gap.numel() - 1)
    else:
        acc_area = acc_gap.item()

    # How far validation loss fell from its peak: the "did it recover?" signal
    # that separates grokking (large drop) from overfitting (~0).
    val_loss_drop = float((log_val.max() - log_val[-1]).clamp_min(0.0).item())

    # When generalisation happened, and how far it lagged behind memorisation.
    n = len(train_accs)
    i_train = _first_cross(train_accs, gen_threshold)
    i_val = _first_cross(val_accs, gen_threshold)
    if i_val is None:
        gen_frac = 1.0  # never generalised: maximally late, by convention
    elif n > 1:
        gen_frac = i_val / (n - 1)
    else:
        gen_frac = 0.0
    if i_train is not None and i_val is not None and i_val >= i_train and n > 1:
        grok_delay = (i_val - i_train) / (n - 1)
    else:
        grok_delay = 0.0

    return GrokkingMetrics(
        grok_area=float(area),
        acc_area=float(acc_area),
        gen_frac=float(gen_frac),
        grok_delay=float(grok_delay),
        val_loss_drop=val_loss_drop,
        generalised=i_val is not None,
        final_train_loss=float(train_losses[-1]),
        final_val_loss=float(val_losses[-1]),
        final_train_acc=float(train_accs[-1]),
        final_val_acc=float(val_accs[-1]),
        gen_threshold=float(gen_threshold),
    )


def area_between_log_losses(
    train_losses: Sequence[float], val_losses: Sequence[float]
) -> float:
    """Just the grokking area -- convenience wrapper used in tests/analysis."""
    log_train = _log_clamped(train_losses)
    log_val = _log_clamped(val_losses)
    gap = (log_val - log_train).clamp_min(0.0)
    if gap.numel() > 1:
        return torch.trapezoid(gap).item() / (gap.numel() - 1)
    return gap.item()
