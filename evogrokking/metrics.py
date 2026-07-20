"""Quantifying grokking.

Grokking is *delayed generalisation*: the training loss collapses almost
immediately while the validation loss lingers high for a long time before it,
too, finally drops.  On a log-loss plot this shows up as a large, long-lived gap
between the two curves.

We measure the *scale* of grokking as the **area between
the train and validation log-loss curves**::

    grok_area = (1/T) * sum_t  max(0, log(val_loss_t) - log(train_loss_t))

The ``max(0, .)`` keeps the measure a "generalisation gap" (validation above
training) and the ``1/T`` normalisation makes runs of different length
comparable.

The area alone is **not** grokking, though -- and rewarding it directly makes the
evolutionary search collapse onto *plain overfitting*, which produces the
biggest, most permanent gap of all (validation loss high forever).  What
distinguishes grokking from overfitting is that in grokking the validation loss
*rises (or lingers) and then falls*, and the model *eventually generalises*.  So
:meth:`GrokkingMetrics.score` only credits the area once a run is **certified**
as grokking on two independent axes:

1. a sharp **generalisation gate** on the final validation accuracy (a soft step
   at ``gen_threshold``), so low-accuracy overfitters score ~0; and
2. a **val-loss-drop** factor -- how far the validation loss fell from its peak
   -- so a run whose validation loss never comes back down scores ~0 even if its
   accuracy is coincidentally acceptable.

Alongside the loss area we also measure the **area between the train and
validation accuracy curves** (``acc_area``).  It captures the very same
delayed-generalisation gap but is *bounded* in ``[0, 1]`` and cannot be inflated
by a diverging loss, so it is a robust second view; the objective blends the two.

We also report ``grok_delay`` (the lag between train and validation reaching
``gen_threshold``; zero if validation never gets there) and ``val_loss_drop`` so
overfitting is easy to spot in the logs.
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

    def score(
        self,
        test_weight: float = 0.1,
        gen_threshold: float = GEN_THRESHOLD,
        acc_weight: float = 5.0,
        sharpness: float = 30.0,
    ) -> float:
        """Composite fitness maximised by the evolutionary search.

        The grokking *magnitude* combines the two views of the delayed gap --
        the loss area and the bounded accuracy area (weighted by
        ``acc_weight``, which scales its ~[0, 0.5] range up to
        the loss area's range so both contribute)::

            magnitude = grok_area + acc_weight * acc_area

        That magnitude is credited *only* for runs certified as grokking rather
        than overfitting, on two independent axes:

        * ``gate``  -- a soft step on final validation accuracy at
          ``gen_threshold``; ~0 well below it, ~1 above.  This is what stops the
          search from being rewarded for plain overfitting.
        * ``drop``  -- ``1 - exp(-val_loss_drop)``; ~0 when the validation loss
          never falls from its peak, ~1 when it plateaus high then collapses.

        Final validation accuracy is added as a low-weight *supporting*
        objective.
        """
        gate = 1.0 / (1.0 + math.exp(-sharpness * (self.final_val_acc - gen_threshold)))
        drop = 1.0 - math.exp(-self.val_loss_drop)
        magnitude = self.grok_area + acc_weight * self.acc_area
        return gate * drop * magnitude + test_weight * self.final_val_acc

    def as_dict(self) -> dict:
        return {
            "grok_area": self.grok_area,
            "acc_area": self.acc_area,
            "grok_delay": self.grok_delay,
            "val_loss_drop": self.val_loss_drop,
            "generalised": self.generalised,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "final_train_acc": self.final_train_acc,
            "final_val_acc": self.final_val_acc,
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

    # Delay between train and validation reaching the generalisation threshold.
    n = len(train_accs)
    i_train = _first_cross(train_accs, gen_threshold)
    i_val = _first_cross(val_accs, gen_threshold)
    if i_train is not None and i_val is not None and i_val >= i_train and n > 1:
        grok_delay = (i_val - i_train) / (n - 1)
    else:
        grok_delay = 0.0

    return GrokkingMetrics(
        grok_area=float(area),
        acc_area=float(acc_area),
        grok_delay=float(grok_delay),
        val_loss_drop=val_loss_drop,
        generalised=i_val is not None,
        final_train_loss=float(train_losses[-1]),
        final_val_loss=float(val_losses[-1]),
        final_train_acc=float(train_accs[-1]),
        final_val_acc=float(val_accs[-1]),
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
