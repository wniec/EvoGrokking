"""Grokking measurement and the minimise-grokking objective."""

import math

from helpers import curves

from evogrokking.metrics import LOSS_MAX, area_between_log_losses, grokking_metrics


def test_score_prefers_early_generalisation_to_grokking():
    # The objective is *minimising* grokking: of two runs that both end up
    # accurate, the one that generalised sooner must win.
    n = 100
    early = grokking_metrics(*curves(n, 5))
    late = grokking_metrics(*curves(n, 70))

    assert early.final_val_acc == late.final_val_acc  # equally accurate at the end
    assert early.gen_frac < late.gen_frac  # ...but generalised much sooner
    assert early.acc_area < late.acc_area  # ...with a far smaller gap
    assert early.score() > late.score()
    # The late run really did grok more, by the classic magnitude measure.
    assert late.grok_magnitude() > early.grok_magnitude()


def test_score_rejects_both_degenerate_non_grokkers():
    # "Not grokking" must not be reachable by refusing to learn.  Both degenerate
    # strategies sit below the accuracy gate and must lose to a real solution.
    n = 100
    good = grokking_metrics(*curves(n, 5))

    # (a) memorises and never generalises: a permanent gap.
    overfit = grokking_metrics(
        [1e-3] * n, [2.0 + 0.05 * i for i in range(n)], [1.0] * n, [0.3] * n
    )
    # (b) never learns anything at all: no gap, but no accuracy either.
    dead = grokking_metrics([2.3] * n, [2.3] * n, [0.1] * n, [0.1] * n)

    assert overfit.score() < good.score()
    assert dead.score() < good.score()
    assert dead.generalised is False and overfit.generalised is False


def test_gen_frac_is_one_when_never_generalising():
    n = 40
    m = grokking_metrics([1e-3] * n, [3.0] * n, [1.0] * n, [0.2] * n)
    assert m.gen_frac == 1.0  # maximally late, by convention
    assert m.generalised is False
    assert m.grok_delay == 0.0
    assert m.val_loss_drop == 0.0


def test_objective_weights_have_the_expected_sign():
    n = 100
    late = grokking_metrics(*curves(n, 70))
    early = grokking_metrics(*curves(n, 5))
    gain_early = early.score(gap_weight=2.0) - early.score(gap_weight=0.0)
    gain_late = late.score(gap_weight=2.0) - late.score(gap_weight=0.0)
    assert gain_early > gain_late
    # Below the accuracy gate the anti-grokking weights buy nothing at all.
    dead = grokking_metrics([2.3] * n, [2.3] * n, [0.1] * n, [0.1] * n)
    assert abs(dead.score(gap_weight=5.0) - dead.score(gap_weight=0.0)) < 1e-6


def test_score_defaults_to_the_threshold_it_was_measured_with():
    # A score reported next to a run must use that run's own gate, not the
    # module default -- otherwise printed fitness and printed score disagree.
    m = grokking_metrics(*curves(100, 5), gen_threshold=0.75)
    assert m.gen_threshold == 0.75
    assert m.score() == m.score(gen_threshold=0.75)
    assert m.as_dict()["score"] == m.score()


def test_acc_area_measures_accuracy_gap():
    n = 100
    train_acc = [1.0] * n  # memorises immediately
    val_acc = [0.0 if i < 50 else 1.0 for i in range(n)]
    train_loss = [1e-3] * n
    val_loss = [2.0 if i < 50 else 0.01 for i in range(n)]
    m = grokking_metrics(train_loss, val_loss, train_acc, val_acc)
    assert abs(m.acc_area - 0.5) < 0.02  # gap is 1.0 for ~half the run


def test_area_zero_when_no_gap():
    flat = [1.0] * 10
    assert area_between_log_losses(flat, flat) == 0.0


def test_losses_are_clamped_to_range():
    # A zero training loss and a diverged / NaN validation loss must not send the
    # measure to +/-inf: both are clamped into [LOSS_MIN, LOSS_MAX] first.
    train = [0.0, 0.0, 0.0]
    val = [1e30, float("nan"), float("inf")]
    m = grokking_metrics(train, val, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    assert math.isfinite(m.grok_area)
    assert m.grok_area <= math.log(LOSS_MAX) - math.log(1e-8) + 1e-6
