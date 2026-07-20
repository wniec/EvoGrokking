"""Fast, CPU-only smoke tests for the core pieces.

Run with:  python -m pytest -q   (or  python tests/test_evogrokking.py)
"""

import math
import random

import torch

from evogrokking import datasets
from evogrokking.genome import INPUT, OUTPUT, Genome, Innovations
from evogrokking.metrics import (
    LOSS_MAX,
    LOSS_MIN,
    area_between_log_losses,
    grokking_metrics,
)
from evogrokking.models import build_model
from evogrokking.train import EarlyStopping, train_and_evaluate


def _reachable_all(conns, start):
    adj = {}
    for c in conns:
        adj.setdefault(c.src, []).append(c.dst)
    seen, stack = set(), [start]
    while stack:
        n = stack.pop()
        for m in adj.get(n, ()):
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return seen


def test_score_prefers_grokking_over_overfitting():
    epochs = 100
    train = [math.exp(-0.2 * i) + 1e-3 for i in range(epochs)]

    # Grokking: val loss high then collapses, val acc reaches ~1 late.
    grok_val = [(2.0 if i < 60 else math.exp(-0.2 * (i - 60))) + 1e-3 for i in range(epochs)]
    grok_val_acc = [0.05 if i < 60 else 0.99 for i in range(epochs)]
    grok = grokking_metrics(train, grok_val, [1.0] * epochs, grok_val_acc)

    # Overfitting: val loss high forever (biggest area of all!), val acc never
    # climbs.  It must NOT beat grokking, despite the larger raw area.
    over_val = [2.0 + 0.05 * i for i in range(epochs)]  # even rising
    over_val_acc = [0.3] * epochs
    over = grokking_metrics(train, over_val, [1.0] * epochs, over_val_acc)

    assert over.grok_area > grok.grok_area  # overfitting really does have more area
    assert over.generalised is False and grok.generalised is True
    assert over.val_loss_drop < 0.5 and grok.val_loss_drop > 2.0
    assert over.score() < 0.05 * grok.score()  # ...yet scores essentially nothing
    # The accuracy area sees the same delayed gap and is bounded in [0, 1].
    assert 0.0 <= grok.acc_area <= 1.0 and grok.acc_area > 0.2


def test_acc_area_measures_accuracy_gap():
    n = 100
    train_acc = [1.0] * n  # memorises immediately
    # Validation stays at chance for the first half, then jumps to perfect.
    val_acc = [0.0 if i < 50 else 1.0 for i in range(n)]
    train_loss = [1e-3] * n
    # Val loss high then collapses, so the run certifies as grokking (drop > 0).
    val_loss = [2.0 if i < 50 else 0.01 for i in range(n)]
    m = grokking_metrics(train_loss, val_loss, train_acc, val_acc)
    # Gap is 1.0 for ~half the run -> area ~0.5.
    assert abs(m.acc_area - 0.5) < 0.02
    # acc_weight scales it into the score; acc_weight=0 removes it.
    with_acc = m.score(acc_weight=5.0)
    without_acc = m.score(acc_weight=0.0)
    assert with_acc > without_acc


def test_overfitting_metrics_flagged():
    n = 50
    train = [1e-3] * n
    val = [3.0] * n  # never recovers
    m = grokking_metrics(train, val, [1.0] * n, [0.2] * n)
    assert m.generalised is False
    assert m.grok_delay == 0.0
    assert m.val_loss_drop == 0.0


def test_area_zero_when_no_gap():
    flat = [1.0] * 10
    assert area_between_log_losses(flat, flat) == 0.0


def test_losses_are_clamped_to_range():
    # A zero training loss and a diverged / NaN validation loss must not blow the
    # measure up to +/-inf: both are clamped into [LOSS_MIN, LOSS_MAX] first.
    import math

    train = [0.0, 0.0, 0.0]
    val = [1e30, float("nan"), float("inf")]
    m = grokking_metrics(train, val, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    assert math.isfinite(m.grok_area)
    # The gap per point is bounded by log(LOSS_MAX) - log(LOSS_MIN).
    assert m.grok_area <= math.log(LOSS_MAX) - math.log(LOSS_MIN) + 1e-6


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


def test_train_respects_early_stopping():
    ds = datasets.modular_addition(p=11, train_frac=0.6)
    genome = _simple_genome(Innovations(), random.Random(0))
    # Target trivially reached at the first evaluation -> stops at epoch 0.
    es = EarlyStopping(target_val_acc=0.0)
    result = train_and_evaluate(
        genome, ds, epochs=500, device=torch.device("cpu"), early_stopping=es, seed=0
    )
    assert result.stopped_epoch == 0
    assert len(result.train_losses) == 1


def _simple_genome(innov, rng):
    """A minimal input->hidden->output graph, for training tests."""
    from evogrokking.genome import ConnGene, NodeGene

    c0 = innov.conn(INPUT, OUTPUT)
    hid = innov.split_node(c0)
    return Genome(
        nodes=(NodeGene(hid, 64, "relu"),),
        conns=(
            ConnGene(c0, INPUT, OUTPUT, False),
            ConnGene(innov.conn(INPUT, hid), INPUT, hid, True),
            ConnGene(innov.conn(hid, OUTPUT), hid, OUTPUT, True),
        ),
        embed_dim=32, weight_decay=1.0, dropout=0.0,
        optimizer="adamw", lr=1e-2, init_scale=1.0,
    )


def test_genome_mutation_keeps_graph_acyclic():
    rng = random.Random(1)
    innov = Innovations()
    g = Genome.random(rng, innov)
    for _ in range(80):
        g = g.mutate(rng, innov)
        # The full connection set must stay a DAG (no node reaches itself).
        for node in g.node_ids():
            assert node not in _reachable_all(g.conns, node)
        # Every hidden node referenced by a connection has a NodeGene.
        assert len(g.nodes) <= 12
        assert g.weight_decay > 0 and g.lr > 0
        assert g.optimizer in ("adam", "adamw", "sgd")


def test_arbitrary_skip_connections_are_reachable():
    # Over enough mutations, some genome grows a direct input->output *and* a
    # multi-hop path, i.e. a genuine skip connection -- something the old
    # layer-list genome could not represent.
    rng = random.Random(7)
    innov = Innovations()
    found_skip = False
    for _ in range(40):
        g = Genome.random(rng, innov, n_mutations=10)
        enabled = {(c.src, c.dst) for c in g.conns if c.enabled}
        has_hidden_path = any(c.dst != OUTPUT for c in g.conns if c.enabled and c.src == INPUT)
        if (INPUT, OUTPUT) in enabled and has_hidden_path:
            found_skip = True
            break
    assert found_skip


def test_crossover_produces_valid_child():
    rng = random.Random(2)
    innov = Innovations()
    a = Genome.random(rng, innov, n_mutations=8)
    b = Genome.random(rng, innov, n_mutations=8)
    child = Genome.crossover(a, b, rng)
    ref = {n.id for n in child.nodes} | {INPUT, OUTPUT}
    # Every connection endpoint is a known node.
    for c in child.conns:
        assert c.src in ref and c.dst in ref


def test_genome_dict_roundtrip():
    rng = random.Random(3)
    innov = Innovations()
    g = Genome.random(rng, innov, n_mutations=10)
    restored = Genome.from_dict(g.as_dict())
    assert restored.nodes == g.nodes
    assert restored.conns == g.conns
    assert restored.weight_decay == g.weight_decay
    assert restored.optimizer == g.optimizer


def test_plot_curves_writes_png(tmp_path=None):
    import os
    import tempfile

    from evogrokking.plots import plot_curves

    d = tmp_path or tempfile.mkdtemp()
    out = os.path.join(str(d), "curves.png")
    n = 30
    plot_curves(
        [1.0 / (i + 1) for i in range(n)],
        [2.0 for _ in range(n)],
        [min(1.0, 0.03 * i) for i in range(n)],
        [0.1 for _ in range(n)],
        out,
        title="test",
        eval_every=5,
        grok_area=3.2,
    )
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_model_builds_for_both_task_types():
    rng = random.Random(2)
    innov = Innovations()
    modadd = datasets.modular_addition(p=7, train_frac=0.5)
    model = build_model(Genome.random(rng, innov, n_mutations=8), modadd.spec)
    out = model(modadd.x_train)
    assert out.shape == (len(modadd.x_train), 7)

    mnist_spec = datasets.DatasetSpec("t", "image", input_dim=20, num_classes=3)
    img_model = build_model(Genome.random(rng, innov, n_mutations=8), mnist_spec)
    x = torch.randn(5, 20)
    assert img_model(x).shape == (5, 3)


def _image_spec():
    return datasets.DatasetSpec(
        "img", "image", input_dim=64, num_classes=5, image_shape=(1, 8, 8)
    )


def test_conv_genome_builds_and_runs():
    rng = random.Random(11)
    innov = Innovations()
    spec = _image_spec()
    # A convolutional genome grown by several structural mutations.
    g = Genome.random(rng, innov, n_mutations=12, allow_conv=True)
    assert g.conv is True
    model = build_model(g, spec)
    assert model.is_conv is True
    x = torch.randn(4, 64)  # 4 flattened 8x8 images
    out = model(x)
    assert out.shape == (4, 5)
    # Convolutions really are in there (not just linear layers).
    assert any(isinstance(m, torch.nn.Conv2d) for m in model.modules())


def test_conv_nodes_carry_kernels_and_channels():
    rng = random.Random(5)
    innov = Innovations()
    g = Genome.random(rng, innov, n_mutations=15, allow_conv=True)
    for node in g.nodes:
        assert node.kernel_size in (3, 5, 7)
        assert 4 <= node.width <= 64  # conv channel bounds


def test_conv_flag_survives_dict_roundtrip():
    rng = random.Random(9)
    innov = Innovations()
    g = Genome.random(rng, innov, n_mutations=10, allow_conv=True)
    restored = Genome.from_dict(g.as_dict())
    assert restored.conv is True
    assert restored.nodes == g.nodes


def test_conv_pool_reduces_resolution_and_memory():
    from evogrokking.models import estimated_activation_mb

    rng = random.Random(4)
    spec = datasets.DatasetSpec("img", "image", input_dim=64, num_classes=5, image_shape=(1, 8, 8))
    full = Genome.random(rng, Innovations(), n_mutations=10, allow_conv=True, conv_pool=1)
    pooled = Genome.from_dict({**full.as_dict(), "conv_pool": 2})

    # Pooling by 2 -> 4x fewer spatial elements -> ~4x less activation memory.
    mem_full = estimated_activation_mb(full, spec, batch_size=100)
    mem_pooled = estimated_activation_mb(pooled, spec, batch_size=100)
    assert mem_pooled < mem_full / 3.0

    # The pooled model actually runs at the reduced resolution.
    model = build_model(pooled, spec)
    assert model.eff_hw == (4, 4)
    assert model(torch.randn(3, 64)).shape == (3, 5)


def test_memory_budget_skips_oversized_genomes():
    from evogrokking.datasets import Dataset, DatasetSpec
    from evogrokking.evolution import Evolution, EvolutionConfig

    spec = DatasetSpec("img", "image", input_dim=64, num_classes=3, image_shape=(1, 8, 8))
    x = torch.randn(30, 64)
    y = torch.randint(0, 3, (30,))
    ds = Dataset(spec, x, y, x.clone(), y.clone())
    cfg = EvolutionConfig(
        population_size=4, generations=1, allow_conv=True, conv_pool=1,
        mem_budget_mb=1e-6, epochs_per_eval=5, workers=1,  # absurdly low budget
    )
    evo = Evolution(ds, cfg, device=torch.device("cpu"))
    best = evo.run()
    # Every genome is over the tiny budget, so all are skipped (never trained).
    assert best.fitness == Evolution._OVER_BUDGET_FITNESS
    assert best.result is None


def test_conv_train_runs():
    # End-to-end: a tiny conv net trains on a fake image dataset without error.
    spec = _image_spec()
    x_tr = torch.randn(40, 64)
    y_tr = torch.randint(0, 5, (40,))
    ds = datasets.Dataset(spec, x_tr, y_tr, x_tr.clone(), y_tr.clone())
    rng = random.Random(3)
    g = Genome.random(rng, Innovations(), n_mutations=10, allow_conv=True)
    result = train_and_evaluate(g, ds, epochs=20, device=torch.device("cpu"), seed=0)
    assert result.n_params > 0
    assert len(result.train_losses) > 0


def test_plot_genome_writes_png():
    import os
    import tempfile

    from evogrokking.plots import plot_genome

    rng = random.Random(21)
    g = Genome.random(rng, Innovations(), n_mutations=10)
    spec = datasets.DatasetSpec("t", "image", input_dim=64, num_classes=4)
    out = os.path.join(tempfile.mkdtemp(), "structure.png")
    plot_genome(g, out, spec=spec, title="test")
    assert os.path.exists(out) and os.path.getsize(out) > 0

    # Also works for a conv genome (labels include channels/kernels).
    cg = Genome.random(rng, Innovations(), n_mutations=10, allow_conv=True)
    cspec = datasets.DatasetSpec("t", "image", input_dim=64, num_classes=4, image_shape=(1, 8, 8))
    out2 = os.path.join(tempfile.mkdtemp(), "structure_conv.png")
    plot_genome(cg, out2, spec=cspec)
    assert os.path.exists(out2) and os.path.getsize(out2) > 0


def test_model_handles_node_without_incoming_edges():
    # A genome whose output has no incoming enabled edge (pure bias) must still
    # produce a correctly batched (batch, num_classes) output, not a 1-D bias.
    from evogrokking.genome import ConnGene

    innov = Innovations()
    c0 = innov.conn(INPUT, OUTPUT)
    genome = Genome(
        nodes=(),
        conns=(ConnGene(c0, INPUT, OUTPUT, False),),  # disabled -> output is const
        embed_dim=16, weight_decay=0.0, dropout=0.0,
        optimizer="adam", lr=1e-3, init_scale=1.0,
    )
    spec = datasets.DatasetSpec("t", "image", input_dim=8, num_classes=4)
    out = build_model(genome, spec)(torch.randn(6, 8))
    assert out.shape == (6, 4)


def test_train_runs_and_returns_metrics():
    ds = datasets.modular_addition(p=11, train_frac=0.6)
    genome = _simple_genome(Innovations(), random.Random(0))
    result = train_and_evaluate(
        genome, ds, epochs=50, device=torch.device("cpu"), seed=0
    )
    assert len(result.train_losses) > 0
    assert result.metrics.final_train_acc >= 0.0
    assert result.n_params > 0
    assert result.seed == 0  # the training seed is recorded for reproduction


def test_same_seed_reproduces_run():
    # The whole point of the fixed seed: two runs of the same genome with the
    # same seed are bit-for-bit identical (so retrain reproduces the search).
    ds = datasets.modular_addition(p=11, train_frac=0.6)
    genome = _simple_genome(Innovations(), random.Random(0))
    a = train_and_evaluate(genome, ds, epochs=60, device=torch.device("cpu"), seed=123)
    b = train_and_evaluate(genome, ds, epochs=60, device=torch.device("cpu"), seed=123)
    assert a.train_losses == b.train_losses
    assert a.val_losses == b.val_losses
    # A longer retrain shares the identical prefix with the shorter search run.
    longer = train_and_evaluate(
        genome, ds, epochs=120, device=torch.device("cpu"), seed=123
    )
    assert longer.train_losses[: len(a.train_losses)] == a.train_losses
    # A different seed gives a different trajectory.
    other = train_and_evaluate(genome, ds, epochs=60, device=torch.device("cpu"), seed=7)
    assert other.train_losses != a.train_losses


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
