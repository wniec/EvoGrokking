"""Fast, CPU-only smoke tests for the core pieces.

Run with:  python -m pytest -q   (or  python tests/test_evogrokking.py)
"""

import math
import random

import torch

from evogrokking import datasets
from evogrokking.genome import INPUT, OUTPUT, Genome, Innovations
from evogrokking.hyperparams import Hyperparams
from evogrokking.metrics import (
    LOSS_MAX,
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


# --------------------------------------------------------------------------
# Metrics + the (minimise-grokking) objective
# --------------------------------------------------------------------------
def _curves(n, val_acc_at, *, final_val_acc=0.99, val_loss_high=2.0):
    """Synthetic curves: trains immediately, generalises at step ``val_acc_at``."""
    train_loss = [math.exp(-0.2 * i) + 1e-3 for i in range(n)]
    train_acc = [1.0] * n
    val_acc = [0.05 if i < val_acc_at else final_val_acc for i in range(n)]
    val_loss = [
        (val_loss_high if i < val_acc_at else math.exp(-0.2 * (i - val_acc_at))) + 1e-3
        for i in range(n)
    ]
    return train_loss, val_loss, train_acc, val_acc


def test_score_prefers_early_generalisation_to_grokking():
    # The objective is *minimising* grokking: of two runs that both end up
    # accurate, the one that generalised sooner must win.
    n = 100
    early = grokking_metrics(*_curves(n, 5))
    late = grokking_metrics(*_curves(n, 70))

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
    good = grokking_metrics(*_curves(n, 5))

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
    late = grokking_metrics(*_curves(n, 70))
    # Turning up the reward for a small gap / early generalisation must raise the
    # score of any run that reaches the gate -- but by less for a late grokker.
    early = grokking_metrics(*_curves(n, 5))
    gain_early = early.score(gap_weight=2.0) - early.score(gap_weight=0.0)
    gain_late = late.score(gap_weight=2.0) - late.score(gap_weight=0.0)
    assert gain_early > gain_late
    # Below the accuracy gate the anti-grokking weights buy nothing at all.
    dead = grokking_metrics([2.3] * n, [2.3] * n, [0.1] * n, [0.1] * n)
    assert abs(dead.score(gap_weight=5.0) - dead.score(gap_weight=0.0)) < 1e-6


def test_acc_area_measures_accuracy_gap():
    n = 100
    train_acc = [1.0] * n  # memorises immediately
    # Validation stays at chance for the first half, then jumps to perfect.
    val_acc = [0.0 if i < 50 else 1.0 for i in range(n)]
    train_loss = [1e-3] * n
    val_loss = [2.0 if i < 50 else 0.01 for i in range(n)]
    m = grokking_metrics(train_loss, val_loss, train_acc, val_acc)
    # Gap is 1.0 for ~half the run -> area ~0.5.
    assert abs(m.acc_area - 0.5) < 0.02


def test_area_zero_when_no_gap():
    flat = [1.0] * 10
    assert area_between_log_losses(flat, flat) == 0.0


def test_losses_are_clamped_to_range():
    # A zero training loss and a diverged / NaN validation loss must not blow the
    # measure up to +/-inf: both are clamped into [LOSS_MIN, LOSS_MAX] first.
    train = [0.0, 0.0, 0.0]
    val = [1e30, float("nan"), float("inf")]
    m = grokking_metrics(train, val, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    assert math.isfinite(m.grok_area)
    assert m.grok_area <= math.log(LOSS_MAX) - math.log(1e-8) + 1e-6


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


def test_train_respects_early_stopping():
    ds = datasets.modular_addition(p=11, train_frac=0.6)
    genome = _simple_genome(Innovations())
    # Target trivially reached at the first evaluation -> stops at epoch 0.
    es = EarlyStopping(target_val_acc=0.0)
    result = train_and_evaluate(
        genome, ds, epochs=500, device=torch.device("cpu"), early_stopping=es, seed=0
    )
    assert result.stopped_epoch == 0
    assert len(result.train_losses) == 1


def _simple_genome(innov):
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
    )


# --------------------------------------------------------------------------
# Hyperparameters are fixed, not evolved
# --------------------------------------------------------------------------
def test_hyperparams_are_not_part_of_the_genome():
    # The whole point of the refactor: a genome carries architecture only.
    g = Genome.random(random.Random(0), Innovations(), n_mutations=6)
    for gene in ("lr", "weight_decay", "dropout", "optimizer", "init_scale", "embed_dim"):
        assert not hasattr(g, gene)
    assert set(g.as_dict()) == {"id", "parents", "conv", "nodes", "conns", "conv_pool"}


def test_hyperparams_overrides_and_roundtrip():
    hp = Hyperparams.for_task("image")
    # None-valued overrides are ignored, so unset CLI flags keep the defaults.
    assert hp.with_overrides(lr=None, dropout=None) == hp
    tuned = hp.with_overrides(lr=3e-4, optimizer="sgd")
    assert tuned.lr == 3e-4 and tuned.optimizer == "sgd"
    assert tuned.weight_decay == hp.weight_decay  # untouched
    assert Hyperparams.from_dict(tuned.as_dict()) == tuned
    # The modular task needs its own recipe to move at all.
    assert Hyperparams.for_task("modular").weight_decay > hp.weight_decay


def test_unknown_optimizer_is_rejected():
    try:
        Hyperparams(optimizer="rmsprop")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown optimizer")


# --------------------------------------------------------------------------
# Architecture genome
# --------------------------------------------------------------------------
def test_genome_mutation_keeps_graph_acyclic():
    rng = random.Random(1)
    innov = Innovations()
    g = Genome.random(rng, innov)
    for _ in range(80):
        g = g.mutate(rng, innov)
        # The full connection set must stay a DAG (no node reaches itself).
        for node in g.node_ids():
            assert node not in _reachable_all(g.conns, node)
        assert len(g.nodes) <= 12


def test_arbitrary_skip_connections_are_reachable():
    # Over enough mutations, some genome grows a direct input->output *and* a
    # multi-hop path, i.e. a genuine skip connection.
    rng = random.Random(7)
    innov = Innovations()
    found_skip = False
    for _ in range(40):
        g = Genome.random(rng, innov, n_mutations=10)
        enabled = {(c.src, c.dst) for c in g.conns if c.enabled}
        has_hidden_path = any(
            c.dst != OUTPUT for c in g.conns if c.enabled and c.src == INPUT
        )
        if (INPUT, OUTPUT) in enabled and has_hidden_path:
            found_skip = True
            break
    assert found_skip


def test_genome_dict_roundtrip():
    rng = random.Random(3)
    innov = Innovations()
    g = Genome.random(rng, innov, n_mutations=10)
    restored = Genome.from_dict(g.as_dict())
    assert restored.nodes == g.nodes
    assert restored.conns == g.conns
    assert restored.conv == g.conv


# --------------------------------------------------------------------------
# neat-python drives reproduction
# --------------------------------------------------------------------------
def _neat_population(pop_size=8, conv=False, seed=0):
    import os
    import tempfile

    import neat

    from evogrokking.neat_arch import ArchGenome, write_config

    path = write_config(
        os.path.join(tempfile.mkdtemp(), "neat.ini"), pop_size=pop_size, conv=conv
    )
    cfg = neat.Config(
        ArchGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        path,
    )
    return neat.Population(cfg, seed=seed), cfg


def test_neat_population_evolves_and_converts_to_genomes():
    from evogrokking.neat_arch import to_genome

    pop, _ = _neat_population(pop_size=10)
    rng = random.Random(0)

    def fitness(genomes, config):
        for _gid, g in genomes:
            g.fitness = rng.random()

    pop.run(fitness, 5)
    assert pop.best_genome is not None

    g = to_genome(pop.best_genome)
    ids = {n.id for n in g.nodes} | {INPUT, OUTPUT}
    # Every connection endpoint is a known node, and the graph stays acyclic.
    for c in g.conns:
        assert c.src in ids and c.dst in ids
    for node in ids:
        assert node not in _reachable_all(g.conns, node)
    # Widths and activations came across from the custom gene attributes.
    for n in g.nodes:
        assert n.width >= 1 and n.activation in ("relu", "gelu", "tanh", "silu")


def test_neat_conv_genomes_have_valid_kernels():
    from evogrokking.neat_arch import to_genome

    pop, _ = _neat_population(pop_size=10, conv=True, seed=3)
    rng = random.Random(1)
    pop.run(lambda gs, c: [setattr(g, "fitness", rng.random()) for _i, g in gs], 5)
    for ng in pop.population.values():
        g = to_genome(ng, conv=True, conv_pool=2)
        assert g.conv is True
        for n in g.nodes:
            # Only odd kernels are reachable -- the index gene guarantees it.
            assert n.kernel_size in (3, 5, 7)
            assert 4 <= n.width <= 32  # conv channel bounds


def test_neat_genome_builds_a_trainable_model():
    from evogrokking.neat_arch import to_genome

    pop, _ = _neat_population(pop_size=6, seed=5)
    rng = random.Random(2)
    pop.run(lambda gs, c: [setattr(g, "fitness", rng.random()) for _i, g in gs], 3)

    spec = datasets.DatasetSpec("t", "image", input_dim=20, num_classes=3)
    for ng in pop.population.values():
        model = build_model(to_genome(ng), spec)
        assert model(torch.randn(4, 20)).shape == (4, 3)


# --------------------------------------------------------------------------
# The distribution shift (Carvalho et al. 2025)
# --------------------------------------------------------------------------
def test_equation_1_matches_the_paper_worked_example():
    from evogrokking.subclasses import equation_1

    # Paper §4: 4 classes x 2 subclasses, gamma_D = 2000, f = 0.2, one subclass
    # per class subsampled -> gamma_s = 4, gamma_r = 4 -> s_s = 84, s_r = 416.
    s_s, s_r = equation_1(total=2000, n_shifted=4, n_kept=4, frac=0.2)
    assert (s_s, s_r) == (84, 416)
    assert 4 * s_s + 4 * s_r == 2000  # ...and the budget is respected


def test_equation_1_endpoints():
    from evogrokking.subclasses import equation_1

    # f = 1 is no shift at all: every subclass gets the same count.
    s_s, s_r = equation_1(total=800, n_shifted=4, n_kept=4, frac=1.0)
    assert s_s == s_r == 100
    # f = 0 removes the shifted subclasses entirely.
    s_s, s_r = equation_1(total=800, n_shifted=4, n_kept=4, frac=0.0)
    assert s_s == 0 and s_r == 200


def test_subsample_shifted_actually_shifts_the_distribution():
    from evogrokking.subclasses import subsample_shifted

    n_subclasses, num_classes, per_sub = 4, 10, 500
    # A balanced pool: every subclass has the same number of samples.
    sub_ids = torch.arange(num_classes * n_subclasses).repeat_interleave(per_sub)

    idx = subsample_shifted(
        sub_ids,
        total=2000,
        n_subclasses=n_subclasses,
        num_classes=num_classes,
        shifted_per_class=1,
        frac=0.05,
        seed=0,
    )
    counts = torch.bincount(sub_ids[idx], minlength=num_classes * n_subclasses)
    # Exactly one subclass per class is under-sampled...
    assert (counts == counts.min()).sum().item() == num_classes
    # ...and it really is much rarer than the rest.
    assert counts.min() * 5 < counts.max()
    assert len(idx) <= 2000 * 1.1

    # Classes themselves stay balanced -- the shift is *within* class, which is
    # what makes train and test differ in representation but not in label prior.
    per_class = torch.bincount(sub_ids[idx] // n_subclasses, minlength=num_classes)
    assert per_class.max() - per_class.min() <= 2


def test_subsample_shifted_can_remove_a_subclass_entirely():
    from evogrokking.subclasses import subsample_shifted

    sub_ids = torch.arange(8).repeat_interleave(100)
    idx = subsample_shifted(
        sub_ids, total=400, n_subclasses=2, num_classes=4,
        shifted_per_class=1, frac=0.0, seed=0,
    )
    counts = torch.bincount(sub_ids[idx], minlength=8)
    assert (counts == 0).sum().item() == 4  # one subclass of each class is gone


def test_kmeans_recovers_separated_clusters():
    from evogrokking.subclasses import _kmeans

    g = torch.Generator().manual_seed(0)
    a = torch.randn(50, 4, generator=g) + 10.0
    b = torch.randn(50, 4, generator=g) - 10.0
    assign = _kmeans(torch.cat([a, b]), k=2, seed=0)
    # Each true cluster ends up wholly inside one predicted cluster.
    assert len(assign[:50].unique()) == 1 and len(assign[50:].unique()) == 1
    assert assign[0] != assign[50]


def test_dataset_registry_exposes_shifted_and_plain():
    assert datasets.is_image("mnist") and not datasets.is_image("modadd")
    assert "mnist" in datasets.SHIFTED and "mnist_plain" not in datasets.SHIFTED
    try:
        datasets.load("nosuchdataset")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown dataset")


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
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


def test_model_uses_the_supplied_hyperparams():
    spec = datasets.DatasetSpec("t", "image", input_dim=8, num_classes=3)
    g = Genome.random(random.Random(4), Innovations(), n_mutations=6)
    plain = build_model(g, spec, Hyperparams(init_scale=1.0))
    scaled = build_model(g, spec, Hyperparams(init_scale=8.0))
    # init_scale is applied to the weights, so the scaled model starts far larger.
    n_plain = sum(p.abs().sum().item() for p in plain.parameters())
    n_scaled = sum(p.abs().sum().item() for p in scaled.parameters())
    assert n_scaled > 3 * n_plain
    assert scaled.hp.init_scale == 8.0


def _image_spec():
    return datasets.DatasetSpec(
        "img", "image", input_dim=64, num_classes=5, image_shape=(1, 8, 8)
    )


def test_conv_genome_builds_and_runs():
    rng = random.Random(11)
    innov = Innovations()
    spec = _image_spec()
    g = Genome.random(rng, innov, n_mutations=12, allow_conv=True)
    assert g.conv is True
    model = build_model(g, spec)
    assert model.is_conv is True
    x = torch.randn(4, 64)  # 4 flattened 8x8 images
    out = model(x)
    assert out.shape == (4, 5)
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
    spec = _image_spec()
    full = Genome.random(rng, Innovations(), n_mutations=10, allow_conv=True, conv_pool=1)
    pooled = Genome.from_dict({**full.as_dict(), "conv_pool": 2})

    # Pooling by 2 -> 4x fewer spatial elements -> ~4x less activation memory.
    mem_full = estimated_activation_mb(full, spec, batch_size=100)
    mem_pooled = estimated_activation_mb(pooled, spec, batch_size=100)
    assert mem_pooled < mem_full / 3.0

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
        mem_budget_mb=1e-9, epochs_per_eval=5, workers=1,  # absurdly low budget
    )
    evo = Evolution(ds, cfg, device=torch.device("cpu"))
    best = evo.run()
    # Every genome is over the tiny budget, so all are skipped (never trained).
    assert best.fitness == Evolution._OVER_BUDGET_FITNESS
    assert best.result is None


def test_evolution_runs_end_to_end():
    # A whole (tiny) search: neat-python reproduces, we train and score.
    import os
    import tempfile

    from evogrokking.datasets import Dataset, DatasetSpec
    from evogrokking.evolution import Evolution, EvolutionConfig

    spec = DatasetSpec("t", "image", input_dim=16, num_classes=3)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(40, 16, generator=g)
    y = torch.randint(0, 3, (40,), generator=g)
    ds = Dataset(spec, x, y, x.clone(), y.clone())

    out = tempfile.mkdtemp()
    cfg = EvolutionConfig(
        population_size=6, generations=3, epochs_per_eval=15, eval_every=5,
        workers=1, seed=0, mem_budget_mb=None,
    )
    evo = Evolution(
        ds, cfg, device=torch.device("cpu"),
        config_path=os.path.join(out, "neat_config.ini"),
    )
    best = evo.run()
    assert len(evo.history) == 3
    assert best.result is not None and best.fitness > float("-inf")
    # The generated neat-python config is kept alongside the run.
    assert os.path.exists(os.path.join(out, "neat_config.ini"))
    # Fitness is the minimise-grokking score of the best individual's run.
    assert abs(best.fitness - best.result.metrics.score()) < 1e-9


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

    cg = Genome.random(rng, Innovations(), n_mutations=10, allow_conv=True)
    cspec = _image_spec()
    out2 = os.path.join(tempfile.mkdtemp(), "structure_conv.png")
    plot_genome(cg, out2, spec=cspec)
    assert os.path.exists(out2) and os.path.getsize(out2) > 0


def test_train_command_writes_curves_and_structure_plots():
    # `train --plot` must produce both visualisations, not just the curves --
    # the same pair `retrain` writes for an evolved genome.
    import os
    import tempfile

    from evogrokking import experiment

    out_root = tempfile.mkdtemp()
    original_root = experiment.RUNS_ROOT
    experiment.RUNS_ROOT = out_root
    try:
        experiment.main(
            [
                "train", "--dataset", "modadd", "--p", "7",
                "--epochs", "6", "--eval-every", "2", "--plot", "--name", "viz",
            ]
        )
    finally:
        experiment.RUNS_ROOT = original_root

    run = os.path.join(out_root, "viz")
    for name in ("curves.png", "structure.png", "curves.json", "result.json"):
        path = os.path.join(run, name)
        assert os.path.exists(path) and os.path.getsize(path) > 0, name


def test_model_handles_node_without_incoming_edges():
    # A genome whose output has no incoming enabled edge (pure bias) must still
    # produce a correctly batched (batch, num_classes) output, not a 1-D bias.
    from evogrokking.genome import ConnGene

    innov = Innovations()
    c0 = innov.conn(INPUT, OUTPUT)
    genome = Genome(
        nodes=(),
        conns=(ConnGene(c0, INPUT, OUTPUT, False),),  # disabled -> output is const
    )
    spec = datasets.DatasetSpec("t", "image", input_dim=8, num_classes=4)
    out = build_model(genome, spec)(torch.randn(6, 8))
    assert out.shape == (6, 4)


def test_train_runs_and_returns_metrics():
    ds = datasets.modular_addition(p=11, train_frac=0.6)
    genome = _simple_genome(Innovations())
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
    genome = _simple_genome(Innovations())
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
