"""Compiling a per-neuron genome into masked dense matmuls."""

import random
from dataclasses import replace

import torch
from helpers import per_neuron_reference

from evogrokking import datasets
from evogrokking.genome import ConnGene, Genome, Innovations, NodeGene
from evogrokking.hyperparams import Hyperparams
from evogrokking.models import _levels, build_model


def _spec(input_dim=6, num_classes=3):
    return datasets.DatasetSpec("t", "image", input_dim=input_dim, num_classes=num_classes)


# --------------------------------------------------------------------------
# The core correctness claim
# --------------------------------------------------------------------------
def test_masked_dense_matches_per_neuron_evaluation():
    # Compiling the sparse per-neuron graph into masked dense matmuls must not
    # change what it computes.
    hp = Hyperparams(init_scale=1.0, dropout=0.0)
    torch.manual_seed(0)
    for seed in range(8):
        g = Genome.random(random.Random(seed), n_inputs=6, n_outputs=3, n_hidden=4,
                          n_mutations=10)
        model = build_model(g, _spec(), hp).eval()
        x = torch.randn(5, 6)
        with torch.no_grad():
            diff = (model(x) - per_neuron_reference(g, model, x)).abs().max().item()
        assert diff < 1e-5, f"seed {seed}: masked dense != per-neuron graph ({diff})"


# --------------------------------------------------------------------------
# The level invariant the scheme rests on
# --------------------------------------------------------------------------
def test_source_neuron_is_computed_before_what_it_feeds():
    # A neuron with no incoming edges is a learned constant, but it still has to
    # land in an *earlier* level than its consumers.  Placing it alongside them
    # would put an edge inside a level, where the mask cannot represent it -- the
    # connection would be silently dropped.
    innov = Innovations(first_hidden=1)
    a, b = innov.new_node(), innov.new_node()
    g = Genome(
        nodes=(NodeGene(0, "relu"), NodeGene(a, "relu"), NodeGene(b, "relu")),
        conns=(
            ConnGene(innov.conn(-1, b), -1, b, True),
            ConnGene(innov.conn(a, b), a, b, True),  # a has no incoming edges
            ConnGene(innov.conn(b, 0), b, 0, True),
        ),
        n_inputs=1,
        n_outputs=1,
    )
    levels = _levels(g.conns, g.node_ids(), set(g.input_ids()))
    assert not any(a in lv and b in lv for lv in levels), "a->b edge inside a level"

    model = build_model(g, _spec(input_dim=1, num_classes=1))
    live = sum(getattr(model, f"mask_{i}").sum().item() for i in range(len(model.levels)))
    assert live == g.n_enabled()  # every enabled edge survives into the masks


def test_no_enabled_edge_ever_lives_inside_a_level():
    # Checked over genomes with connections randomly disabled, which manufactures
    # neurons with no incoming edges, dead ends and isolated sources.
    spec = _spec(input_dim=5, num_classes=3)
    for seed in range(40):
        rng = random.Random(seed)
        g = Genome.random(rng, n_inputs=5, n_outputs=3, n_hidden=4, n_mutations=12)
        g = replace(
            g, conns=tuple(replace(c, enabled=(rng.random() > 0.35)) for c in g.conns)
        )
        model = build_model(g, spec)

        level_of = {n: i for i, lv in enumerate(model.levels) for n in lv}
        for n in g.input_ids():
            level_of[n] = -1

        represented = 0
        for c in g.conns:
            if not c.enabled or c.src not in level_of or c.dst not in level_of:
                continue
            assert level_of[c.src] < level_of[c.dst], f"seed {seed}: edge within a level"
            represented += 1
        in_masks = sum(
            getattr(model, f"mask_{i}").sum().item() for i in range(len(model.levels))
        )
        assert in_masks == represented, f"seed {seed}: {represented - in_masks} edges lost"


def test_disabled_connections_carry_no_weight():
    # A disabled gene must be exactly zero in the mask, so no gradient flows and
    # the connection genuinely does not exist in the trained network.
    g = Genome.dense(4, 2, 3)
    disabled = replace(
        g,
        conns=tuple(
            replace(c, enabled=False) if i % 2 == 0 else c
            for i, c in enumerate(g.conns)
        ),
    )
    model = build_model(disabled, _spec(input_dim=4, num_classes=2),
                        Hyperparams(init_scale=1.0))
    live = sum(getattr(model, f"mask_{i}").sum().item() for i in range(len(model.levels)))
    assert live == disabled.n_enabled()


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------
def test_model_builds_for_both_task_types():
    modadd = datasets.modular_addition(p=7, train_frac=0.5)
    model = build_model(Genome.dense(modadd.spec.input_dim, 7, 8), modadd.spec)
    assert model(modadd.x_train).shape == (len(modadd.x_train), 7)

    img = build_model(Genome.dense(20, 3, 6), _spec(input_dim=20, num_classes=3))
    assert img(torch.randn(5, 20)).shape == (5, 3)


def test_model_uses_the_supplied_hyperparams():
    g = Genome.dense(8, 3, 5)
    spec = _spec(input_dim=8, num_classes=3)
    plain = build_model(g, spec, Hyperparams(init_scale=1.0))
    scaled = build_model(g, spec, Hyperparams(init_scale=8.0))
    n_plain = sum(p.abs().sum().item() for p in plain.parameters())
    n_scaled = sum(p.abs().sum().item() for p in scaled.parameters())
    assert n_scaled > 3 * n_plain
    assert scaled.hp.init_scale == 8.0


def test_model_handles_output_without_incoming_edges():
    # An output neuron with no path from the input must still emit a logit (its
    # bias), so every class always gets a score.
    g = Genome.dense(4, 3, 2)
    dead = replace(g, conns=tuple(replace(c, enabled=False) for c in g.conns))
    out = build_model(dead, _spec(input_dim=4, num_classes=3))(torch.randn(6, 4))
    assert out.shape == (6, 3)


def test_pruned_neurons_cost_nothing():
    # A hidden neuron with no route to an output is not built at all.
    g = Genome.dense(4, 2, 3)
    hid = g.hidden_ids()[0]
    cut = replace(
        g, conns=tuple(replace(c, enabled=False) if c.src == hid else c for c in g.conns)
    )
    model = build_model(cut, _spec(input_dim=4, num_classes=2))
    assert hid not in model.col
