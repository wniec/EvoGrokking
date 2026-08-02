"""The classical NEAT genome: one node = one neuron, plus the fixed recipe."""

import random

from helpers import reachable_all

from evogrokking.genome import Genome, Innovations
from evogrokking.hyperparams import Hyperparams


# --------------------------------------------------------------------------
# One node = one neuron
# --------------------------------------------------------------------------
def test_nodes_are_single_neurons():
    g = Genome.random(random.Random(0), n_inputs=5, n_outputs=3, n_hidden=4)
    for n in g.nodes:
        # A neuron carries an activation and nothing else -- no width, no kernel.
        assert set(vars(n)) == {"id", "activation"}
        assert n.activation in ("relu", "gelu", "tanh", "silu")


def test_dense_start_is_fully_connected():
    n_in, n_out, n_hid = 6, 3, 5
    g = Genome.dense(n_in, n_out, n_hid)
    assert len(g.hidden_ids()) == n_hid
    assert len(g.nodes) == n_hid + n_out  # hidden + output neurons carry genes
    # input->hidden, hidden->output and input->output are all present.
    edges = {(c.src, c.dst) for c in g.conns if c.enabled}
    assert len(edges) == n_in * n_hid + n_hid * n_out + n_in * n_out
    for i in g.input_ids():
        for h in g.hidden_ids():
            assert (i, h) in edges
        for o in g.output_ids():
            assert (i, o) in edges
    # Without direct wiring the input->output shortcuts are gone.
    nodirect = Genome.dense(n_in, n_out, n_hid, direct=False)
    assert nodirect.n_enabled() == n_in * n_hid + n_hid * n_out


def test_dense_start_is_exactly_one_hidden_layer():
    # The founding structure is always one parallel block of hidden neurons;
    # depth only appears later, through add-node mutations.
    from evogrokking.models import _levels

    g = Genome.dense(6, 2, 5)
    levels = _levels(g.conns, g.node_ids(), set(g.input_ids()))
    assert [len(lv) for lv in levels] == [6, 5, 2]


def test_key_convention_separates_inputs_outputs_hidden():
    g = Genome.dense(4, 2, 3)
    assert g.input_ids() == [-1, -2, -3, -4]
    assert g.output_ids() == [0, 1]
    assert all(h >= g.n_outputs for h in g.hidden_ids())


# --------------------------------------------------------------------------
# Structural mutation
# --------------------------------------------------------------------------
def test_genome_mutation_keeps_graph_acyclic():
    rng = random.Random(1)
    innov = Innovations(first_hidden=3)
    g = Genome.dense(5, 3, 4, innov=innov)
    for _ in range(60):
        g = g.mutate(rng, innov)
        for node in g.node_ids():
            assert node not in reachable_all(g.conns, node)


def test_add_node_deepens_the_network():
    # Splitting a connection inserts a neuron *between* its endpoints, so the
    # network can grow past the single hidden layer it started with.
    from evogrokking.models import _levels

    innov = Innovations(first_hidden=2)
    g = Genome.dense(3, 2, 0, innov=innov)  # input -> output only
    rng = random.Random(0)

    def depth(genome):
        return len(_levels(genome.conns, genome.node_ids(), set(genome.input_ids())))

    start = depth(g)
    for _ in range(3):
        g = g._add_node(rng, innov)
    assert depth(g) > start


def test_genome_dict_roundtrip():
    g = Genome.random(random.Random(3), n_mutations=10)
    restored = Genome.from_dict(g.as_dict())
    assert restored.nodes == g.nodes
    assert restored.conns == g.conns
    assert restored.n_inputs == g.n_inputs and restored.n_outputs == g.n_outputs


# --------------------------------------------------------------------------
# Hyperparameters are fixed, not evolved
# --------------------------------------------------------------------------
def test_hyperparams_are_not_part_of_the_genome():
    g = Genome.random(random.Random(0))
    for gene in ("lr", "weight_decay", "dropout", "optimizer", "init_scale"):
        assert not hasattr(g, gene)
    assert set(g.as_dict()) == {
        "id", "parents", "n_inputs", "n_outputs", "nodes", "conns"
    }


def test_hyperparams_overrides_and_roundtrip():
    hp = Hyperparams.for_task("image")
    assert hp.with_overrides(lr=None, dropout=None) == hp  # unset flags are ignored
    tuned = hp.with_overrides(lr=3e-4, optimizer="sgd")
    assert tuned.lr == 3e-4 and tuned.optimizer == "sgd"
    assert tuned.weight_decay == hp.weight_decay  # untouched
    assert Hyperparams.from_dict(tuned.as_dict()) == tuned
    assert Hyperparams.for_task("modular").weight_decay > hp.weight_decay
    assert not hasattr(hp, "embed_dim")  # embeddings are gone


def test_unknown_optimizer_is_rejected():
    try:
        Hyperparams(optimizer="rmsprop")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown optimizer")
