"""Learning-curve and network-structure rendering."""

import os
import random
import tempfile

from evogrokking import datasets
from evogrokking.genome import Genome
from evogrokking.plots import plot_curves, plot_genome


def _out(name):
    return os.path.join(tempfile.mkdtemp(), name)


def test_plot_curves_writes_png():
    out = _out("curves.png")
    n = 30
    plot_curves(
        [1.0 / (i + 1) for i in range(n)],
        [2.0 for _ in range(n)],
        [min(1.0, 0.03 * i) for i in range(n)],
        [0.1 for _ in range(n)],
        out, title="test", eval_every=5, grok_area=3.2,
    )
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_plot_genome_detailed_mode():
    # A small graph is drawn neuron by neuron.
    spec = datasets.DatasetSpec("t", "image", input_dim=8, num_classes=3)
    g = Genome.random(random.Random(21), n_inputs=8, n_outputs=3, n_hidden=4)
    out = _out("small.png")
    plot_genome(g, out, spec=spec, title="detailed")
    assert os.path.getsize(out) > 0


def test_plot_genome_collapses_large_graphs():
    # A classical-NEAT MNIST genome has 784 input neurons and ~50k edges, far
    # past what a node-and-arrow diagram can show, so levels are collapsed.
    big = Genome.dense(784, 10, 64)
    spec = datasets.DatasetSpec("t", "image", input_dim=784, num_classes=10)
    out = _out("big.png")
    plot_genome(big, out, spec=spec, title="collapsed")
    assert os.path.getsize(out) > 0


def test_detail_limit_selects_the_mode():
    # Forcing the limit low pushes even a small graph into collapsed mode; both
    # paths must render.
    spec = datasets.DatasetSpec("t", "image", input_dim=8, num_classes=3)
    g = Genome.dense(8, 3, 4)
    for limit, name in ((1000, "forced_detail.png"), (2, "forced_collapsed.png")):
        out = _out(name)
        plot_genome(g, out, spec=spec, detail_limit=limit)
        assert os.path.getsize(out) > 0
