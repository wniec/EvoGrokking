"""Shared test utilities.

Imported as a plain module (``from helpers import ...``): pytest puts the tests
directory on ``sys.path``, and there is no ``__init__.py`` here on purpose.
"""

from __future__ import annotations

import math

import torch

_ACT = {
    "relu": torch.relu,
    "gelu": torch.nn.functional.gelu,
    "tanh": torch.tanh,
    "silu": torch.nn.functional.silu,
}


def reachable_all(conns, start):
    """Every node reachable from ``start`` over *all* edges (enabled or not)."""
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


def curves(n, val_acc_at, *, final_val_acc=0.99, val_loss_high=2.0):
    """Synthetic run: trains immediately, generalises at step ``val_acc_at``.

    Returns ``(train_losses, val_losses, train_accs, val_accs)`` ready to hand to
    :func:`evogrokking.metrics.grokking_metrics`.
    """
    train_loss = [math.exp(-0.2 * i) + 1e-3 for i in range(n)]
    train_acc = [1.0] * n
    val_acc = [0.05 if i < val_acc_at else final_val_acc for i in range(n)]
    val_loss = [
        (val_loss_high if i < val_acc_at else math.exp(-0.2 * (i - val_acc_at))) + 1e-3
        for i in range(n)
    ]
    return train_loss, val_loss, train_acc, val_acc


def per_neuron_reference(genome, model, x):
    """Evaluate a genome one neuron at a time -- the definition of the graph.

    Deliberately independent of :meth:`GrokNet.forward`: it walks the connection
    genes directly, so it can contradict the masked-matmul compilation rather
    than merely echo it.
    """
    vals = {-i - 1: x[:, i] for i in range(genome.n_inputs)}
    incoming = {}
    for c in genome.conns:
        if c.enabled:
            incoming.setdefault(c.dst, []).append(c.src)
    acts = genome.activations()
    todo = [n.id for n in genome.nodes if n.id in model.col]
    while todo:
        progressed = False
        for nid in list(todo):
            srcs = [s for s in incoming.get(nid, []) if s in model.col]
            if not all(s in vals for s in srcs):
                continue
            li = next(i for i, lv in enumerate(model.levels) if nid in lv)
            j = model.levels[li].index(nid)
            w = (model.weights[li] * getattr(model, f"mask_{li}"))[:, j]
            pre = model.biases[li][j].expand(x.shape[0]).clone()
            for s in srcs:
                pre = pre + vals[s] * w[model.col[s]]
            vals[nid] = pre if nid < genome.n_outputs else _ACT[acts[nid]](pre)
            todo.remove(nid)
            progressed = True
        if not progressed:
            break
    return torch.stack([vals[o] for o in genome.output_ids()], dim=1)


def neat_population(pop_size=8, n_inputs=6, n_outputs=3, n_hidden=4, seed=0):
    """A neat-python population wired up the way the search wires it."""
    import os
    import tempfile

    import neat

    from evogrokking.neat_arch import ArchGenome, write_config

    path = write_config(
        os.path.join(tempfile.mkdtemp(), "neat.ini"),
        pop_size=pop_size,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_hidden=n_hidden,
    )
    cfg = neat.Config(
        ArchGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        path,
    )
    return neat.Population(cfg, seed=seed), cfg


def random_fitness(rng):
    """A fitness callback that scores genomes at random (structure probing)."""

    def assign(genomes, config):
        for _gid, genome in genomes:
            genome.fitness = rng.random()

    return assign
