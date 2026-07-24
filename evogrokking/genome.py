"""The evolving genome -- a NEAT-style directed graph describing an *architecture*.

This module holds the **phenotype description**: the graph that
:mod:`evogrokking.models` turns into a trainable network.  Reproduction itself
(mutation, crossover, speciation, innovation bookkeeping) is delegated to the
`neat-python <https://neat-python.readthedocs.io>`_ library -- see
:mod:`evogrokking.neat_arch`, which converts a ``neat.DefaultGenome`` into the
:class:`Genome` defined here.

Only the *architecture* is evolved.  The training recipe (learning rate, weight
decay, dropout, optimizer, init scale, embedding width) is **fixed** for a whole
run and lives in :class:`evogrokking.hyperparams.Hyperparams`; it is deliberately
not part of the genome, so the search cannot trade architecture quality against
regularisation tricks.

A genome is:

* ``nodes``  -- the hidden :class:`NodeGene`\\s (each with a width + activation,
  plus a kernel size in conv mode).  The input node (id ``INPUT=0``) and output
  node (id ``OUTPUT=1``) are implicit; their sizes come from the dataset.
* ``conns``  -- the :class:`ConnGene`\\s: directed ``src -> dst`` edges, each with
  an ``enabled`` flag and a global ``innovation`` number.

Structural invariant: the full connection set is always a DAG (cycle-free), so any
enabled subset is feed-forward and can be evaluated in topological order.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace

ACTIVATIONS = ("relu", "gelu", "tanh", "silu")

INPUT = 0
OUTPUT = 1

# Bounds keep evolution inside a sane, trainable region.
MIN_WIDTH, MAX_WIDTH = 8, 512
MAX_HIDDEN = 12

# Convolutional mode (image tasks): a hidden node is a spatial feature map with
# this many channels (``width`` doubles as channel count) and one of these
# same-padding kernel sizes.  Channels *and* the number of conv nodes are kept
# small because same-conv preserves the spatial resolution, so activation memory
# grows as batch x channels x H x W x nodes.
CONV_CH = (4, 32)
MAX_HIDDEN_CONV = 6
KERNEL_SIZES = (3, 5, 7)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# --------------------------------------------------------------------------
# Innovation bookkeeping -- shared across a whole run so that homologous
# structures in different genomes get the same ids.  neat-python keeps its own
# tracker during evolution; this one backs the hand-built baseline architectures
# and the random architectures used in tests.
# --------------------------------------------------------------------------
class Innovations:
    """Hands out globally-consistent connection innovation numbers and node ids."""

    def __init__(self) -> None:
        self._conn: dict[tuple[int, int], int] = {}
        self._split: dict[int, int] = {}
        self._next_conn = 0
        self._next_node = OUTPUT + 1  # 0 = input, 1 = output are reserved

    def conn(self, src: int, dst: int) -> int:
        key = (src, dst)
        if key not in self._conn:
            self._conn[key] = self._next_conn
            self._next_conn += 1
        return self._conn[key]

    def split_node(self, conn_innovation: int) -> int:
        """Node id created by splitting a given connection (stable across genomes)."""
        if conn_innovation not in self._split:
            self._split[conn_innovation] = self._next_node
            self._next_node += 1
        return self._split[conn_innovation]


# --------------------------------------------------------------------------
@dataclass(frozen=True)
class NodeGene:
    id: int
    width: int  # units (linear mode) or channels (conv mode)
    activation: str
    kernel_size: int = 3  # conv mode only; ignored by linear nodes


@dataclass(frozen=True)
class ConnGene:
    innovation: int
    src: int
    dst: int
    enabled: bool


# --------------------------------------------------------------------------
def _reachable(conns: tuple[ConnGene, ...], start: int) -> set[int]:
    """Nodes reachable from ``start`` following *every* connection (enabled or
    not), used for the acyclicity check."""
    adj: dict[int, list[int]] = {}
    for c in conns:
        adj.setdefault(c.src, []).append(c.dst)
    seen: set[int] = set()
    stack = [start]
    while stack:
        n = stack.pop()
        for m in adj.get(n, ()):
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return seen


@dataclass(frozen=True)
class Genome:
    """An architecture: a DAG of width-carrying nodes and directed edges."""

    nodes: tuple[NodeGene, ...]
    conns: tuple[ConnGene, ...]
    conv: bool = False  # image tasks: nodes are spatial maps, edges are convolutions
    conv_pool: int = 1  # conv mode: downsample the input by this factor (memory knob)
    # Bookkeeping filled in by the evolutionary loop.
    id: int = field(default=-1, compare=False)
    parents: tuple[int, ...] = field(default=(), compare=False)

    # -- helpers -----------------------------------------------------------
    def node_ids(self) -> set[int]:
        return {n.id for n in self.nodes} | {INPUT, OUTPUT}

    # -- construction ------------------------------------------------------
    @staticmethod
    def minimal(
        innov: Innovations,
        conv: bool = False,
        conv_pool: int = 1,
    ) -> "Genome":
        """The NEAT seed: no hidden nodes, input wired straight to output."""
        conn = ConnGene(innov.conn(INPUT, OUTPUT), INPUT, OUTPUT, True)
        return Genome(
            nodes=(),
            conns=(conn,),
            conv=conv,
            conv_pool=conv_pool if conv else 1,
        )

    @staticmethod
    def random(
        rng: random.Random,
        innov: Innovations,
        n_mutations: int = 6,
        allow_conv: bool = False,
        conv_pool: int = 1,
    ) -> "Genome":
        """A random architecture grown by a handful of structural mutations.

        Used for hand-built baselines and tests; the evolutionary search itself
        grows its population with neat-python (:mod:`evogrokking.neat_arch`).
        """
        g = Genome.minimal(innov, conv=allow_conv, conv_pool=conv_pool)
        for _ in range(n_mutations):
            g = g.mutate(rng, innov, rate=0.6)
        return g

    def _new_node_gene(self, rng: random.Random, new_id: int) -> "NodeGene":
        """A fresh hidden node, sized for conv (channels + kernel) or linear."""
        if self.conv:
            return NodeGene(
                new_id,
                width=rng.randint(*CONV_CH),
                activation=rng.choice(ACTIVATIONS),
                kernel_size=rng.choice(KERNEL_SIZES),
            )
        return NodeGene(
            new_id,
            width=rng.randint(MIN_WIDTH, MAX_WIDTH),
            activation=rng.choice(ACTIVATIONS),
        )

    # -- structural mutation primitives -----------------------------------
    def _add_node(self, rng: random.Random, innov: Innovations) -> "Genome":
        enabled = [c for c in self.conns if c.enabled]
        if not enabled:
            return self
        old = rng.choice(enabled)
        new_id = innov.split_node(old.innovation)
        if any(n.id == new_id for n in self.nodes):
            return self  # this connection was already split in this genome
        node = self._new_node_gene(rng, new_id)
        conns = [c for c in self.conns if c is not old]
        conns.append(ConnGene(old.innovation, old.src, old.dst, False))  # disable old
        conns.append(ConnGene(innov.conn(old.src, new_id), old.src, new_id, True))
        conns.append(ConnGene(innov.conn(new_id, old.dst), new_id, old.dst, True))
        return replace(self, nodes=self.nodes + (node,), conns=tuple(conns))

    def _add_connection(self, rng: random.Random, innov: Innovations) -> "Genome":
        ids = list(self.node_ids())
        existing = {(c.src, c.dst) for c in self.conns}
        rng.shuffle(ids)
        for src in ids:
            if src == OUTPUT:  # output has no outgoing edges
                continue
            reach_from_dst_cache: dict[int, set[int]] = {}
            for dst in ids:
                if dst == INPUT or dst == src:  # input has no incoming edges
                    continue
                if (src, dst) in existing:
                    continue
                # Adding src->dst is safe iff dst cannot already reach src.
                reach = reach_from_dst_cache.get(dst)
                if reach is None:
                    reach = reach_from_dst_cache[dst] = _reachable(self.conns, dst)
                if src in reach:
                    continue
                conn = ConnGene(innov.conn(src, dst), src, dst, True)
                return replace(self, conns=self.conns + (conn,))
        return self  # graph fully connected -- nothing to add

    def _toggle_connection(self, rng: random.Random) -> "Genome":
        if not self.conns:
            return self
        i = rng.randrange(len(self.conns))
        c = self.conns[i]
        conns = list(self.conns)
        conns[i] = ConnGene(c.innovation, c.src, c.dst, not c.enabled)
        return replace(self, conns=tuple(conns))

    def _perturb_node(self, rng: random.Random) -> "Genome":
        if not self.nodes:
            return self
        i = rng.randrange(len(self.nodes))
        n = self.nodes[i]
        nodes = list(self.nodes)
        width = n.width
        activation = n.activation
        kernel_size = n.kernel_size
        # In conv mode a third of the time we mutate the kernel (receptive field).
        r = rng.random()
        if self.conv and r < 0.33:
            kernel_size = rng.choice(KERNEL_SIZES)
        elif r < 0.66:
            lo, hi = CONV_CH if self.conv else (MIN_WIDTH, MAX_WIDTH)
            width = int(_clamp(round(width * math.exp(rng.gauss(0, 0.4))), lo, hi))
        else:
            activation = rng.choice(ACTIVATIONS)
        nodes[i] = NodeGene(n.id, width, activation, kernel_size)
        return replace(self, nodes=tuple(nodes))

    # -- full mutation -----------------------------------------------------
    def mutate(self, rng: random.Random, innov: Innovations, rate: float = 0.3) -> "Genome":
        """Return a structurally mutated copy (NEAT add-node / add-connection /
        toggle / node-perturb).  Architecture only -- there is nothing else left
        in the genome to mutate."""
        g = self

        max_hidden = MAX_HIDDEN_CONV if g.conv else MAX_HIDDEN
        if rng.random() < rate and len(g.nodes) < max_hidden:
            g = g._add_node(rng, innov)
        if rng.random() < rate:
            g = g._add_connection(rng, innov)
        if rng.random() < rate * 0.5:
            g = g._toggle_connection(rng)
        if rng.random() < rate:
            g = g._perturb_node(rng)

        return replace(g, id=-1, parents=(self.id,))

    # -- misc --------------------------------------------------------------
    def summary(self) -> str:
        n_enabled = sum(c.enabled for c in self.conns)
        kind = "conv" if self.conv else "mlp"
        conv_info = f" pool={self.conv_pool}" if self.conv else ""
        widths = ",".join(str(n.width) for n in self.nodes) or "-"
        return (
            f"[{kind}{conv_info}] nodes={len(self.nodes)} "
            f"conns={n_enabled}/{len(self.conns)} widths={widths}"
        )

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "parents": list(self.parents),
            "conv": self.conv,
            "nodes": [
                {"id": n.id, "width": n.width, "activation": n.activation, "kernel_size": n.kernel_size}
                for n in self.nodes
            ],
            "conns": [
                {"innovation": c.innovation, "src": c.src, "dst": c.dst, "enabled": c.enabled}
                for c in self.conns
            ],
            "conv_pool": self.conv_pool,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        """Reconstruct a genome from :meth:`as_dict` output (e.g. a saved
        ``best.json``), so an evolved individual can be reloaded and retrained."""
        return cls(
            nodes=tuple(
                NodeGene(n["id"], n["width"], n["activation"], n.get("kernel_size", 3))
                for n in d["nodes"]
            ),
            conns=tuple(
                ConnGene(c["innovation"], c["src"], c["dst"], c["enabled"])
                for c in d["conns"]
            ),
            conv=d.get("conv", False),
            conv_pool=d.get("conv_pool", 1),
            id=d.get("id", -1),
            parents=tuple(d.get("parents", ())),
        )
