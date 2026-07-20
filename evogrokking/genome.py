"""The evolving genome -- a NEAT-style directed graph.

Classic NEAT evolves a network as a graph of **node genes** and **connection
genes**, growing arbitrary topologies (skip connections, multiple paths, ...) from
a minimal seed via structural mutations, and aligning genomes by *innovation
number* during crossover.  We follow that design, with one twist: the connection
*weights* are found by gradient descent rather than evolved, so a "node" here
carries a whole vector of units (a width) and a "connection" is a learnable linear
map between two nodes.  This keeps arbitrary DAG topologies while staying
GPU-trainable and efficient.

A genome is:

* ``nodes``  -- the hidden :class:`NodeGene`\\s (each with a width + activation).
  The input node (id ``INPUT=0``) and output node (id ``OUTPUT=1``) are implicit;
  their sizes come from the dataset.
* ``conns``  -- the :class:`ConnGene`\\s: directed ``src -> dst`` edges, each with
  an ``enabled`` flag and a global ``innovation`` number.
* the training recipe -- ``embed_dim`` (modular tasks), ``weight_decay`` (the
  **evolved regularisation strength**, the key grokking knob), ``dropout``,
  ``optimizer``, ``lr`` and ``init_scale``.

Structural invariant: the full connection set is always a DAG (cycle-free), so any
enabled subset is feed-forward and can be evaluated in topological order.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace

ACTIVATIONS = ("relu", "gelu", "tanh", "silu")
OPTIMIZERS = ("adam", "adamw", "sgd")

INPUT = 0
OUTPUT = 1

# Bounds keep evolution inside a sane, trainable region.
_MIN_WIDTH, _MAX_WIDTH = 8, 512
_MAX_HIDDEN = 12
_WD_RANGE = (1e-6, 3.0)  # log-uniform
_LR_RANGE = (1e-4, 3e-1)  # log-uniform
_DROPOUT_RANGE = (0.0, 0.6)
_INIT_RANGE = (0.2, 8.0)  # weight-init scale multiplier; large init aids grokking
_EMBED_RANGE = (8, 256)

# Convolutional mode (image tasks): a hidden node is a spatial feature map with
# this many channels (``width`` doubles as channel count) and one of these
# same-padding kernel sizes.  Channels *and* the number of conv nodes are kept
# small because same-conv preserves the spatial resolution, so activation memory
# grows as batch x channels x H x W x nodes.
_CONV_CH = (4, 32)
_MAX_HIDDEN_CONV = 6
KERNEL_SIZES = (3, 5, 7)


def _log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# --------------------------------------------------------------------------
# Innovation bookkeeping -- shared across a whole run so that homologous
# structures in different genomes get the same ids (this is what makes crossover
# alignment meaningful).
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
    nodes: tuple[NodeGene, ...]
    conns: tuple[ConnGene, ...]
    embed_dim: int
    weight_decay: float
    dropout: float
    optimizer: str
    lr: float
    init_scale: float
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
        rng: random.Random,
        innov: Innovations,
        conv: bool = False,
        conv_pool: int = 1,
    ) -> "Genome":
        """The NEAT seed: no hidden nodes, input wired straight to output."""
        conn = ConnGene(innov.conn(INPUT, OUTPUT), INPUT, OUTPUT, True)
        return Genome(
            nodes=(),
            conns=(conn,),
            embed_dim=rng.randint(*_EMBED_RANGE),
            weight_decay=_log_uniform(rng, *_WD_RANGE),
            dropout=rng.uniform(*_DROPOUT_RANGE),
            optimizer=rng.choice(OPTIMIZERS),
            lr=_log_uniform(rng, *_LR_RANGE),
            init_scale=rng.uniform(*_INIT_RANGE),
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
        """Seed genome grown by a handful of random structural mutations so the
        initial population is topologically diverse.  ``allow_conv`` makes the
        genome a convolutional one (image tasks only); ``conv_pool`` downsamples
        the input to cap activation memory."""
        g = Genome.minimal(rng, innov, conv=allow_conv, conv_pool=conv_pool)
        for _ in range(n_mutations):
            g = g.mutate(rng, innov, rate=0.6)
        return g

    def _new_node_gene(self, rng: random.Random, new_id: int) -> "NodeGene":
        """A fresh hidden node, sized for conv (channels + kernel) or linear."""
        if self.conv:
            return NodeGene(
                new_id,
                width=rng.randint(*_CONV_CH),
                activation=rng.choice(ACTIVATIONS),
                kernel_size=rng.choice(KERNEL_SIZES),
            )
        return NodeGene(
            new_id,
            width=rng.randint(_MIN_WIDTH, _MAX_WIDTH),
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
            lo, hi = _CONV_CH if self.conv else (_MIN_WIDTH, _MAX_WIDTH)
            width = int(_clamp(round(width * math.exp(rng.gauss(0, 0.4))), lo, hi))
        else:
            activation = rng.choice(ACTIVATIONS)
        nodes[i] = NodeGene(n.id, width, activation, kernel_size)
        return replace(self, nodes=tuple(nodes))

    # -- full mutation -----------------------------------------------------
    def mutate(self, rng: random.Random, innov: Innovations, rate: float = 0.3) -> "Genome":
        """Return a mutated copy: structural graph mutations + recipe drift."""
        g = self

        # Structural mutations (NEAT's add-node / add-connection, plus toggling).
        max_hidden = _MAX_HIDDEN_CONV if g.conv else _MAX_HIDDEN
        if rng.random() < rate and len(g.nodes) < max_hidden:
            g = g._add_node(rng, innov)
        if rng.random() < rate:
            g = g._add_connection(rng, innov)
        if rng.random() < rate * 0.5:
            g = g._toggle_connection(rng)
        if rng.random() < rate:
            g = g._perturb_node(rng)

        # Training-recipe genes drift log-normally / normally, then clamp.
        embed_dim = g.embed_dim
        if rng.random() < rate:
            embed_dim = int(_clamp(round(embed_dim * math.exp(rng.gauss(0, 0.4))), *_EMBED_RANGE))

        weight_decay = g.weight_decay
        if rng.random() < rate:
            weight_decay = _clamp(weight_decay * math.exp(rng.gauss(0, 0.7)), *_WD_RANGE)

        lr = g.lr
        if rng.random() < rate:
            lr = _clamp(lr * math.exp(rng.gauss(0, 0.5)), *_LR_RANGE)

        dropout = g.dropout
        if rng.random() < rate:
            dropout = _clamp(dropout + rng.gauss(0, 0.15), *_DROPOUT_RANGE)

        init_scale = g.init_scale
        if rng.random() < rate:
            init_scale = _clamp(init_scale * math.exp(rng.gauss(0, 0.4)), *_INIT_RANGE)

        optimizer = g.optimizer
        if rng.random() < rate:
            optimizer = rng.choice(OPTIMIZERS)

        return replace(
            g,
            embed_dim=embed_dim,
            weight_decay=weight_decay,
            dropout=dropout,
            optimizer=optimizer,
            lr=lr,
            init_scale=init_scale,
            id=-1,
            parents=(self.id,),
        )

    # -- crossover (innovation-aligned) -----------------------------------
    @staticmethod
    def crossover(
        fitter: "Genome", other: "Genome", rng: random.Random
    ) -> "Genome":
        """NEAT crossover: matching connection genes (same innovation) are
        inherited from a random parent; disjoint/excess genes come from the
        *fitter* parent.  Hidden nodes referenced by the child's connections are
        inherited (preferring the fitter parent's attributes)."""
        other_by_innov = {c.innovation: c for c in other.conns}
        child_conns: list[ConnGene] = []
        for c in fitter.conns:
            match = other_by_innov.get(c.innovation)
            if match is not None and rng.random() < 0.5:
                chosen = match
            else:
                chosen = c
            # A gene disabled in either parent is disabled with prob 0.75 (NEAT).
            enabled = chosen.enabled
            if match is not None and (not c.enabled or not match.enabled):
                enabled = rng.random() >= 0.75
            child_conns.append(ConnGene(chosen.innovation, chosen.src, chosen.dst, enabled))

        needed = {c.src for c in child_conns} | {c.dst for c in child_conns}
        fitter_nodes = {n.id: n for n in fitter.nodes}
        other_nodes = {n.id: n for n in other.nodes}
        child_nodes = tuple(
            fitter_nodes.get(nid) or other_nodes[nid]
            for nid in needed
            if nid in fitter_nodes or nid in other_nodes
        )

        pick = lambda a, b: a if rng.random() < 0.5 else b
        return Genome(
            nodes=child_nodes,
            conns=tuple(child_conns),
            embed_dim=pick(fitter.embed_dim, other.embed_dim),
            weight_decay=math.sqrt(fitter.weight_decay * other.weight_decay),
            dropout=0.5 * (fitter.dropout + other.dropout),
            optimizer=pick(fitter.optimizer, other.optimizer),
            lr=math.sqrt(fitter.lr * other.lr),
            init_scale=0.5 * (fitter.init_scale + other.init_scale),
            conv=fitter.conv,  # conv (+ pool) is a run-level mode, shared by parents
            conv_pool=fitter.conv_pool,
            id=-1,
            parents=(fitter.id, other.id),
        )

    # -- misc --------------------------------------------------------------
    def summary(self) -> str:
        n_enabled = sum(c.enabled for c in self.conns)
        kind = "conv" if self.conv else "mlp"
        conv_info = f" pool={self.conv_pool}" if self.conv else ""
        return (
            f"[{kind}{conv_info}] nodes={len(self.nodes)} "
            f"conns={n_enabled}/{len(self.conns)} "
            f"emb={self.embed_dim} wd={self.weight_decay:.2g} do={self.dropout:.2f} "
            f"opt={self.optimizer} lr={self.lr:.2g} init={self.init_scale:.2f}"
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
            "embed_dim": self.embed_dim,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "optimizer": self.optimizer,
            "lr": self.lr,
            "init_scale": self.init_scale,
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
            embed_dim=d["embed_dim"],
            weight_decay=d["weight_decay"],
            dropout=d["dropout"],
            optimizer=d["optimizer"],
            lr=d["lr"],
            init_scale=d["init_scale"],
            conv=d.get("conv", False),
            conv_pool=d.get("conv_pool", 1),
            id=d.get("id", -1),
            parents=tuple(d.get("parents", ())),
        )
