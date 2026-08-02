"""The evolving genome -- a classical NEAT graph of **individual neurons**.

This module holds the **phenotype description**: the graph that
:mod:`evogrokking.models` turns into a trainable network.  Reproduction itself
(mutation, crossover, speciation, innovation bookkeeping) is delegated to the
`neat-python <https://neat-python.readthedocs.io>`_ library -- see
:mod:`evogrokking.neat_arch`, which converts a ``neat.DefaultGenome`` into the
:class:`Genome` defined here.

This is classical NEAT: a node is **one neuron**, not a layer, and a connection
is **one scalar weight** between two neurons.  Networks are plain MLPs -- there
are no convolutions, no embeddings, and no width genes.  What evolves is purely
the *topology*: which neurons exist, which of them are wired together, and each
neuron's activation function.  Connection weights and biases are found by
gradient descent rather than evolved, which is what lets a run produce the
learning curves the grokking metrics are computed from.

Node keys follow neat-python's convention, so translation is near-identity:

* **inputs**  ``-1 .. -n_inputs``   -- one neuron per input feature,
* **outputs** ``0 .. n_outputs-1``  -- one neuron per class logit,
* **hidden**  ``>= n_outputs``.

A genome is:

* ``nodes``  -- the hidden *and* output :class:`NodeGene`\\s (each just an
  activation function).  Input neurons carry no gene: they are the data.
* ``conns``  -- the :class:`ConnGene`\\s: directed ``src -> dst`` edges, each with
  an ``enabled`` flag and a global ``innovation`` number.

Structural invariant: the full connection set is always a DAG (cycle-free), so any
enabled subset is feed-forward and can be evaluated in topological order.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from random import Random

ACTIVATIONS = ("relu", "gelu", "tanh", "silu")

#: Upper bound on hidden neurons, so a runaway add-node mutation cannot make a
#: genome untrainable.
MAX_HIDDEN = 4096


# --------------------------------------------------------------------------
# Innovation bookkeeping -- shared across a whole run so that homologous
# structures in different genomes get the same ids.  neat-python keeps its own
# tracker during evolution; this one backs the hand-built dense baseline and the
# random architectures used in tests.
# --------------------------------------------------------------------------
class Innovations:
    """Hands out globally-consistent connection innovation numbers and node ids."""

    def __init__(self, first_hidden: int = 1) -> None:
        self._conn: dict[tuple[int, int], int] = {}
        self._split: dict[int, int] = {}
        self._next_conn = 0
        self._next_node = first_hidden

    def conn(self, src: int, dst: int) -> int:
        key = (src, dst)
        if key not in self._conn:
            self._conn[key] = self._next_conn
            self._next_conn += 1
        return self._conn[key]

    def new_node(self) -> int:
        nid = self._next_node
        self._next_node += 1
        return nid

    def split_node(self, conn_innovation: int) -> int:
        """Node id created by splitting a given connection (stable across genomes)."""
        if conn_innovation not in self._split:
            self._split[conn_innovation] = self.new_node()
        return self._split[conn_innovation]


# --------------------------------------------------------------------------
@dataclass(frozen=True)
class NodeGene:
    """One neuron.  Its bias is trained, so the only gene is the activation."""

    id: int
    activation: str


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
    """A classical NEAT architecture: a DAG of individual neurons."""

    nodes: tuple[NodeGene, ...]  # hidden + output neurons
    conns: tuple[ConnGene, ...]
    n_inputs: int
    n_outputs: int
    # Bookkeeping filled in by the evolutionary loop.
    id: int = field(default=-1, compare=False)
    parents: tuple[int, ...] = field(default=(), compare=False)

    # -- helpers -----------------------------------------------------------
    def input_ids(self) -> list[int]:
        return [-i - 1 for i in range(self.n_inputs)]

    def output_ids(self) -> list[int]:
        return list(range(self.n_outputs))

    def hidden_ids(self) -> list[int]:
        return sorted(n.id for n in self.nodes if n.id >= self.n_outputs)

    def node_ids(self) -> set[int]:
        return {n.id for n in self.nodes} | set(self.input_ids())

    def activations(self) -> dict[int, str]:
        return {n.id: n.activation for n in self.nodes}

    def n_enabled(self) -> int:
        return sum(c.enabled for c in self.conns)

    # -- construction ------------------------------------------------------
    @staticmethod
    def dense(
        n_inputs: int,
        n_outputs: int,
        n_hidden: int,
        *,
        activation: str = "relu",
        direct: bool = True,
        innov: Innovations | None = None,
    ) -> "Genome":
        """A **big, densely connected** starting network.

        Every input neuron is wired to every hidden neuron, every hidden neuron to
        every output, and (with ``direct``) every input straight to every output
        as well.  This is the founding structure the search starts from and then
        prunes and rewires -- the opposite of NEAT's usual minimal seed.
        """
        innov = innov or Innovations(first_hidden=n_outputs)
        hidden = [innov.new_node() for _ in range(n_hidden)]
        inputs = [-i - 1 for i in range(n_inputs)]
        outputs = list(range(n_outputs))

        nodes = tuple(
            NodeGene(nid, activation) for nid in outputs + hidden
        )
        conns: list[ConnGene] = []
        for i in inputs:
            for h in hidden:
                conns.append(ConnGene(innov.conn(i, h), i, h, True))
            if direct:
                for o in outputs:
                    conns.append(ConnGene(innov.conn(i, o), i, o, True))
        for h in hidden:
            for o in outputs:
                conns.append(ConnGene(innov.conn(h, o), h, o, True))

        return Genome(nodes, tuple(conns), n_inputs, n_outputs)

    @staticmethod
    def random(
        rng: Random,
        n_inputs: int = 6,
        n_outputs: int = 3,
        n_hidden: int = 4,
        n_mutations: int = 8,
        innov: Innovations | None = None,
    ) -> "Genome":
        """A random architecture: a small dense net, then structural mutations.

        Used for tests; the evolutionary search grows its population with
        neat-python (:mod:`evogrokking.neat_arch`).
        """
        innov = innov or Innovations(first_hidden=n_outputs)
        g = Genome.dense(n_inputs, n_outputs, n_hidden, innov=innov)
        for _ in range(n_mutations):
            g = g.mutate(rng, innov, rate=0.6)
        return g

    # -- structural mutation primitives -----------------------------------
    def _add_node(self, rng: Random, innov: Innovations) -> "Genome":
        """Split an enabled connection with a new neuron (NEAT's add-node)."""
        enabled = [c for c in self.conns if c.enabled]
        if not enabled or len(self.hidden_ids()) >= MAX_HIDDEN:
            return self
        old = rng.choice(enabled)
        new_id = innov.split_node(old.innovation)
        if any(n.id == new_id for n in self.nodes):
            return self  # this connection was already split in this genome
        node = NodeGene(new_id, rng.choice(ACTIVATIONS))
        conns = [c for c in self.conns if c is not old]
        conns.append(ConnGene(old.innovation, old.src, old.dst, False))  # disable old
        conns.append(ConnGene(innov.conn(old.src, new_id), old.src, new_id, True))
        conns.append(ConnGene(innov.conn(new_id, old.dst), new_id, old.dst, True))
        return replace(self, nodes=self.nodes + (node,), conns=tuple(conns))

    def _add_connection(self, rng: Random, innov: Innovations) -> "Genome":
        ids = list(self.node_ids())
        existing = {(c.src, c.dst) for c in self.conns}
        rng.shuffle(ids)
        inputs = set(self.input_ids())
        for src in ids:
            reach_from_dst_cache: dict[int, set[int]] = {}
            for dst in ids:
                if dst in inputs or dst == src:  # inputs have no incoming edges
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

    def _toggle_connection(self, rng: Random) -> "Genome":
        if not self.conns:
            return self
        i = rng.randrange(len(self.conns))
        c = self.conns[i]
        conns = list(self.conns)
        conns[i] = ConnGene(c.innovation, c.src, c.dst, not c.enabled)
        return replace(self, conns=tuple(conns))

    def _perturb_node(self, rng: Random) -> "Genome":
        """Swap one neuron's activation function."""
        hidden = [i for i, n in enumerate(self.nodes) if n.id >= self.n_outputs]
        if not hidden:
            return self
        i = rng.choice(hidden)
        n = self.nodes[i]
        nodes = list(self.nodes)
        nodes[i] = NodeGene(n.id, rng.choice(ACTIVATIONS))
        return replace(self, nodes=tuple(nodes))

    # -- full mutation -----------------------------------------------------
    def mutate(self, rng: Random, innov: Innovations, rate: float = 0.3) -> "Genome":
        """Return a structurally mutated copy (NEAT add-node / add-connection /
        toggle / activation swap).  Architecture only -- there is nothing else
        left in the genome to mutate."""
        g = self
        if rng.random() < rate:
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
        return (
            f"[mlp] in={self.n_inputs} hidden={len(self.hidden_ids())} "
            f"out={self.n_outputs} conns={self.n_enabled()}/{len(self.conns)}"
        )

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "parents": list(self.parents),
            "n_inputs": self.n_inputs,
            "n_outputs": self.n_outputs,
            "nodes": [{"id": n.id, "activation": n.activation} for n in self.nodes],
            "conns": [
                {"innovation": c.innovation, "src": c.src, "dst": c.dst, "enabled": c.enabled}
                for c in self.conns
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        """Reconstruct a genome from :meth:`as_dict` output (e.g. a saved
        ``best.json``), so an evolved individual can be reloaded and retrained."""
        return cls(
            nodes=tuple(NodeGene(n["id"], n["activation"]) for n in d["nodes"]),
            conns=tuple(
                ConnGene(c["innovation"], c["src"], c["dst"], c["enabled"])
                for c in d["conns"]
            ),
            n_inputs=d["n_inputs"],
            n_outputs=d["n_outputs"],
            id=d.get("id", -1),
            parents=tuple(d.get("parents", ())),
        )
