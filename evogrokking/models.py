"""Turn a classical NEAT graph :class:`~evogrokking.genome.Genome` into a
trainable MLP.

The genome is an arbitrary directed-acyclic graph of **individual neurons**: each
node is one unit with a bias and an activation, each enabled edge is one scalar
weight.  Only the sub-graph that can reach an output is built, so disconnected
neurons cost nothing.

Evaluating such a graph one neuron at a time would be hopelessly slow, so the
network is compiled into a handful of **masked dense matmuls**:

* neurons are sorted into topological *levels* -- a level is a set of neurons
  with no edges among themselves, so all of them can be computed at once;
* level ``L`` reads every neuron computed before it, so its weights are one dense
  ``(prefix, |L|)`` matrix paired with a 0/1 **connectivity mask**.  The mask is
  applied at every forward pass, so gradients flow only along edges the genome
  actually has -- absent edges are exactly zero and stay that way;
* activations are applied per neuron (neurons in one level may carry different
  activation genes), and output neurons emit their pre-activation sum as logits.

The result is mathematically the per-neuron sparse network the genome describes,
but it runs as a few big GEMMs instead of thousands of tiny ones.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from evogrokking.datasets import DatasetSpec
from evogrokking.genome import Genome
from evogrokking.hyperparams import Hyperparams

_ACT = {
    "relu": torch.relu,
    "gelu": torch.nn.functional.gelu,
    "tanh": torch.tanh,
    "silu": torch.nn.functional.silu,
    "identity": lambda t: t,
}


def _reverse_reachable(conns, starts) -> set[int]:
    """Nodes that can reach any of ``starts`` via *enabled* edges (incl. them)."""
    radj: dict[int, list[int]] = {}
    for c in conns:
        if c.enabled:
            radj.setdefault(c.dst, []).append(c.src)
    seen = set(starts)
    stack = list(starts)
    while stack:
        n = stack.pop()
        for m in radj.get(n, ()):
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return seen


def _levels(conns, active: set[int], inputs: set[int]) -> list[list[int]]:
    """Group ``active`` nodes into topological levels over enabled edges.

    Level 0 is the input neurons; a node lands one level after the deepest node
    feeding it, so no level contains an edge between two of its own members --
    the invariant the whole masked-matmul scheme rests on.

    Non-input neurons are seeded at depth **1**, not 0, *before* propagation.  A
    neuron with no incoming edges is a learned constant (just its bias) and must
    still be computed before anything it feeds; seeding it at 0 and bumping it
    afterwards would let a consumer settle at depth 1 alongside it, putting an
    edge *inside* a level -- where the mask cannot represent it.
    """
    depth: dict[int, int] = {n: (0 if n in inputs else 1) for n in active}
    succ: dict[int, list[int]] = {n: [] for n in active}
    indeg: dict[int, int] = {n: 0 for n in active}
    for c in conns:
        if c.enabled and c.src in active and c.dst in active:
            succ[c.src].append(c.dst)
            indeg[c.dst] += 1

    queue = [n for n in active if indeg[n] == 0]
    order: list[int] = []
    while queue:
        n = queue.pop()
        order.append(n)
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    # ``order`` is a valid topological order, so every predecessor of ``n`` has
    # its final depth by the time ``n`` is processed.
    for n in order:
        for m in succ[n]:
            depth[m] = max(depth[m], depth[n] + 1)

    by_depth: dict[int, list[int]] = {}
    for n in active:
        by_depth.setdefault(depth[n], []).append(n)
    return [sorted(by_depth[d]) for d in sorted(by_depth)]


class GrokNet(nn.Module):
    """The MLP described by a classical NEAT genome."""

    def __init__(self, genome: Genome, spec: DatasetSpec, hp: Hyperparams | None = None):
        super().__init__()
        hp = hp or Hyperparams.for_task(spec.task)
        self.spec = spec
        self.hp = hp
        self.dropout_p = hp.dropout
        self.n_inputs = genome.n_inputs
        self.n_outputs = genome.n_outputs

        inputs = set(genome.input_ids())
        outputs = genome.output_ids()
        acts = genome.activations()

        # Only neurons that can reach an output matter; keep every input neuron
        # in the prefix so the input vector maps onto columns directly.
        active = _reverse_reachable(genome.conns, outputs) | inputs
        raw_levels = _levels(genome.conns, active, inputs)

        # Column index of every active neuron in the running activation tensor.
        # Inputs come first, in feature order, so x maps straight onto them.
        self.col: dict[int, int] = {nid: i for i, nid in enumerate(genome.input_ids())}
        self.levels: list[list[int]] = [
            [n for n in level if n not in inputs] for level in raw_levels
        ]
        self.levels = [lv for lv in self.levels if lv]

        incoming: dict[int, list[int]] = {}
        for c in genome.conns:
            if c.enabled and c.src in active and c.dst in active and c.dst not in inputs:
                incoming.setdefault(c.dst, []).append(c.src)

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self._act_groups: list[list[tuple[str, torch.Tensor]]] = []

        prefix = genome.n_inputs
        for li, level in enumerate(self.levels):
            width = len(level)
            mask = torch.zeros(prefix, width)
            for j, nid in enumerate(level):
                for src in incoming.get(nid, ()):
                    si = self.col.get(src)
                    if si is not None and si < prefix:
                        mask[si, j] = 1.0
            # Fan-in scaled init, as for a dense layer of the same connectivity.
            fan_in = mask.sum(dim=0).clamp_min(1.0)
            w = torch.randn(prefix, width) / fan_in.sqrt()
            self.weights.append(nn.Parameter(w * hp.init_scale))
            self.biases.append(nn.Parameter(torch.zeros(width)))
            self.register_buffer(f"mask_{li}", mask, persistent=False)

            # Per-neuron activations; outputs stay linear (they are logits).
            groups: dict[str, list[int]] = {}
            for j, nid in enumerate(level):
                name = "identity" if nid < genome.n_outputs else acts.get(nid, "relu")
                groups.setdefault(name, []).append(j)
            self._act_groups.append(
                [(name, torch.tensor(idx, dtype=torch.long)) for name, idx in groups.items()]
            )
            for j, nid in enumerate(level):
                self.col[nid] = prefix + j
            prefix += width

        # Where the output logits ended up in the final activation tensor.  An
        # output with no path from the input is still built (as a pure bias), so
        # every class always gets a logit.
        self.register_buffer(
            "out_index",
            torch.tensor([self.col[o] for o in outputs], dtype=torch.long),
            persistent=False,
        )

    # -- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc = x
        for i, level in enumerate(self.levels):
            w = self.weights[i] * getattr(self, f"mask_{i}")
            pre = acc @ w + self.biases[i]
            groups = self._act_groups[i]
            if len(groups) == 1:
                h = _ACT[groups[0][0]](pre)
            else:
                h = torch.empty_like(pre)
                for name, idx in groups:
                    idx = idx.to(pre.device)
                    h = h.index_copy(1, idx, _ACT[name](pre.index_select(1, idx)))
            if self.dropout_p > 0:
                h = torch.nn.functional.dropout(h, self.dropout_p, self.training)
            acc = torch.cat([acc, h], dim=1)
        return acc.index_select(1, self.out_index)


def build_model(
    genome: Genome, spec: DatasetSpec, hp: Hyperparams | None = None
) -> GrokNet:
    return GrokNet(genome, spec, hp)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# Rough multiplier turning stored-activation elements into peak training memory
# (forward activations + gradients + autograd slack).
_ACT_OVERHEAD = 3.0


def estimated_activation_mb(genome: Genome, spec: DatasetSpec, batch_size: int) -> float:
    """Estimate the peak memory (MB) of a full-batch fwd+bwd pass.

    Used as a cheap, build-free guard so the evolutionary search can skip genomes
    that would exceed the memory budget instead of OOM-ing on them.  Counts the
    activation tensor -- which, because levels are concatenated, is re-materialised
    at every level -- plus the dense level weight matrices, whose size is what
    really grows as the graph deepens.
    """
    n_hidden = len(genome.hidden_ids())
    total_nodes = genome.n_inputs + n_hidden + genome.n_outputs
    n_levels = max(2, min(n_hidden + 1, 8))  # unknown before building; assume shallow
    act_elems = batch_size * total_nodes * n_levels
    weight_elems = total_nodes * (n_hidden + genome.n_outputs)
    return (act_elems + weight_elems) * 4 * _ACT_OVERHEAD / (1024 * 1024)
