"""Turn a NEAT graph :class:`~evogrokking.genome.Genome` into a trainable network.

The genome is an arbitrary directed-acyclic graph of nodes, evaluated in
topological order.  Only the sub-graph that can reach the output is built, so
disconnected genes cost nothing.  There are two modes:

**Linear mode** (modular tasks, and image tasks without ``--conv``).  Each node
holds a vector of ``width`` units; each enabled edge ``src -> dst`` is a learnable
``Linear(width[src], width[dst])``; a node sums its incoming edges plus a bias and
applies its activation.  The input node is the feature vector (concatenated token
embeddings for modular tasks, flattened pixels for images); the output node emits
the class logits.

**Convolutional mode** (image tasks with ``conv=True``).  Each node is a *spatial
feature map* with ``width`` channels; each enabled edge between hidden nodes is a
**same-padding** ``Conv2d`` (stride 1, so every map keeps the input resolution and
arbitrary skip connections line up).  The input node is the image ``(C, H, W)``.
Edges into the output node global-average-pool the source map and apply a
``Linear`` to the class logits -- a standard GAP classifier head.

The training recipe (``init_scale``, ``dropout``, ``embed_dim``) is *not* part
of the genome -- it arrives as a :class:`~evogrokking.hyperparams.Hyperparams`
fixed for the whole run.  ``init_scale`` multiplies every weight; a large initial
weight norm together with weight decay is a documented driver of grokking (Liu et
al., "Omnigrok").
"""

from __future__ import annotations

import torch
import torch.nn as nn

from evogrokking.datasets import DatasetSpec
from evogrokking.genome import INPUT, OUTPUT, Genome
from evogrokking.hyperparams import Hyperparams

_ACT = {
    "relu": torch.relu,
    "gelu": torch.nn.functional.gelu,
    "tanh": torch.tanh,
    "silu": torch.nn.functional.silu,
}


def _reverse_reachable(conns, start: int) -> set[int]:
    """Nodes that can reach ``start`` via *enabled* edges (incl. ``start``)."""
    radj: dict[int, list[int]] = {}
    for c in conns:
        if c.enabled:
            radj.setdefault(c.dst, []).append(c.src)
    seen = {start}
    stack = [start]
    while stack:
        n = stack.pop()
        for m in radj.get(n, ()):
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return seen


def _topo_order(conns, active: set[int]) -> list[int]:
    """Kahn topological sort of ``active`` nodes over enabled edges."""
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
    return order


class GrokNet(nn.Module):
    """Network described by a NEAT graph genome (linear or convolutional)."""

    def __init__(self, genome: Genome, spec: DatasetSpec, hp: Hyperparams | None = None):
        super().__init__()
        hp = hp or Hyperparams.for_task(spec.task)
        self.spec = spec
        self.hp = hp
        self.dropout_p = hp.dropout
        self.is_conv = bool(genome.conv and spec.image_shape is not None)

        # Per-node output size: units (linear) or channels (conv); the input and
        # output nodes are sized by the dataset.
        if self.is_conv:
            self.embedding = None
            channels, self.image_shape = spec.image_shape[0], spec.image_shape
            # Downsample the input once so every same-conv map runs at the reduced
            # resolution -- the main lever on conv activation memory.
            self.conv_pool = max(1, genome.conv_pool)
            self.eff_hw = (
                self.image_shape[1] // self.conv_pool,
                self.image_shape[2] // self.conv_pool,
            )
            sizes = {INPUT: channels, OUTPUT: spec.num_classes}
            self.kernels: dict[int, int] = {}
        elif spec.task == "modular":
            assert spec.vocab_size is not None
            self.embedding = nn.Embedding(spec.vocab_size, hp.embed_dim)
            sizes = {INPUT: spec.input_dim * hp.embed_dim, OUTPUT: spec.num_classes}
        else:
            self.embedding = None
            sizes = {INPUT: spec.input_dim, OUTPUT: spec.num_classes}

        self.acts: dict[int, str] = {}
        for node in genome.nodes:
            sizes[node.id] = node.width
            self.acts[node.id] = node.activation
            if self.is_conv:
                self.kernels[node.id] = node.kernel_size

        # Only the sub-graph that can reach the output matters.
        active = _reverse_reachable(genome.conns, OUTPUT)
        self.order = [n for n in _topo_order(genome.conns, active) if n != INPUT]

        # One module per enabled edge whose endpoints are both active; one bias
        # per computed node.  ModuleDict/ParameterDict keys must be strings.
        self.edges = nn.ModuleDict()
        self.incoming: dict[int, list[tuple[int, str]]] = {n: [] for n in self.order}
        for c in genome.conns:
            if not (c.enabled and c.src in active and c.dst in active and c.dst != INPUT):
                continue
            key = f"{c.src}_{c.dst}"
            self.edges[key] = self._make_edge(sizes, c.src, c.dst)
            self.incoming[c.dst].append((c.src, key))

        self.biases = nn.ParameterDict(
            {str(n): nn.Parameter(torch.zeros(sizes[n])) for n in self.order}
        )
        self.input_used = INPUT in active
        self._scale_init(hp.init_scale)

    def _make_edge(self, sizes: dict[int, int], src: int, dst: int) -> nn.Module:
        if not self.is_conv:
            return nn.Linear(sizes[src], sizes[dst], bias=False)
        if dst == OUTPUT:
            # Leaving the spatial domain: global-average-pool then classify.
            return nn.Linear(sizes[src], sizes[dst], bias=False)
        k = self.kernels[dst]
        return nn.Conv2d(sizes[src], sizes[dst], k, padding=k // 2, bias=False)

    def _scale_init(self, scale: float) -> None:
        if scale == 1.0:
            return
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d)):
                    module.weight.mul_(scale)

    # -- forward -----------------------------------------------------------
    def _input_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_conv:
            img = x.view(x.shape[0], *self.image_shape)
            if self.conv_pool > 1:
                img = torch.nn.functional.avg_pool2d(img, self.conv_pool)
            return img
        if self.embedding is not None:
            return self.embedding(x).flatten(start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        vals: dict[int, torch.Tensor] = {}
        if self.input_used:
            vals[INPUT] = self._input_tensor(x)

        for node in self.order:
            incoming = self.incoming[node]
            bias = self.biases[str(node)]

            if node == OUTPUT:
                if incoming:
                    pre = bias
                    for src, key in incoming:
                        feat = vals[src]
                        if self.is_conv:
                            feat = feat.mean(dim=(2, 3))  # global average pool
                        pre = pre + self.edges[key](feat)
                else:
                    # No path to the output: emit the (broadcast) bias as logits.
                    pre = bias.unsqueeze(0).expand(batch, -1)
                vals[node] = pre  # logits: no activation
                continue

            if incoming:
                pre = bias.view(1, -1, 1, 1) if self.is_conv else bias
                for src, key in incoming:
                    pre = pre + self.edges[key](vals[src])
            else:
                # A node with no incoming edges is a learned constant; broadcast
                # its bias across the batch (and spatial dims in conv mode).
                if self.is_conv:
                    c, (h, w) = bias.shape[0], self.eff_hw
                    pre = bias.view(1, c, 1, 1).expand(batch, c, h, w)
                else:
                    pre = bias.unsqueeze(0).expand(batch, -1)

            h = _ACT[self.acts[node]](pre)
            if self.dropout_p > 0:
                h = torch.nn.functional.dropout(h, self.dropout_p, self.training)
            vals[node] = h

        return vals[OUTPUT]


def build_model(
    genome: Genome, spec: DatasetSpec, hp: Hyperparams | None = None
) -> GrokNet:
    return GrokNet(genome, spec, hp)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# Rough multiplier turning stored-activation elements into peak training memory
# (forward activations + gradients + conv workspace slack).
_ACT_OVERHEAD = 3.0


def estimated_activation_mb(
    genome: Genome, spec: DatasetSpec, batch_size: int, embed_dim: int = 128
) -> float:
    """Estimate the peak activation memory (MB) of a full-batch fwd+bwd pass.

    Used as a cheap, build-free guard so the evolutionary search can skip
    genomes that would exceed the GPU-memory budget instead of OOM-ing on them.
    Sums ``batch * units`` over the hidden nodes (units = channels x H x W in conv
    mode, at the pooled resolution), a conservative upper bound that ignores dead
    genes.
    """
    if genome.conv and spec.image_shape is not None:
        c0, h, w = spec.image_shape
        pool = max(1, genome.conv_pool)
        spatial = (h // pool) * (w // pool)
        elems = batch_size * c0 * spatial  # input feature map
        elems += batch_size * sum(n.width * spatial for n in genome.nodes)
    else:
        in_dim = spec.input_dim * (embed_dim if spec.task == "modular" else 1)
        elems = batch_size * in_dim
        elems += batch_size * sum(n.width for n in genome.nodes)
    return elems * 4 * _ACT_OVERHEAD / (1024 * 1024)
