"""Plots for a run.

* :func:`plot_curves` -- the log-loss and accuracy learning curves (the shaded
  gap on the log axis is exactly the *grokking area* we measure).
* :func:`plot_genome` -- the evolved NEAT network as a graph, laid out
  left-to-right by topological depth.  Small graphs are drawn neuron by neuron;
  larger ones collapse each level into a single box labelled with its neuron
  count, with connection counts on the arrows.

Matplotlib is imported lazily with the non-interactive ``Agg`` backend so this
works headless (no display needed) and only costs anything when a plot is asked
for.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

from evogrokking.genome import Genome


def plot_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    train_accs: Sequence[float],
    val_accs: Sequence[float],
    out_path: str,
    *,
    title: str = "",
    eval_every: int = 1,
    grok_area: float | None = None,
    acc_area: float | None = None,
) -> str:
    """Render learning curves to ``out_path`` (PNG). Returns ``out_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [i * eval_every for i in range(len(train_losses))]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.8))

    # -- loss (log scale): the shaded gap is the grokking area --------------
    ax_loss.plot(epochs, train_losses, label="train", color="#1f77b4")
    ax_loss.plot(epochs, val_losses, label="val", color="#d62728")
    ax_loss.fill_between(
        epochs,
        train_losses,
        val_losses,
        where=[v > t for v, t in zip(val_losses, train_losses)],
        color="#d62728",
        alpha=0.12,
        label="grokking gap",
    )
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss (log scale)")
    loss_title = "Loss"
    if grok_area is not None:
        loss_title += f"  (grok_area = {grok_area:.3f})"
    ax_loss.set_title(loss_title)
    ax_loss.legend(loc="upper right")
    ax_loss.grid(True, which="both", alpha=0.2)

    # -- accuracy: the shaded gap is the accuracy area ----------------------
    tr = [a * 100 for a in train_accs]
    va = [a * 100 for a in val_accs]
    ax_acc.plot(epochs, tr, label="train", color="#1f77b4")
    ax_acc.plot(epochs, va, label="val", color="#d62728")
    ax_acc.fill_between(
        epochs,
        va,
        tr,
        where=[t > v for t, v in zip(tr, va)],
        color="#1f77b4",
        alpha=0.12,
        label="grokking gap",
    )
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy (%)")
    ax_acc.set_ylim(-2, 102)
    acc_title = "Accuracy"
    if acc_area is not None:
        acc_title += f"  (acc_area = {acc_area:.3f})"
    ax_acc.set_title(acc_title)
    ax_acc.legend(loc="lower right")
    ax_acc.grid(True, alpha=0.2)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Network-structure graph
# --------------------------------------------------------------------------
def _levels(genome: Genome) -> dict[int, int]:
    """Assign each neuron a column = longest path from an input over *enabled*
    edges (a standard feed-forward layering)."""
    node_ids = genome.node_ids()
    enabled = [
        (c.src, c.dst)
        for c in genome.conns
        if c.enabled and c.src in node_ids and c.dst in node_ids
    ]

    succ: dict[int, list[int]] = {n: [] for n in node_ids}
    indeg: dict[int, int] = {n: 0 for n in node_ids}
    for s, d in enabled:
        succ[s].append(d)
        indeg[d] += 1

    queue = deque(n for n in node_ids if indeg[n] == 0)
    remaining = dict(indeg)
    order: list[int] = []
    while queue:
        n = queue.popleft()
        order.append(n)
        for d in succ[n]:
            remaining[d] -= 1
            if remaining[d] == 0:
                queue.append(d)

    depth = {n: 0 for n in node_ids}
    for n in order:
        for d in succ[n]:
            depth[d] = max(depth[d], depth[n] + 1)
    # Inputs are always column 0 and outputs always the last, so the picture
    # reads left-to-right even when a stray edge would say otherwise.
    for n in genome.input_ids():
        depth[n] = 0
    last = max(depth.values(), default=1)
    for n in genome.output_ids():
        depth[n] = max(last, 1)
    return depth


#: Above this many neurons the per-neuron picture is unreadable, so levels are
#: drawn collapsed into one box each.
DETAIL_LIMIT = 60


def plot_genome(
    genome: Genome, out_path: str, *, spec=None, title: str = "", detail_limit: int | None = None
) -> str:
    """Render the evolved network as a left-to-right graph. Returns ``out_path``.

    Classical NEAT graphs have one node per *neuron*, so an MNIST genome has 784
    input neurons and can carry tens of thousands of edges -- far past what any
    node-and-arrow diagram can show.  Two modes handle that:

    * **detailed** (small graphs): every neuron is a box labelled with its id and
      activation; solid arrows are the enabled connections, faint dotted arrows
      the disabled genes the genome still carries.
    * **collapsed** (anything larger): each topological level becomes one box
      giving its neuron count, and each pair of levels one arrow labelled with the
      number of connections running between them -- so the *shape* of the evolved
      network, and where it is densely or sparsely wired, is still legible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    limit = DETAIL_LIMIT if detail_limit is None else detail_limit
    depth = _levels(genome)
    n_nodes = len(depth)
    acts = genome.activations()
    inputs = set(genome.input_ids())
    outputs = set(genome.output_ids())

    columns: dict[int, list[int]] = {}
    for node, col in depth.items():
        columns.setdefault(col, []).append(node)
    n_cols = max(columns) + 1

    detailed = n_nodes <= limit

    if detailed:
        pos: dict[int, tuple[float, float]] = {}
        for col, members in columns.items():
            members = sorted(members)
            k = len(members)
            for i, node in enumerate(members):
                pos[node] = (float(col), (k - 1) / 2.0 - i)
        max_rows = max(len(m) for m in columns.values())
        fig, ax = plt.subplots(
            figsize=(max(6.0, 2.4 * n_cols), max(3.0, 0.9 * max_rows))
        )

        for c in genome.conns:
            if c.src not in pos or c.dst not in pos:
                continue
            x0, y0 = pos[c.src]
            x1, y1 = pos[c.dst]
            if c.enabled:
                arrow = dict(arrowstyle="-|>", color="#555555", lw=1.0, alpha=0.75,
                             shrinkA=14, shrinkB=14, connectionstyle="arc3,rad=0.08")
            else:
                arrow = dict(arrowstyle="-|>", color="#cccccc", lw=0.7, ls=":",
                             shrinkA=14, shrinkB=14, alpha=0.6,
                             connectionstyle="arc3,rad=0.08")
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=arrow, zorder=1)

        for node, (x, y) in pos.items():
            if node in inputs:
                label, face, edge = f"in{-node}", "#d9f0d3", "#2ca02c"
            elif node in outputs:
                label, face, edge = f"out{node}", "#f7d5d5", "#d62728"
            else:
                label, face, edge = f"#{node}\n{acts.get(node, '?')}", "#d6e5f3", "#1f77b4"
            ax.text(
                x, y, label, ha="center", va="center", fontsize=8, zorder=2,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=face, edgecolor=edge, lw=1.4),
            )
        ax.set_ylim(-max_rows / 2 - 0.8, max_rows / 2 + 0.8)
    else:
        # Collapsed: one box per level, arrows labelled with connection counts.
        level_of = depth
        pair_counts: dict[tuple[int, int], int] = {}
        for c in genome.conns:
            if not c.enabled or c.src not in level_of or c.dst not in level_of:
                continue
            key = (level_of[c.src], level_of[c.dst])
            pair_counts[key] = pair_counts.get(key, 0) + 1

        fig, ax = plt.subplots(figsize=(max(7.0, 2.6 * n_cols), 4.2))
        pos = {col: (float(col), 0.0) for col in columns}

        for (a, b), count in sorted(pair_counts.items()):
            x0, _ = pos[a]
            x1, _ = pos[b]
            rad = 0.0 if b == a + 1 else 0.45  # curve the skip connections
            ax.annotate(
                "", xy=(x1, 0.0), xytext=(x0, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.3,
                                shrinkA=34, shrinkB=34,
                                connectionstyle=f"arc3,rad={rad}"),
                zorder=1,
            )
            # Straight arrows are labelled just above the line; skip arcs bow
            # *below* the row of boxes, so their labels follow them down rather
            # than landing on top of the level they jump over.
            if rad == 0.0:
                ly, va = 0.10, "bottom"
            else:
                ly, va = -(0.28 + 0.55 * abs(rad)), "top"
            ax.text(
                (x0 + x1) / 2, ly, f"{count:,}",
                ha="center", va=va, fontsize=8, color="#333333", zorder=3,
            )

        for col, members in sorted(columns.items()):
            k = len(members)
            if col == 0:
                label, face, edge = f"input\n{k:,} neurons", "#d9f0d3", "#2ca02c"
            elif set(members) <= outputs:
                label, face, edge = f"output\n{k} neurons", "#f7d5d5", "#d62728"
            else:
                used = sorted({acts[m] for m in members if m in acts})
                label = f"level {col}\n{k:,} neurons\n{'/'.join(used) or '-'}"
                face, edge = "#d6e5f3", "#1f77b4"
            ax.text(
                col, 0.0, label, ha="center", va="center", fontsize=9, zorder=2,
                bbox=dict(boxstyle="round,pad=0.45", facecolor=face, edgecolor=edge, lw=1.6),
            )
        ax.set_ylim(-1.2, 1.6)
        ax.text(
            0.5, 0.02,
            f"levels collapsed ({n_nodes:,} neurons, {genome.n_enabled():,} enabled "
            f"connections); arrow labels are connection counts",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8, color="#666666",
        )

    ax.set_xlim(-0.7, n_cols - 0.3)
    ax.axis("off")
    ax.set_title(title or genome.summary(), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
