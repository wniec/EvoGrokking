"""Plots for a run.

* :func:`plot_curves` -- the log-loss and accuracy learning curves (the shaded
  gap on the log axis is exactly the *grokking area* we measure).
* :func:`plot_genome` -- the evolved NEAT network as a graph, laid out
  left-to-right by topological depth, so the obtained architecture (nodes, their
  widths/kernels, and the skip connections between them) is visible at a glance.

Matplotlib is imported lazily with the non-interactive ``Agg`` backend so this
works headless (no display needed) and only costs anything when a plot is asked
for.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

from evogrokking.genome import INPUT, OUTPUT, Genome


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
def _layered_layout(genome: Genome) -> tuple[dict[int, int], list[int]]:
    """Assign each node a column = longest path from the input over *enabled*
    edges (a standard feed-forward layering), returning ``{node: column}`` and the
    topological order."""
    node_ids = {INPUT, OUTPUT} | {n.id for n in genome.nodes}
    enabled = [(c.src, c.dst) for c in genome.conns
               if c.enabled and c.src in node_ids and c.dst in node_ids]

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
    # Pin the output to the right-most column so it always reads as the head.
    depth[OUTPUT] = max(depth.values())
    return depth, order


def plot_genome(
    genome: Genome, out_path: str, *, spec=None, title: str = ""
) -> str:
    """Render the evolved network as a left-to-right graph. Returns ``out_path``.

    Nodes are the input, output and hidden units (labelled with their width /
    channels + kernel and activation); solid arrows are the enabled connections
    that make up the trained network, faint dotted arrows the disabled genes the
    genome still carries.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth, _ = _layered_layout(genome)
    columns: dict[int, list[int]] = {}
    for node, col in depth.items():
        columns.setdefault(col, []).append(node)

    # Position: x = column, y = evenly spread within the column (input/output
    # centred, hidden ordered by id for a stable picture).
    pos: dict[int, tuple[float, float]] = {}
    for col, members in columns.items():
        members = sorted(members)
        k = len(members)
        for i, node in enumerate(members):
            pos[node] = (float(col), (k - 1) / 2.0 - i)

    n_cols = max(depth.values()) + 1
    max_rows = max(len(m) for m in columns.values())
    fig, ax = plt.subplots(figsize=(max(6.0, 2.4 * n_cols), max(3.0, 1.5 * max_rows)))

    is_conv = genome.conv
    kernels = {n.id: n.kernel_size for n in genome.nodes}
    acts = {n.id: n.activation for n in genome.nodes}
    widths = {n.id: n.width for n in genome.nodes}

    def label(node: int) -> str:
        if node == INPUT:
            if spec is None:
                return "input"
            if spec.task == "modular":
                return f"input\n2×{genome.embed_dim} emb"
            if is_conv and spec.image_shape:
                c, h, w = spec.image_shape
                pooled = f" ÷{genome.conv_pool}" if genome.conv_pool > 1 else ""
                return f"input\nimage {c}×{h}×{w}{pooled}"
            return f"input\n{spec.input_dim}px"
        if node == OUTPUT:
            return "output" + (f"\n{spec.num_classes} classes" if spec else "")
        if is_conv:
            return f"#{node}\n{widths[node]}ch·k{kernels[node]}\n{acts[node]}"
        return f"#{node}\n{widths[node]}u·{acts[node]}"

    def style(node: int):
        if node == INPUT:
            return "#d9f0d3", "#2ca02c"
        if node == OUTPUT:
            return "#f7d5d5", "#d62728"
        return ("#e7dcf0", "#9467bd") if is_conv else ("#d6e5f3", "#1f77b4")

    # Edges first (behind the node boxes).
    for c in genome.conns:
        if c.src not in pos or c.dst not in pos:
            continue
        x0, y0 = pos[c.src]
        x1, y1 = pos[c.dst]
        if c.enabled:
            arrow = dict(arrowstyle="-|>", color="#555555", lw=1.4,
                         shrinkA=18, shrinkB=18, connectionstyle="arc3,rad=0.08")
        else:
            arrow = dict(arrowstyle="-|>", color="#cccccc", lw=0.8, ls=":",
                         shrinkA=18, shrinkB=18, alpha=0.7,
                         connectionstyle="arc3,rad=0.08")
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=arrow, zorder=1)

    # Node boxes.
    for node, (x, y) in pos.items():
        face, edge = style(node)
        ax.text(
            x, y, label(node), ha="center", va="center", fontsize=9, zorder=2,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=face, edgecolor=edge, lw=1.6),
        )

    ax.set_xlim(-0.6, n_cols - 0.4)
    ax.set_ylim(-max_rows / 2 - 0.8, max_rows / 2 + 0.8)
    ax.axis("off")
    ax.set_title(title or genome.summary(), fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
