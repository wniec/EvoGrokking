# EvoGrokking

**Evolving neural architectures that maximally induce grokking.**

Grokking is *delayed generalisation*: a network fits its training set almost
immediately, then continues training for a long time before it suddenly
generalises to held-out data. This project searches — with an evolutionary
algorithm — for the architecture *and* regularisation recipe that induces the
strongest grokking, across several datasets.

## What it does

The five main parts, one module each:

| Part | Module | Notes |
|------|--------|-------|
| **Grokking measurement** | [`metrics.py`](evogrokking/metrics.py) | area between the train/val **log-loss** curves, plus the bounded **accuracy-curve** area |
| **Datasets** | [`datasets.py`](evogrokking/datasets.py) | MNIST, FashionMNIST, and p-modular addition `(a+b) mod p` |
| **NN training** | [`train.py`](evogrokking/train.py) | full-batch Adam / AdamW / SGD, CUDA when available, max-iteration cap + grokking-aware early stopping |
| **Architecture evolution** | [`genome.py`](evogrokking/genome.py), [`models.py`](evogrokking/models.py), [`evolution.py`](evogrokking/evolution.py) | NEAT graph GA (arbitrary connections, optional **conv** filters), **evolved regularisation**, parallel evaluation |
| **Experiment endpoint** | [`experiment.py`](evogrokking/experiment.py), [`plots.py`](evogrokking/plots.py) | `train` / `evolve` / `retrain` CLI + learning-curve plots |

### Measuring grokking

The grokking *scale* is the normalised area between the log-loss curves:

```
grok_area = (1/T) · Σ_t  max(0, log(val_loss_t) − log(train_loss_t))
```

Both losses are clamped into `[1e-8, 1e8]` (and NaN/±inf from diverged runs mapped
into that range) before the log, so neither a perfectly fit split nor a blown-up
run can send the area to ±∞.

#### Area ≠ grokking: not rewarding overfitting

The area alone is *maximised by plain overfitting* — a model whose validation
loss stays high forever has the biggest, most permanent gap of all. So the
objective the search maximises credits the area **only for runs certified as
grokking**, on two independent axes (see `GrokkingMetrics.score`):

1. a **sharp generalisation gate** — a soft step on final validation accuracy at
   `--gen-threshold` (default 0.9); low-accuracy overfitters score ≈ 0;
2. a **val-loss-drop factor** `1 − exp(−(log peak − log final))` — ≈ 0 when the
   validation loss never falls back from its peak (overfitting), ≈ 1 when it
   plateaus high then collapses (grokking).

```
magnitude = grok_area + acc_weight · acc_area
score     = gate(final_val_acc) · drop(val_loss) · magnitude  +  test_weight · final_val_acc
```

The **magnitude** blends two views of the delayed gap: the log-loss area
and the **accuracy-curve area** `acc_area` — the
area between the train and val accuracy curves. `acc_area` is bounded in `[0, 1]`
and immune to loss-scale artefacts, so it's a robust second signal; `--acc-weight`
(default `5.0`, set `0` to disable) scales its `~[0, 0.5]` range up to the loss
area's. Both panels of the learning-curve plot shade their respective gap.

Final test accuracy stays a **low-weight supporting objective** (`--test-weight`,
default `0.1`). The metrics `generalised`, `grok_delay`, and `val_loss_drop` are
reported per run so overfitting is easy to spot. Concretely, a forced overfit
(no weight decay) has the **maximum** `acc_area` (≈1.0) *and* **3× the loss area**
of a genuine grokker, yet scores **0** — because it never generalises and its val
loss never drops, so the gate and drop factors zero both area terms.

For hard image subsets that can't reach 90 % val accuracy, lower
`--gen-threshold` accordingly.

### Run length: max iterations & early stopping

`--epochs` is the **maximum number of iterations** (a hard cap). Optionally,
training stops early once the run has clearly finished:

* `--target-val-acc 0.95` — stop the moment the model groks (val accuracy target);
* `--patience N` (`--min-delta`) — stop after `N` evaluations with no improvement
  in the *best-so-far* validation loss.

Tracking the best-so-far loss makes patience safe for grokking: the long
pre-grokking plateau doesn't trigger a stop until the loss has actually bottomed
out. Both flags work for `train` and `evolve`; early stopping is off unless one is
set.

### What evolves — a NEAT graph

Each genome is a **NEAT-style directed graph**: *node genes* (each a vector of
units with its own width + activation) and *connection genes* (`src → dst` edges
with an `enabled` flag and a global innovation number). Networks grow from a
minimal `input → output` seed via structural mutations — **add-node** (split an
edge) and **add-connection** (any acyclic edge, including skip/multi-path
connections) — so arbitrary topologies are reachable, not just stacked layers.
Connection *weights* are found by gradient descent; a connection is a learnable
`Linear(width[src], width[dst])` and a node sums its incoming edges, so multiple
paths compose naturally (see [`models.py`](evogrokking/models.py), which
evaluates the DAG in topological order).

Alongside the graph, each genome carries its training recipe — token-embedding
width, optimizer, learning rate, initial-weight scale, dropout, and — most
importantly for grokking — **weight-decay strength**. Crossover aligns
connection genes by innovation number (matching genes inherited randomly,
disjoint/excess from the fitter parent); selection is tournament-based with
elitism.

### Convolutional mode (image tasks)

Flattening 28×28 images to 784-vectors throws away spatial structure, so image
tasks can opt into a **convolutional graph** with `--conv`. In this mode each node
is a *spatial feature map* — its `width` gene becomes a **channel count** and it
gains an evolvable **`kernel_size`** (3/5/7) — and each edge is a **same-padding
`Conv2d`** (stride 1). Because every map keeps the input resolution, arbitrary
skip connections still align and sum, so the whole NEAT machinery (add-node,
add-connection, innovation crossover) is unchanged. The input node is the image
`(1, 28, 28)`; the output node is a global-average-pool + `Linear` classifier
head. The search then evolves the number of conv layers, their channels and
kernels, and the skip-connection topology. The modular / MLP path is untouched
(`--conv` is ignored for non-image tasks):

```bash
python main.py evolve --dataset mnist --conv --workers 4 \
    --generations 8 --population 16 --gen-threshold 0.7 --name mnist_conv
```

**Bounding memory.** Same-conv preserves the full resolution, so conv activation
memory grows as `batch × channels × H × W × nodes` — easily large under full-batch
training. Three controls keep it in check:

- **`--conv-pool` (default 2)** — average-pools the input once so all maps run at
  14×14 instead of 28×28, cutting peak activation memory ~4× (measured
  857 MB → 270 MB for the same net) while keeping every map the same size so skip
  connections still align.
- **`--mem-budget-mb` (default 1500)** — the search estimates each genome's
  activation footprint and **skips** (does not train, assigns worst fitness) any
  that exceed the budget, so evolution can never OOM. Set `0` to disable.
- **Tighter bounds** — conv channels are capped at 4–32 and conv nets to 6 hidden
  nodes.

### Parallel evaluation

Every fitness evaluation is an independent training run, so the population is
evaluated across worker processes with `--workers N` (a `spawn`-based
`ProcessPoolExecutor`, CUDA-safe; the dataset is shipped to each worker once).
Each genome trains under a deterministic per-genome seed, so the search returns
**identical results regardless of `--workers`** — only faster.

## Install

```bash
uv pip install -e .        # or: pip install -e .
```

Requires PyTorch (CUDA optional but auto-detected) and torchvision.

## Usage

Train a single baseline architecture and report its grokking metrics:

```bash
python main.py train --dataset modadd --p 31 --train-frac 0.5 --epochs 4000
python main.py train --dataset mnist  --train-size 1000 --epochs 2000
```

Run the evolutionary architecture search:

```bash
python main.py evolve --dataset modadd --p 31 --generations 12 \
    --population 24 --epochs 1500 --workers 4 --name modadd_search
```

This saves the winning individual to `runs/modadd_search/best.json` (plus the
generation history and the dataset used).

Reload that best individual, retrain it, and plot its learning curves:

```bash
python main.py retrain --from modadd_search          # reproduces the search run exactly
python main.py retrain --from modadd_search --epochs 8000   # or train it longer
```

**Reproducibility.** A whole run is deterministic from `--seed` (default **123**):
every genome trains under that one fixed seed (no per-genome offset), and CUDA is
put into deterministic mode. `retrain` reuses the run's saved seed, epoch budget
and eval cadence (from `meta.json`) and builds the model *after* seeding, so by
default it reproduces the search phase's best individual **bit-for-bit** — every
metric matches exactly. `--epochs N` trains the winner longer (the shared prefix
still matches); `--seed` tries a different initialisation.

`retrain` rebuilds the exact dataset the search used, writes the trained weights
(`best_model.pt`), the curves (`curves.json` + `curves.png`), and a
**`structure.png`** graph of the evolved network — nodes laid out left-to-right by
topological depth, labelled with their width / channels·kernel and activation,
with solid arrows for the enabled connections (including skip connections) and
faint dotted arrows for the disabled genes the genome still carries. On the
learning-curve plot's log-loss panel the shaded region between the train and
validation curves *is* the grokking area the search maximises:

![example learning curves](docs/example_curves.png)

`train --plot` produces the same plot for the hand-picked baseline. All commands
write results under `runs/<name>/` (loss/accuracy curves, best genome,
per-generation history as JSONL).

`python main.py <cmd> --help` lists every knob (population size, tournament size,
mutation rate, crossover probability, epochs-per-evaluation, `--test-weight`, …).

## Tests

```bash
python -m pytest -q          # or: PYTHONPATH=. python tests/test_evogrokking.py
```
