# EvoGrokking

**Evolving neural architectures that generalise *without* grokking.**

Grokking is *delayed generalisation*: a network fits its training set almost
immediately, then continues training for a long time before it suddenly
generalises to held-out data. This project searches — with a NEAT evolutionary
algorithm — for the **architecture** that generalises *as early as possible*
while still ending up accurate, on datasets whose train/test distribution shift
is what makes grokking happen in the first place.

## What it does

The main parts, one module each:

| Part | Module | Notes |
|------|--------|-------|
| **Grokking measurement + objective** | [`metrics.py`](evogrokking/metrics.py) | log-loss area, bounded accuracy-curve area, `gen_frac`; the **minimise-grokking** score |
| **Datasets** | [`datasets.py`](evogrokking/datasets.py), [`subclasses.py`](evogrokking/subclasses.py) | **distribution-shifted** MNIST/FashionMNIST (Carvalho et al. 2025), plus `(a+b) mod p` |
| **NN training** | [`train.py`](evogrokking/train.py) | full-batch Adam / AdamW / SGD, CUDA when available, max-iteration cap + grokking-aware early stopping |
| **Architecture evolution** | [`genome.py`](evogrokking/genome.py), [`neat_arch.py`](evogrokking/neat_arch.py), [`models.py`](evogrokking/models.py), [`evolution.py`](evogrokking/evolution.py) | **neat-python** GA with speciation (arbitrary connections, optional **conv** filters), parallel evaluation |
| **Fixed training recipe** | [`hyperparams.py`](evogrokking/hyperparams.py) | lr / weight decay / dropout / optimizer / init scale — set per run, **not** evolved |
| **Experiment endpoint** | [`experiment.py`](evogrokking/experiment.py), [`plots.py`](evogrokking/plots.py) | `train` / `evolve` / `retrain` CLI + learning-curve and network-structure plots |

### The objective: minimise grokking, keep accuracy high

The grokking *scale* is the normalised area between the log-loss curves:

```
grok_area = (1/T) · Σ_t  max(0, log(val_loss_t) − log(train_loss_t))
```

Both losses are clamped into `[1e-8, 1e8]` (and NaN/±inf from diverged runs mapped
into that range) before the log, so neither a perfectly fit split nor a blown-up
run can send the area to ±∞.

The search **minimises** grokking. The trap is that there are two trivial ways to
not grok — *never learn anything* (no gap at all) and *memorise forever* (a
permanent gap) — so the score pays the anti-grokking reward only through a sharp
**accuracy gate** at `--gen-threshold`:

```
gate      = sigmoid(30 · (final_val_acc − gen_threshold))
tightness = 1 − acc_area   # validation tracked training closely
speed     = 1 − gen_frac   # generalisation happened early
score     = acc_weight · final_val_acc
          + gate · (gap_weight · tightness + speed_weight · speed)
```

Below the gate an individual is left with only its (low) final accuracy, so
neither degenerate strategy pays. Above it, credit is proportional to how small
the train/validation gap was (`acc_area`, bounded in `[0, 1]`) and how early
validation accuracy first crossed the threshold (`gen_frac`, `1.0` if it never
did). The three weights are `--acc-weight`, `--gap-weight` and `--speed-weight`
(all default `1.0`).

`grok_magnitude`, `grok_delay`, `val_loss_drop` and `generalised` are still
reported per run so you can see how much grokking actually happened. Both panels
of the learning-curve plot shade the gap.

For hard image subsets that can't reach 90 % val accuracy, lower
`--gen-threshold` accordingly (shifted MNIST at 1 k training images tops out
around 80 %, so `--gen-threshold 0.75` is a reasonable setting).

### The datasets: an induced distribution shift

Following Carvalho et al., *"Grokking Explained: A Statistical Phenomenon"*
(2025), grokking is treated as a consequence of a **train/test distribution
shift**, and MNIST is adapted to induce one on purpose:

1. a small CNN is trained on the full training set; its penultimate layer is the
   **learned feature space** (the paper uses a ResNet — a smaller net is used
   here because the search runs hundreds of trainings, and the result is cached
   per dataset/seed);
2. each digit class is clustered in that space into `--n-subclasses` **latent
   subclasses**;
3. `--shifted-per-class` subclasses of every class are **under-sampled** in the
   *training* set to a fraction `--shift-frac` (`f`) of the rest, following the
   paper's Equation 1:

```
s_s = ⌈ f·γ_D / (f·γ_s + γ_r) ⌉       s_r = ⌊ γ_D / (f·γ_s + γ_r) ⌋
```

The validation split is the untouched test set. Train and test therefore hold the
same digits, in the same class proportions, drawn from **different distributions
over the latent space** — which is exactly the condition the paper identifies as
producing grokking. `--shift-frac 1.0` removes the shift (a balanced subsample);
`--shift-frac 0.0` drops those subclasses from training entirely.

`--dataset mnist` / `fashionmnist` are the shifted versions; `mnist_plain` /
`fashionmnist_plain` are the classic un-shifted ones.

### What evolves — a NEAT graph, architecture only

Reproduction is handled by [**neat-python**](https://neat-python.readthedocs.io)
(`ArchGenome` in [`neat_arch.py`](evogrokking/neat_arch.py)), which brings
speciation, stagnation handling and fitness sharing on top of the classic
add-node / add-connection structural mutations and innovation-aligned crossover.

Two things are bridged to make a NEAT graph a trainable net:

* **nodes are layers.** A neat-python node is one unit; ours is a whole layer, so
  the stock node gene is extended with evolvable `width` and `kernel_idx`
  attributes (the kernel is an *index* into `(3, 5, 7)` so mutation can only land
  on a valid odd size);
* **weights are not evolved.** Connection weights come from gradient descent — a
  connection is a learnable `Linear(width[src], width[dst])` and a node sums its
  incoming edges — so the stock `weight` gene is pinned to a constant and only
  `enabled` matters. [`models.py`](evogrokking/models.py) evaluates the DAG in
  topological order, so multiple paths and skip connections compose naturally.

**Only the architecture evolves.** The training recipe — learning rate, weight
decay, dropout, optimizer, initial-weight scale, embedding width — is *fixed for
the whole run* and shared by every individual
([`hyperparams.py`](evogrokking/hyperparams.py)), so a difference in fitness is
attributable to the graph rather than to a luckier regularisation setting.
Defaults follow the paper (`lr = 1e-3`, `weight_decay = 1e-4`, init scaled by 8);
override any of them with `--lr`, `--weight-decay`, `--dropout`, `--optimizer`,
`--init-scale`, `--embed-dim`. The modular task gets its own defaults, since it
needs strong weight decay to move at all.

The generated neat-python config is written to `runs/<name>/neat_config.ini`, so
the exact search settings are recorded with the results.

### Convolutional mode (image tasks)

Flattening 28×28 images to 784-vectors throws away spatial structure, so image
tasks can opt into a **convolutional graph** with `--conv`. In this mode each node
is a *spatial feature map* — its `width` gene becomes a **channel count** and it
gains an evolvable **kernel size** (3/5/7) — and each edge is a **same-padding
`Conv2d`** (stride 1). Because every map keeps the input resolution, arbitrary
skip connections still align and sum, so the whole NEAT machinery is unchanged.
The input node is the image `(1, 28, 28)`; the output node is a
global-average-pool + `Linear` classifier head. The modular / MLP path is
untouched (`--conv` is ignored for non-image tasks):

```bash
python main.py evolve --dataset mnist --conv --workers 4 \
    --generations 8 --population 16 --gen-threshold 0.75 --name mnist_conv
```

**Bounding memory.** Same-conv preserves the full resolution, so conv activation
memory grows as `batch × channels × H × W × nodes` — easily large under full-batch
training. Three controls keep it in check:

- **`--conv-pool` (default 2)** — average-pools the input once so all maps run at
  14×14 instead of 28×28, cutting peak activation memory ~4× while keeping every
  map the same size so skip connections still align.
- **`--mem-budget-mb` (default 1500)** — the search estimates each genome's
  activation footprint and **skips** (does not train, assigns worst fitness) any
  that exceed the budget, so evolution can never OOM. Set `0` to disable.
- **Tighter bounds** — conv channels are capped at 4–32.

### Run length: max iterations & early stopping

`--epochs` is the **maximum number of iterations** (a hard cap). Optionally,
training stops early once the run has clearly finished:

* `--target-val-acc 0.95` — stop the moment the target is reached;
* `--patience N` (`--min-delta`) — stop after `N` evaluations with no improvement
  in the *best-so-far* validation loss.

Tracking the best-so-far loss makes patience safe for grokking: the long
pre-grokking plateau doesn't trigger a stop until the loss has actually bottomed
out. Both flags work for `train` and `evolve`; early stopping is off unless one is
set.

### Parallel evaluation

Every fitness evaluation is an independent training run, so the population is
evaluated across worker processes with `--workers N` (a `spawn`-based
`ProcessPoolExecutor`, CUDA-safe; the dataset is shipped to each worker once).
Each genome trains under the run's fixed seed, so the search returns **identical
results regardless of `--workers`** — only faster.

## Install

```bash
uv pip install -e .        # or: pip install -e .
```

Requires PyTorch (CUDA optional but auto-detected), torchvision and neat-python.

## Usage

Train a single baseline architecture and report its grokking metrics:

```bash
python main.py train --dataset mnist  --train-size 1000 --epochs 2000 --plot
python main.py train --dataset modadd --p 31 --train-frac 0.5 --epochs 4000 --plot
```

Run the evolutionary architecture search:

```bash
python main.py evolve --dataset mnist --generations 12 --population 24 \
    --epochs 1500 --workers 4 --gen-threshold 0.75 --name mnist_search
```

This saves the winning individual to `runs/mnist_search/best.json` (plus the
generation history, the neat-python config, and the dataset settings used).

Reload that best individual, retrain it, and plot its learning curves:

```bash
python main.py retrain --from mnist_search          # reproduces the search run exactly
python main.py retrain --from mnist_search --epochs 8000   # or train it longer
```

**Reproducibility.** A whole run is deterministic from `--seed` (default **123**):
every genome trains under that one fixed seed (no per-genome offset), and CUDA is
put into deterministic mode. `retrain` reuses the run's saved seed, training
recipe, epoch budget and eval cadence (from `meta.json`) and builds the model
*after* seeding, so by default it reproduces the search phase's best individual
**bit-for-bit**. `--epochs N` trains the winner longer (the shared prefix still
matches); `--seed` tries a different initialisation.

`retrain` rebuilds the exact dataset the search used, writes the trained weights
(`best_model.pt`), the curves (`curves.json` + `curves.png`), and a
**`structure.png`** graph of the evolved network — nodes laid out left-to-right by
topological depth, labelled with their width / channels·kernel and activation,
with solid arrows for the enabled connections (including skip connections) and
faint dotted arrows for the disabled genes the genome still carries. On the
learning-curve plot's log-loss panel the shaded region between the train and
validation curves *is* the grokking area — which the search now works to make
**small**:

![example learning curves](docs/example_curves.png)

`train --plot` produces the same **pair** of plots for the hand-picked baseline —
`curves.png` and `structure.png` — so a single training run can be inspected
without going through a search first. All commands write results under
`runs/<name>/` (loss/accuracy curves, best genome, per-generation history as
JSONL).

`python main.py <cmd> --help` lists every knob (population size, objective
weights, the distribution-shift settings, the neat-python speciation parameters,
the fixed training recipe, …).

## Tests

```bash
python -m pytest -q          # or: PYTHONPATH=. python tests/test_evogrokking.py
```
