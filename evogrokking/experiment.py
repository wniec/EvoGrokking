"""The experiment-running endpoint.

A single CLI ties the pieces together.  Three sub-commands:

* ``train``   -- train one architecture on a dataset and report its grokking
  metrics.  With no genome given it trains a sensible hand-picked baseline, which
  is handy for sanity-checking that a dataset groks at all.  ``--plot`` draws both
  the learning curves and the network structure, as ``retrain`` does.
* ``evolve``  -- run the evolutionary architecture search (neat-python) and save
  the best genome, the full generation history, and the dataset used.
* ``retrain`` -- reload that saved best genome, train it from scratch (for
  longer) and plot its learning curves, saving the trained weights.

Two things to know about what the search optimises:

* it **minimises** grokking -- the winner is the architecture that generalises
  *earliest* while still reaching high validation accuracy (see
  :mod:`evogrokking.metrics`); and
* only the **architecture** is evolved.  The training recipe (learning rate,
  weight decay, dropout, optimizer, init scale, embedding width) is fixed for the
  whole run and set from the flags below, so every individual is trained the same
  way.

Results are written under ``runs/<name>/`` as JSON (plus ``curves.png`` and
``structure.png``) so they can be inspected or compared later.

Examples
--------
    python -m evogrokking.experiment train   --dataset mnist --epochs 4000 --plot
    python -m evogrokking.experiment evolve  --dataset mnist --generations 12 \
        --population 24 --workers 4 --name mnist_search
    python -m evogrokking.experiment retrain --from mnist_search --epochs 8000
"""

from __future__ import annotations

import argparse
import json
import os
import random

import torch

from evogrokking import datasets
from evogrokking.evolution import Evolution, EvolutionConfig
from evogrokking.genome import Genome
from evogrokking.hyperparams import OPTIMIZERS, Hyperparams
from evogrokking.plots import plot_curves, plot_genome
from evogrokking.train import EarlyStopping, default_device, train_and_evaluate

RUNS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs")


# --------------------------------------------------------------------------
def _dataset_load_kwargs(args) -> dict:
    kwargs = {"seed": args.seed}
    if not datasets.is_image(args.dataset):
        # The modular task keeps its historical default fraction.
        kwargs.update(
            p=args.p,
            train_frac=0.4 if args.train_frac is None else args.train_frac,
        )
        return kwargs

    kwargs.update(train_size=args.train_size, val_size=args.val_size)
    # ``--train-frac`` also applies to image datasets: when given it selects a
    # fraction of the full training set and overrides ``--train-size``.
    if args.train_frac is not None:
        kwargs["train_frac"] = args.train_frac
    if args.dataset.lower() in datasets.SHIFTED:
        kwargs.update(
            n_subclasses=args.n_subclasses,
            shifted_per_class=args.shifted_per_class,
            shift_frac=args.shift_frac,
            feature_epochs=args.feature_epochs,
        )
    return kwargs


def _load_dataset(args) -> datasets.Dataset:
    return datasets.load(args.dataset, **_dataset_load_kwargs(args))


def _hyperparams(args, dataset: datasets.Dataset) -> Hyperparams:
    """The run's fixed training recipe: per-task defaults plus any CLI overrides."""
    return Hyperparams.for_task(dataset.spec.task).with_overrides(
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        optimizer=args.optimizer,
        init_scale=args.init_scale,
    )


def _baseline_genome(dataset: datasets.Dataset, n_hidden: int) -> Genome:
    """The **big densely connected** starting network, as a genome.

    One neuron per input feature, ``n_hidden`` hidden neurons, one neuron per
    class, fully wired (including direct input -> output edges).  This is exactly
    the founding structure the search starts from, so ``train`` measures the
    baseline the search is trying to improve on.
    """
    return Genome.dense(
        dataset.spec.input_dim, dataset.spec.num_classes, n_hidden
    )


def _run_dir(name: str) -> str:
    path = os.path.join(RUNS_ROOT, name)
    os.makedirs(path, exist_ok=True)
    return path


def _make_early_stopping(args) -> EarlyStopping | None:
    if args.patience is None and args.target_val_acc is None:
        return None
    return EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        target_val_acc=args.target_val_acc,
    )


def _save_curves(out: str, result, eval_every: int) -> None:
    with open(os.path.join(out, "curves.json"), "w") as f:
        json.dump(
            {
                "train_losses": result.train_losses,
                "val_losses": result.val_losses,
                "train_accs": result.train_accs,
                "val_accs": result.val_accs,
                "eval_every": eval_every,
            },
            f,
        )


def _plot_result(out: str, result, eval_every: int, title: str) -> None:
    path = plot_curves(
        result.train_losses,
        result.val_losses,
        result.train_accs,
        result.val_accs,
        os.path.join(out, "curves.png"),
        title=title,
        eval_every=eval_every,
        grok_area=result.metrics.grok_area,
        acc_area=result.metrics.acc_area,
    )
    print(f"Saved learning-curve plot to {path}")


def _plot_structure(out: str, genome: Genome, spec, title: str) -> None:
    path = plot_genome(
        genome, os.path.join(out, "structure.png"), spec=spec, title=title
    )
    print(f"Saved network-structure graph to {path}")


# --------------------------------------------------------------------------
def cmd_train(args) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = default_device()
    dataset = _load_dataset(args)
    hp = _hyperparams(args, dataset)
    genome = _baseline_genome(dataset, args.hidden)
    print(
        f"Device: {device} | dataset: {dataset.spec.name} "
        f"({len(dataset.x_train)} train / {len(dataset.x_val)} val)\n"
        f"  arch: {genome.summary()}\n"
        f"  recipe: {hp.summary()}"
    )

    early_stopping = _make_early_stopping(args)
    result = train_and_evaluate(
        genome,
        dataset,
        hp=hp,
        epochs=args.epochs,
        eval_every=args.eval_every,
        device=device,
        log_every=max(1, args.epochs // 10),
        early_stopping=early_stopping,
        gen_threshold=args.gen_threshold,
    )
    if result.stopped_epoch is not None:
        print(f"\nEarly-stopped at epoch {result.stopped_epoch} (max {args.epochs}).")

    m = result.metrics
    print("\n=== Grokking metrics ===")
    for k, v in m.as_dict().items():
        print(f"  {k:16s}: {v:.4f}")

    out = _run_dir(args.name)
    with open(os.path.join(out, "result.json"), "w") as f:
        json.dump(
            {
                "genome": genome.as_dict(),
                "hyperparams": hp.as_dict(),
                "result": result.as_dict(),
            },
            f,
            indent=2,
        )
    _save_curves(out, result, args.eval_every)
    if args.plot:
        _plot_result(out, result, args.eval_every, f"{dataset.spec.name} (baseline)")
        _plot_structure(
            out,
            genome,
            dataset.spec,
            f"Dense baseline — {dataset.spec.name}\n{genome.summary()}",
        )
    print(f"\nSaved to {out}/")


def cmd_evolve(args) -> None:
    torch.manual_seed(args.seed)
    dataset = _load_dataset(args)
    hp = _hyperparams(args, dataset)
    out = _run_dir(args.name)
    logfile = os.path.join(out, "history.jsonl")
    open(logfile, "w").close()  # truncate

    config = EvolutionConfig(
        population_size=args.population,
        generations=args.generations,
        epochs_per_eval=args.epochs,
        eval_every=args.eval_every,
        seed=args.seed,
        workers=args.workers,
        n_hidden=args.hidden,
        initial_connection=args.initial_connection,
        initial_connection_fraction=args.initial_connection_fraction,
        structural_mutation_rounds=args.mutation_rounds,
        mem_budget_mb=(args.mem_budget_mb or None),  # 0 disables the guard
        hyperparams=hp,
        acc_weight=args.acc_weight,
        gap_weight=args.gap_weight,
        speed_weight=args.speed_weight,
        gen_threshold=args.gen_threshold,
        elitism=args.elitism,
        survival_threshold=args.survival_threshold,
        compatibility_threshold=args.compatibility_threshold,
        max_stagnation=args.max_stagnation,
        species_elitism=args.species_elitism,
        activation_mutate_rate=args.activation_mutate_rate,
        early_stop_patience=args.patience,
        early_stop_min_delta=args.min_delta,
        early_stop_target_val_acc=args.target_val_acc,
    )
    print(
        f"Device: {default_device()} | dataset: {dataset.spec.name} "
        f"({len(dataset.x_train)} train / {len(dataset.x_val)} val)\n"
        f"  search: pop={config.population_size} gens={config.generations} "
        f"epochs/eval={config.epochs_per_eval} workers={config.workers}\n"
        f"  start: dense {dataset.spec.input_dim}-{config.n_hidden}-"
        f"{dataset.spec.num_classes} ({config.initial_connection})\n"
        f"  recipe (fixed, not evolved): {hp.summary()}\n"
        f"  objective: minimise grokking "
        f"(acc x{config.acc_weight} + gap x{config.gap_weight} "
        f"+ speed x{config.speed_weight}, gate at {config.gen_threshold:.2f})"
    )

    evo = Evolution(
        dataset,
        config,
        logfile=logfile,
        config_path=os.path.join(out, "neat_config.ini"),
    )
    best = evo.run()

    print("\n=== Best (earliest-generalising) architecture ===")
    print(f"  fitness: {best.fitness:.4f}")
    print(f"  {best.genome.summary()}")
    if best.result is not None:
        for k, v in best.result.metrics.as_dict().items():
            print(f"  {k:16s}: {v:.4f}")

    with open(os.path.join(out, "best.json"), "w") as f:
        json.dump(best.as_dict(), f, indent=2)
    with open(os.path.join(out, "history_summary.json"), "w") as f:
        json.dump(evo.history, f, indent=2)
    # Record the dataset + training settings so `retrain` can rebuild the exact
    # task and reproduce the best individual's run identically.
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "load_kwargs": _dataset_load_kwargs(args),
                "hyperparams": hp.as_dict(),
                "seed": config.seed,
                "epochs": config.epochs_per_eval,
                "eval_every": config.eval_every,
                "gen_threshold": config.gen_threshold,
            },
            f,
            indent=2,
        )
    print(f"\nSaved best genome + history to {out}/")
    print(
        f"Reproduce & plot the winner with:\n"
        f"  python main.py retrain --from {args.name}\n"
        f"  (add --epochs N to train it longer)"
    )


def _resolve_run(ref: str) -> str:
    """Accept either a run name under ``runs/`` or a direct path to a run dir."""
    if os.path.isdir(ref):
        return ref
    candidate = os.path.join(RUNS_ROOT, ref)
    if os.path.isdir(candidate):
        return candidate
    raise SystemExit(f"no such run: {ref!r} (looked in {candidate})")


def cmd_retrain(args) -> None:
    """Load the best individual from a finished `evolve` run and retrain it.

    By default this reproduces the search phase's run of that individual exactly:
    it reuses the run's saved seed, training recipe, epoch budget and eval cadence
    (from ``meta.json``), so with the same genome + seed the training trajectory
    is identical.  Pass ``--epochs`` to train it *longer* (the shared prefix still
    matches), or ``--seed`` to try a different initialisation."""
    run = _resolve_run(getattr(args, "from"))
    with open(os.path.join(run, "best.json")) as f:
        best = json.load(f)
    genome = Genome.from_dict(best["genome"])

    # Rebuild the exact dataset + training settings the search used (meta.json),
    # so the run reproduces; fall back to CLI args for older runs without meta.
    meta = {}
    meta_path = os.path.join(run, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    if meta:
        dataset = datasets.load(meta["dataset"], **meta["load_kwargs"])
    else:
        dataset = _load_dataset(args)

    hp = (
        Hyperparams.from_dict(meta["hyperparams"])
        if "hyperparams" in meta
        else _hyperparams(args, dataset)
    )
    # --seed reproduces the search when it matches the search seed; default to the
    # saved seed so reproduction is automatic even if the search used a custom one.
    seed = meta.get("seed", args.seed)
    eval_every = meta.get("eval_every", args.eval_every)
    gen_threshold = meta.get("gen_threshold", args.gen_threshold)
    # Reproduce the search epoch budget unless the user asked to train longer.
    epochs = args.epochs if args.epochs is not None else meta.get("epochs", 8000)

    device = default_device()
    print(
        f"Device: {device} | run: {run} | dataset: {dataset.spec.name} | "
        f"seed={seed} epochs={epochs}\n"
        f"Retraining best genome: {genome.summary()}\n"
        f"  recipe: {hp.summary()}"
    )

    # Let train_and_evaluate build the model *after* seeding (as the search does)
    # so the initialisation is reproduced; return_model gives it back for saving.
    result = train_and_evaluate(
        genome,
        dataset,
        hp=hp,
        epochs=epochs,
        eval_every=eval_every,
        device=device,
        log_every=max(1, epochs // 10),
        early_stopping=_make_early_stopping(args),
        seed=seed,
        gen_threshold=gen_threshold,
        return_model=True,
    )
    model = result.model
    if result.stopped_epoch is not None:
        print(f"\nEarly-stopped at epoch {result.stopped_epoch} (max {epochs}).")

    saved_score = best.get("fitness")
    print("\n=== Grokking metrics (retrained) ===")
    for k, v in result.metrics.as_dict().items():
        print(f"  {k:16s}: {v:.4f}")
    if saved_score is not None:
        print(f"  (search-phase fitness was {saved_score:.4f})")

    _save_curves(run, result, eval_every)
    torch.save(model.state_dict(), os.path.join(run, "best_model.pt"))
    with open(os.path.join(run, "retrain_result.json"), "w") as f:
        json.dump(
            {
                "genome": genome.as_dict(),
                "hyperparams": hp.as_dict(),
                "result": result.as_dict(),
            },
            f,
            indent=2,
        )
    _plot_result(run, result, eval_every, f"{dataset.spec.name} — evolved best")
    _plot_structure(
        run,
        genome,
        dataset.spec,
        f"Evolved architecture — {dataset.spec.name}\n{genome.summary()}",
    )
    print(f"\nSaved retrained weights + curves + structure to {run}/")


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evogrokking",
        description="Evolve neural architectures that generalise without grokking.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument(
            "--dataset",
            default="mnist",
            help="mnist | fashionmnist (distribution-shifted, the default form) | "
            "mnist_plain | fashionmnist_plain (classic) | modadd (default: mnist)",
        )
        p.add_argument(
            "--seed",
            type=int,
            default=123,
            help="global RNG seed; a whole run is reproducible from it, so "
            "`retrain` with the same seed reproduces the search's best individual "
            "(default: 123)",
        )
        p.add_argument("--eval-every", type=int, default=2)
        p.add_argument(
            "--hidden",
            type=int,
            default=32,
            help="hidden neurons in the big densely connected starting network "
            "(default: 32). This sets the gene count -- roughly "
            "(inputs + outputs) x hidden + inputs x outputs connections per "
            "genome -- and neat-python's bookkeeping grows ~quadratically in it, "
            "so raise it only if you can afford the wall-clock (see README)",
        )
        p.add_argument(
            "--mem-budget-mb",
            type=float,
            default=1500.0,
            help="evolve: skip genomes whose estimated activation memory exceeds "
            "this many MB, so the search never OOMs (default: 1500; 0 disables)",
        )
        p.add_argument(
            "--gen-threshold",
            type=float,
            default=0.9,
            help="validation accuracy a run must reach before it earns any "
            "anti-grokking credit (default: 0.9; lower for hard image subsets)",
        )
        # early stopping (shared by train / evolve; disabled unless one is set)
        p.add_argument(
            "--patience",
            type=int,
            default=None,
            help="stop after this many evaluations without val-loss improvement "
            "(in units of --eval-every epochs; default: off)",
        )
        p.add_argument(
            "--min-delta",
            type=float,
            default=1e-4,
            help="minimum val-loss improvement to reset patience (default: 1e-4)",
        )
        p.add_argument(
            "--target-val-acc",
            type=float,
            default=None,
            help="stop as soon as validation accuracy reaches this value (default: off)",
        )

        # -- the fixed training recipe (no longer evolved) ------------------
        hg = p.add_argument_group(
            "training recipe (fixed for the run; shared by every individual)"
        )
        hg.add_argument("--lr", type=float, default=None, help="learning rate")
        hg.add_argument(
            "--weight-decay", type=float, default=None, help="optimizer weight decay"
        )
        hg.add_argument("--dropout", type=float, default=None)
        hg.add_argument("--optimizer", choices=OPTIMIZERS, default=None)
        hg.add_argument(
            "--init-scale",
            type=float,
            default=None,
            help="multiply every initial weight by this (large init aids grokking)",
        )

        # modular-task knobs
        mg = p.add_argument_group("modular task")
        mg.add_argument("--p", type=int, default=97, help="modulus for modadd")
        # image-task knobs
        ig = p.add_argument_group("image tasks")
        p.add_argument(
            "--train-frac",
            type=float,
            default=None,
            help="fraction of the full dataset used for training (default 0.4 for "
            "modadd). Also applies to image datasets, where it overrides --train-size.",
        )
        ig.add_argument(
            "--train-size",
            type=int,
            default=1000,
            help="number of training images (ignored if --train-frac is set)",
        )
        ig.add_argument("--val-size", type=int, default=2000)

        # -- the distribution shift (Carvalho et al. 2025) -----------------
        sg = p.add_argument_group(
            "distribution shift (shifted image datasets only)"
        )
        sg.add_argument(
            "--n-subclasses",
            type=int,
            default=4,
            help="latent subclasses to cluster each class into (default: 4)",
        )
        sg.add_argument(
            "--shifted-per-class",
            type=int,
            default=1,
            help="how many subclasses of each class are under-sampled (default: 1)",
        )
        sg.add_argument(
            "--shift-frac",
            type=float,
            default=0.05,
            help="fraction f of Equation 1: how strongly those subclasses are "
            "under-sampled. 1.0 = no shift, 0.0 = removed entirely (default: 0.05)",
        )
        sg.add_argument(
            "--feature-epochs",
            type=int,
            default=2,
            help="epochs to train the feature extractor used for clustering; the "
            "result is cached per dataset/seed (default: 2)",
        )

    t = sub.add_parser("train", help="train one architecture and report grokking")
    add_common(t)
    t.add_argument(
        "--epochs", type=int, default=4000, help="maximum number of iterations"
    )
    t.add_argument("--name", default="baseline")
    t.add_argument(
        "--plot",
        action="store_true",
        help="also save the learning-curve plot and the network-structure graph (PNG)",
    )
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evolve", help="evolutionary architecture search (neat-python)")
    add_common(e)
    e.add_argument(
        "--epochs",
        type=int,
        default=1500,
        help="maximum iterations per fitness evaluation",
    )
    e.add_argument("--generations", type=int, default=10)
    e.add_argument("--population", type=int, default=40)
    e.add_argument(
        "--workers",
        type=int,
        default=1,
        help="evaluate the population across this many processes (default: 1)",
    )

    og = e.add_argument_group("objective (minimise grokking at high accuracy)")
    og.add_argument(
        "--acc-weight",
        type=float,
        default=1.0,
        help="weight on final validation accuracy (default: 1.0)",
    )
    og.add_argument(
        "--gap-weight",
        type=float,
        default=1.0,
        help="reward for a small train/validation accuracy gap, i.e. little "
        "delayed generalisation (default: 1.0)",
    )
    og.add_argument(
        "--speed-weight",
        type=float,
        default=1.0,
        help="reward for reaching the generalisation threshold early (default: 1.0)",
    )

    ng = e.add_argument_group("neat-python")
    ng.add_argument(
        "--initial-connection",
        default="full_direct",
        choices=["full_direct", "full_nodirect"],
        help="how the dense starting network is wired (default: full_direct, i.e. "
        "input->hidden, hidden->output and input->output)",
    )
    ng.add_argument(
        "--initial-connection-fraction",
        type=float,
        default=1.0,
        help="give each founding genome its own random fraction of that wiring "
        "(default: 1.0 = fully wired). Below 1.0 the founders differ from each "
        "other, which is the cheapest way to get diversity into generation 0 -- "
        "with full wiring every founder is identical and the first generation "
        "trains the same network --population times",
    )
    ng.add_argument(
        "--mutation-rounds",
        type=int,
        default=1,
        help="repeat the NEAT add/delete operators this many times per genome "
        "(default: 1). neat-python changes at most one gene of each kind per "
        "generation, which is a negligible step for a dense genome of tens of "
        "thousands of genes -- raise this so the step scales with the genome",
    )
    ng.add_argument(
        "--activation-mutate-rate",
        type=float,
        default=0.05,
        help="per-neuron chance of swapping activation function (default: 0.05)",
    )
    ng.add_argument("--elitism", type=int, default=2)
    ng.add_argument("--survival-threshold", type=float, default=0.2)
    ng.add_argument(
        "--compatibility-threshold",
        type=float,
        default=0.2,
        help="genomic-distance threshold for splitting species. neat-python "
        "normalises distance per gene, so this is a *mean per-gene* difference "
        "and is far below textbook NEAT's ~3.0 (default: 0.2)",
    )
    ng.add_argument("--max-stagnation", type=int, default=15)
    ng.add_argument("--species-elitism", type=int, default=2)

    e.add_argument("--name", default="search")
    e.set_defaults(func=cmd_evolve)

    r = sub.add_parser(
        "retrain",
        help="load an evolved run's best genome, train it and plot learning curves",
    )
    add_common(r)
    r.add_argument(
        "--from",
        required=True,
        metavar="RUN",
        help="finished evolve run to load (name under runs/, or a path)",
    )
    r.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="max iterations for the retrain; omit to reproduce the search's "
        "epoch budget exactly, or set higher to train the winner longer",
    )
    r.set_defaults(func=cmd_retrain)

    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
