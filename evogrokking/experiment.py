"""The experiment-running endpoint.

A single CLI ties the pieces together.  Three sub-commands:

* ``train``   -- train one architecture on a dataset and report its grokking
  metrics.  With no genome given it trains a sensible hand-picked baseline, which
  is handy for sanity-checking that a dataset groks at all.
* ``evolve``  -- run the evolutionary architecture search and save the best
  grokking-inducing genome, the full generation history, and the dataset used.
* ``retrain`` -- reload that saved best genome, train it from scratch (for
  longer) and plot its learning curves, saving the trained weights.

Results are written under ``runs/<name>/`` as JSON (+ a ``curves.png`` plot) so
they can be inspected or compared later.

Examples
--------
    python -m evogrokking.experiment train   --dataset modadd --epochs 4000 --plot
    python -m evogrokking.experiment evolve  --dataset modadd --generations 12 \
        --population 24 --workers 4 --name modadd_search
    python -m evogrokking.experiment retrain --from modadd_search --epochs 8000
"""

from __future__ import annotations

import argparse
import json
import os
import random

import torch

from evogrokking import datasets
from evogrokking.evolution import Evolution, EvolutionConfig
from evogrokking.genome import INPUT, OUTPUT, ConnGene, Genome, Innovations, NodeGene
from evogrokking.plots import plot_curves, plot_genome
from evogrokking.train import EarlyStopping, default_device, train_and_evaluate

RUNS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runs")


# --------------------------------------------------------------------------
def _dataset_load_kwargs(args) -> dict:
    kwargs = {"seed": args.seed}
    if args.dataset in ("modadd", "modular", "modular_addition"):
        kwargs.update(p=args.p, train_frac=args.train_frac)
    else:
        kwargs.update(train_size=args.train_size, val_size=args.val_size)
    return kwargs


def _load_dataset(args) -> datasets.Dataset:
    return datasets.load(args.dataset, **_dataset_load_kwargs(args))


def _baseline_genome(
    dataset: datasets.Dataset, conv: bool = False, conv_pool: int = 1
) -> Genome:
    """A hand-picked single-hidden-node graph: known to grok on the modular task,
    and a reasonable small-data image baseline otherwise.  Built directly as a
    NEAT graph (input -> hidden -> output) so it round-trips through the same code
    path as evolved genomes.  With ``conv`` the hidden node is a conv feature map."""
    if dataset.spec.task == "modular":
        width, act, kernel = 256, "relu", 3
        wd, do, opt, lr, init, emb = 1.0, 0.0, "adamw", 1e-2, 1.0, 128
    elif conv:
        width, act, kernel = 16, "relu", 3  # 16 conv channels, 3x3 filters
        wd, do, opt, lr, init, emb = 1e-2, 0.0, "adamw", 1e-3, 1.0, 64
    else:
        width, act, kernel = 256, "relu", 3
        wd, do, opt, lr, init, emb = 1e-2, 0.0, "adamw", 1e-3, 3.0, 64

    innov = Innovations()
    c_in_out = innov.conn(INPUT, OUTPUT)
    hid = innov.split_node(c_in_out)
    return Genome(
        nodes=(NodeGene(hid, width, act, kernel),),
        conns=(
            ConnGene(c_in_out, INPUT, OUTPUT, False),
            ConnGene(innov.conn(INPUT, hid), INPUT, hid, True),
            ConnGene(innov.conn(hid, OUTPUT), hid, OUTPUT, True),
        ),
        embed_dim=emb,
        weight_decay=wd,
        dropout=do,
        optimizer=opt,
        lr=lr,
        init_scale=init,
        conv=conv and dataset.spec.task == "image",
        conv_pool=conv_pool,
    )


def _conv_enabled(args, dataset: datasets.Dataset) -> bool:
    """Convolutions only make sense for image tasks; warn if asked for elsewhere."""
    if not getattr(args, "conv", False):
        return False
    if dataset.spec.task != "image":
        print(f"note: --conv ignored for non-image task {dataset.spec.name!r}")
        return False
    return True


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


# --------------------------------------------------------------------------
def cmd_train(args) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = default_device()
    dataset = _load_dataset(args)
    conv = _conv_enabled(args, dataset)
    genome = _baseline_genome(dataset, conv=conv, conv_pool=args.conv_pool)
    print(f"Device: {device} | dataset: {dataset.spec.name} | {genome.summary()}")

    early_stopping = _make_early_stopping(args)
    result = train_and_evaluate(
        genome,
        dataset,
        epochs=args.epochs,
        eval_every=args.eval_every,
        device=device,
        log_every=max(1, args.epochs // 10),
        early_stopping=early_stopping,
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
            {"genome": genome.as_dict(), "result": result.as_dict()}, f, indent=2
        )
    _save_curves(out, result, args.eval_every)
    if args.plot:
        _plot_result(out, result, args.eval_every, f"{dataset.spec.name} (baseline)")
    print(f"\nSaved to {out}/")


def cmd_evolve(args) -> None:
    torch.manual_seed(args.seed)
    dataset = _load_dataset(args)
    out = _run_dir(args.name)
    logfile = os.path.join(out, "history.jsonl")
    open(logfile, "w").close()  # truncate

    config = EvolutionConfig(
        population_size=args.population,
        generations=args.generations,
        elitism=args.elitism,
        tournament_size=args.tournament,
        crossover_prob=args.crossover,
        mutation_rate=args.mutation,
        epochs_per_eval=args.epochs,
        eval_every=args.eval_every,
        test_weight=args.test_weight,
        acc_weight=args.acc_weight,
        gen_threshold=args.gen_threshold,
        seed=args.seed,
        workers=args.workers,
        allow_conv=_conv_enabled(args, dataset),
        conv_pool=args.conv_pool,
        mem_budget_mb=(args.mem_budget_mb or None),  # 0 disables the guard
        early_stop_patience=args.patience,
        early_stop_min_delta=args.min_delta,
        early_stop_target_val_acc=args.target_val_acc,
    )
    print(
        f"Device: {default_device()} | dataset: {dataset.spec.name} | "
        f"pop={config.population_size} gens={config.generations} "
        f"epochs/eval={config.epochs_per_eval} workers={config.workers} "
        f"conv={config.allow_conv}"
    )

    evo = Evolution(dataset, config, logfile=logfile)
    best = evo.run()

    print("\n=== Best grokking-inducing genome ===")
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
                "seed": config.seed,
                "epochs": config.epochs_per_eval,
                "eval_every": config.eval_every,
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
    it reuses the run's saved seed, epoch budget and eval cadence (from
    ``meta.json``), so with the same genome + seed the training trajectory is
    identical.  Pass ``--epochs`` to train it *longer* (the shared prefix still
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

    # --seed reproduces the search when it matches the search seed; default to the
    # saved seed so reproduction is automatic even if the search used a custom one.
    seed = meta.get("seed", args.seed)
    eval_every = meta.get("eval_every", args.eval_every)
    # Reproduce the search epoch budget unless the user asked to train longer.
    epochs = args.epochs if args.epochs is not None else meta.get("epochs", 8000)

    device = default_device()
    print(
        f"Device: {device} | run: {run} | dataset: {dataset.spec.name} | "
        f"seed={seed} epochs={epochs}\n"
        f"Retraining best genome: {genome.summary()}"
    )

    # Let train_and_evaluate build the model *after* seeding (as the search does)
    # so the initialisation is reproduced; return_model gives it back for saving.
    result = train_and_evaluate(
        genome,
        dataset,
        epochs=epochs,
        eval_every=eval_every,
        device=device,
        log_every=max(1, epochs // 10),
        early_stopping=_make_early_stopping(args),
        seed=seed,
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
        json.dump({"genome": genome.as_dict(), "result": result.as_dict()}, f, indent=2)
    _plot_result(run, result, eval_every, f"{dataset.spec.name} — evolved best")
    # Draw the evolved network structure.
    struct_path = plot_genome(
        genome,
        os.path.join(run, "structure.png"),
        spec=dataset.spec,
        title=f"Evolved architecture — {dataset.spec.name}\n{genome.summary()}",
    )
    print(f"Saved network-structure graph to {struct_path}")
    print(f"\nSaved retrained weights + curves + structure to {run}/")


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evogrokking",
        description="Evolve neural architectures that maximally induce grokking.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument(
            "--dataset",
            default="modadd",
            help="modadd | mnist | fashionmnist (default: modadd)",
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
            "--conv",
            action="store_true",
            help="image tasks only: evolve/use convolutional feature maps (nodes "
            "become spatial channels, edges become same-padding conv filters)",
        )
        p.add_argument(
            "--conv-pool",
            type=int,
            default=2,
            help="conv mode: downsample the input by this factor to cap activation "
            "memory (1 = full resolution; default: 2)",
        )
        p.add_argument(
            "--mem-budget-mb",
            type=float,
            default=1500.0,
            help="evolve: skip genomes whose estimated activation memory exceeds "
            "this many MB, so the search never OOMs (default: 1500; 0 disables)",
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
            help="stop as soon as validation accuracy reaches this value, i.e. the "
            "model has grokked (default: off)",
        )
        # modular-task knobs
        p.add_argument("--p", type=int, default=97, help="modulus for modadd")
        p.add_argument("--train-frac", type=float, default=0.4)
        # image-task knobs
        p.add_argument("--train-size", type=int, default=1000)
        p.add_argument("--val-size", type=int, default=2000)

    t = sub.add_parser("train", help="train one architecture and report grokking")
    add_common(t)
    t.add_argument(
        "--epochs", type=int, default=4000, help="maximum number of iterations"
    )
    t.add_argument("--name", default="baseline")
    t.add_argument(
        "--plot", action="store_true", help="also save a learning-curve plot (PNG)"
    )
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evolve", help="evolutionary architecture search")
    add_common(e)
    e.add_argument(
        "--epochs",
        type=int,
        default=1500,
        help="maximum iterations per fitness evaluation",
    )
    e.add_argument("--generations", type=int, default=10)
    e.add_argument("--population", type=int, default=40)
    e.add_argument("--elitism", type=int, default=4)
    e.add_argument("--tournament", type=int, default=3)
    e.add_argument("--crossover", type=float, default=0.6)
    e.add_argument("--mutation", type=float, default=0.3)
    e.add_argument(
        "--workers",
        type=int,
        default=1,
        help="evaluate the population across this many processes (default: 1)",
    )
    e.add_argument(
        "--test-weight",
        type=float,
        default=0.1,
        help="low weight on final test accuracy (supporting objective)",
    )
    e.add_argument(
        "--gen-threshold",
        type=float,
        default=0.9,
        help="validation accuracy a run must reach to be credited as grokking "
        "rather than overfitting (default: 0.9; lower for hard image subsets)",
    )
    e.add_argument(
        "--acc-weight",
        type=float,
        default=5.0,
        help="weight on the accuracy-curve area within the grokking magnitude "
        "(default: 5.0; set 0 to use only the loss area)",
    )
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
