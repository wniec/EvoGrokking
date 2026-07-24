"""Evolutionary architecture search for *minimal* grokking.

A NEAT genetic algorithm -- supplied by
`neat-python <https://neat-python.readthedocs.io>`_, see
:mod:`evogrokking.neat_arch` -- searches the space of network *graphs* for the
architecture that generalises **as early as possible** while still ending up
accurate.  Each individual's fitness is obtained by actually training it
(:func:`~evogrokking.train.train_and_evaluate`) and scoring the run with
:meth:`~evogrokking.metrics.GrokkingMetrics.score`, which rewards high final
accuracy, a small train/validation gap and early generalisation.

Only the architecture evolves.  The training recipe
(:class:`~evogrokking.hyperparams.Hyperparams`) is fixed for the whole run and
shared by every individual, so differences in fitness are attributable to the
graph rather than to a luckier learning rate.

Because every fitness evaluation is an independent training run, the population
is evaluated **in parallel** across worker processes when ``workers > 1`` (a
``ProcessPoolExecutor`` with the ``spawn`` start method, so it is CUDA-safe).
Every genome is trained under the same fixed ``config.seed``, so a run is fully
reproducible: results do not depend on how the work is scheduled, and the best
individual can be retrained identically afterwards from that same seed.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Callable

import neat
import torch

from evogrokking.datasets import Dataset
from evogrokking.genome import Genome
from evogrokking.hyperparams import Hyperparams
from evogrokking.models import estimated_activation_mb
from evogrokking.neat_arch import ArchGenome, to_genome, write_config
from evogrokking.train import (
    EarlyStopping,
    TrainResult,
    default_device,
    train_and_evaluate,
)


@dataclass
class Individual:
    genome: Genome
    fitness: float = float("-inf")
    result: TrainResult | None = None

    def as_dict(self) -> dict:
        d = {"genome": self.genome.as_dict(), "fitness": self.fitness}
        if self.result is not None:
            d["metrics"] = self.result.as_dict()
        return d


@dataclass
class EvolutionConfig:
    population_size: int = 20
    generations: int = 10
    epochs_per_eval: int = 1500  # maximum iterations per fitness evaluation
    eval_every: int = 2
    seed: int = 0
    workers: int = 1  # >1 evaluates the population across that many processes
    allow_conv: bool = False  # image tasks: evolve convolutional feature maps
    conv_pool: int = 1  # conv mode: input downsample factor (memory knob)
    mem_budget_mb: float | None = None  # skip genomes whose est. activation memory exceeds this
    hyperparams: Hyperparams = field(default_factory=Hyperparams)

    # -- objective weights (see GrokkingMetrics.score) ---------------------
    acc_weight: float = 1.0  # weight on final validation accuracy
    gap_weight: float = 1.0  # reward for a small train/validation accuracy gap
    speed_weight: float = 1.0  # reward for generalising early
    gen_threshold: float = 0.9  # val-acc a run must reach to unlock those rewards

    # -- neat-python knobs -------------------------------------------------
    elitism: int = 2
    survival_threshold: float = 0.2
    compatibility_threshold: float = 3.0
    max_stagnation: int = 15
    species_elitism: int = 2
    conn_add_prob: float = 0.5
    conn_delete_prob: float = 0.2
    node_add_prob: float = 0.3
    node_delete_prob: float = 0.15

    # Early stopping (applied to each individual's training run). ``patience`` is
    # counted in evaluations (i.e. units of ``eval_every`` epochs); None disables.
    early_stop_patience: int | None = None
    early_stop_min_delta: float = 1e-4
    early_stop_target_val_acc: float | None = None


# --------------------------------------------------------------------------
# Parallel-evaluation workers.  The dataset is pickled to each worker once (via
# the pool initializer), not per task; each task then just trains one genome.
# --------------------------------------------------------------------------
_W_DATASET: Dataset | None = None
_W_DEVICE: torch.device | None = None


def _init_worker(dataset: Dataset, device_str: str) -> None:
    global _W_DATASET, _W_DEVICE
    torch.set_num_threads(1)  # avoid oversubscribing CPUs across processes
    _W_DATASET = dataset
    _W_DEVICE = torch.device(device_str)


def _score(result: TrainResult, ec: dict) -> float:
    return result.metrics.score(
        acc_weight=ec["acc_weight"],
        gap_weight=ec["gap_weight"],
        speed_weight=ec["speed_weight"],
        gen_threshold=ec["gen_threshold"],
    )


def _eval_worker(task: tuple[Genome, dict]):
    genome, ec = task
    assert _W_DATASET is not None and _W_DEVICE is not None
    result = _train_one(genome, _W_DATASET, _W_DEVICE, ec)
    return genome.id, result, _score(result, ec)


def _train_one(genome: Genome, dataset: Dataset, device, ec: dict) -> TrainResult:
    early_stopping = None
    if ec["es_patience"] is not None or ec["es_target"] is not None:
        early_stopping = EarlyStopping(
            patience=ec["es_patience"],
            min_delta=ec["es_min_delta"],
            target_val_acc=ec["es_target"],
        )
    return train_and_evaluate(
        genome,
        dataset,
        hp=Hyperparams.from_dict(ec["hyperparams"]),
        epochs=ec["epochs"],
        eval_every=ec["eval_every"],
        device=device,
        early_stopping=early_stopping,
        seed=ec["seed"],
        gen_threshold=ec["gen_threshold"],
    )


# --------------------------------------------------------------------------
class Evolution:
    """Drives a neat-python population, training every genome to score it."""

    # Genomes over the memory budget are not trained; they get a fitness below
    # any real score so the search abandons them (rather than risking an OOM).
    _OVER_BUDGET_FITNESS = -1e6

    def __init__(
        self,
        dataset: Dataset,
        config: EvolutionConfig | None = None,
        device: torch.device | None = None,
        on_generation: Callable[[int, list[Individual]], None] | None = None,
        logfile: str | None = None,
        config_path: str | None = None,
    ):
        self.dataset = dataset
        self.config = config or EvolutionConfig()
        self.device = device or default_device()
        self.on_generation = on_generation
        self.logfile = logfile
        # neat-python is configured from a file; write one next to the results so
        # the exact search settings are recorded with the run.  With no run
        # directory to write into, fall back to a scratch dir rather than
        # dropping the file into the current working directory.
        if config_path is None:
            run_dir = os.path.dirname(logfile) if logfile else tempfile.mkdtemp()
            config_path = os.path.join(run_dir, "neat_config.ini")
        self.config_path = config_path
        self.history: list[dict] = []
        self.best: Individual | None = None
        self._generation = 0
        self._pool: ProcessPoolExecutor | None = None

    # -- helpers -----------------------------------------------------------
    def _eval_params(self) -> dict:
        cfg = self.config
        return {
            "epochs": cfg.epochs_per_eval,
            "eval_every": cfg.eval_every,
            "acc_weight": cfg.acc_weight,
            "gap_weight": cfg.gap_weight,
            "speed_weight": cfg.speed_weight,
            "gen_threshold": cfg.gen_threshold,
            "seed": cfg.seed,
            "hyperparams": cfg.hyperparams.as_dict(),
            "es_patience": cfg.early_stop_patience,
            "es_min_delta": cfg.early_stop_min_delta,
            "es_target": cfg.early_stop_target_val_acc,
        }

    def _neat_config(self) -> neat.Config:
        cfg = self.config
        write_config(
            self.config_path,
            pop_size=cfg.population_size,
            conv=cfg.allow_conv,
            elitism=cfg.elitism,
            survival_threshold=cfg.survival_threshold,
            compatibility_threshold=cfg.compatibility_threshold,
            max_stagnation=cfg.max_stagnation,
            species_elitism=cfg.species_elitism,
            conn_add_prob=cfg.conn_add_prob,
            conn_delete_prob=cfg.conn_delete_prob,
            node_add_prob=cfg.node_add_prob,
            node_delete_prob=cfg.node_delete_prob,
        )
        return neat.Config(
            ArchGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_path,
        )

    def _over_budget(self, genome: Genome) -> bool:
        budget = self.config.mem_budget_mb
        if budget is None:
            return False
        mb = estimated_activation_mb(
            genome,
            self.dataset.spec,
            len(self.dataset.x_train),
            embed_dim=self.config.hyperparams.embed_dim,
        )
        return mb > budget

    # -- fitness -----------------------------------------------------------
    def _evaluate(self, neat_genomes, neat_config) -> list[Individual]:
        """neat-python's fitness callback: train every genome and score it."""
        cfg = self.config
        ec = self._eval_params()

        individuals: list[tuple[object, Individual]] = []
        pending: list[Individual] = []
        skipped = 0
        for _gid, ng in neat_genomes:
            genome = to_genome(ng, conv=cfg.allow_conv, conv_pool=cfg.conv_pool)
            ind = Individual(genome)
            individuals.append((ng, ind))
            if self._over_budget(genome):
                ind.fitness = self._OVER_BUDGET_FITNESS
                skipped += 1
            else:
                pending.append(ind)
        if skipped:
            print(
                f"  (skipped {skipped} genome(s) over "
                f"{cfg.mem_budget_mb:.0f} MB memory budget)"
            )

        if self._pool is None:
            for ind in pending:
                ind.result = _train_one(ind.genome, self.dataset, self.device, ec)
                ind.fitness = _score(ind.result, ec)
        else:
            by_id = {ind.genome.id: ind for ind in pending}
            tasks = [(ind.genome, ec) for ind in pending]
            for gid, result, fitness in self._pool.map(_eval_worker, tasks):
                ind = by_id[gid]
                ind.result = result
                ind.fitness = fitness

        # Hand the scores back to neat-python, which drives selection with them.
        for ng, ind in individuals:
            ng.fitness = float(ind.fitness)
        return [ind for _ng, ind in individuals]

    # -- main loop ---------------------------------------------------------
    def run(self) -> Individual:
        cfg = self.config
        neat_config = self._neat_config()
        population = neat.Population(neat_config, seed=cfg.seed)

        if cfg.workers > 1:
            ctx = get_context("spawn")
            self._pool = ProcessPoolExecutor(
                max_workers=cfg.workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(self.dataset, str(self.device)),
            )

        def fitness_function(genomes, config) -> None:
            t0 = time.time()
            individuals = self._evaluate(genomes, config)
            individuals.sort(key=lambda i: i.fitness, reverse=True)
            gen_best = individuals[0]
            if self.best is None or gen_best.fitness > self.best.fitness:
                self.best = gen_best
            self._record(self._generation, individuals, time.time() - t0)
            if self.on_generation:
                self.on_generation(self._generation, individuals)
            self._generation += 1

        try:
            population.run(fitness_function, cfg.generations)
        finally:
            if self._pool is not None:
                self._pool.shutdown()
                self._pool = None

        assert self.best is not None
        return self.best

    # -- logging -----------------------------------------------------------
    def _record(self, gen: int, population: list[Individual], secs: float) -> None:
        best = population[0]
        fitnesses = [i.fitness for i in population]
        entry = {
            "generation": gen,
            "seconds": secs,
            "best_fitness": best.fitness,
            "mean_fitness": sum(fitnesses) / len(fitnesses),
            "best": best.as_dict(),
        }
        self.history.append(entry)
        m = best.result.metrics if best.result else None
        print(
            f"[gen {gen:3d}] best={best.fitness:.4f} "
            f"mean={entry['mean_fitness']:.4f} "
            + (
                f"val_acc={m.final_val_acc:.1%} gap={m.acc_area:.3f} "
                f"gen_at={m.gen_frac:.2f} grok={m.grok_magnitude():.2f} "
                f"{'early' if m.generalised and m.gen_frac < 0.5 else 'late/none'} "
                if m
                else ""
            )
            + f"({secs:.1f}s) | {best.genome.summary()}"
        )
        if self.logfile:
            with open(self.logfile, "a") as f:
                f.write(json.dumps(entry) + "\n")
