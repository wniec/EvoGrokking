"""Evolutionary architecture search for grokking.

A NEAT genetic algorithm searches the space of network *graphs* and
regularisation strengths (:class:`~evogrokking.genome.Genome`) for the recipe
that maximises grokking.  Each individual's fitness is obtained by actually
training it (:func:`~evogrokking.train.train_and_evaluate`) and scoring the run
with :meth:`~evogrokking.metrics.GrokkingMetrics.score`, whose *primary* term is
the grokking area and whose low-weight *supporting* term is final test accuracy.

The loop is a standard generational GA with elitism and tournament selection.
Because every fitness evaluation is an independent training run, the population
is evaluated **in parallel** across worker processes when ``workers > 1`` (a
``ProcessPoolExecutor`` with the ``spawn`` start method, so it is CUDA-safe).
Every genome is trained under the same fixed ``config.seed``, so a run is fully
reproducible: results do not depend on how the work is scheduled, and the best
individual can be retrained identically afterwards from that same seed.
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from multiprocessing import get_context
from typing import Callable

import torch

from evogrokking.datasets import Dataset
from evogrokking.genome import Genome, Innovations
from evogrokking.models import estimated_activation_mb
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
    elitism: int = 2
    tournament_size: int = 3
    crossover_prob: float = 0.6
    mutation_rate: float = 0.3
    epochs_per_eval: int = 1500  # maximum iterations per fitness evaluation
    eval_every: int = 2
    test_weight: float = 0.1  # low weight on final test acc
    acc_weight: float = 5.0  # weight on the accuracy-curve area in the magnitude
    gen_threshold: float = 0.9  # val-acc a run must reach to count as grokking
    seed: int = 0
    workers: int = 1  # >1 evaluates the population across that many processes
    allow_conv: bool = False  # image tasks: evolve convolutional feature maps
    conv_pool: int = 1  # conv mode: input downsample factor (memory knob)
    mem_budget_mb: float | None = None  # skip genomes whose est. activation memory exceeds this
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
        test_weight=ec["test_weight"],
        gen_threshold=ec["gen_threshold"],
        acc_weight=ec["acc_weight"],
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
        epochs=ec["epochs"],
        eval_every=ec["eval_every"],
        device=device,
        early_stopping=early_stopping,
        seed=ec["seed"],
        gen_threshold=ec["gen_threshold"],
    )


# --------------------------------------------------------------------------
class Evolution:
    def __init__(
        self,
        dataset: Dataset,
        config: EvolutionConfig | None = None,
        device: torch.device | None = None,
        on_generation: Callable[[int, list[Individual]], None] | None = None,
        logfile: str | None = None,
    ):
        self.dataset = dataset
        self.config = config or EvolutionConfig()
        self.device = device or default_device()
        self.rng = random.Random(self.config.seed)
        self.innov = Innovations()
        self.on_generation = on_generation
        self.logfile = logfile
        self._next_id = 0
        self.history: list[dict] = []
        self.best: Individual | None = None

    # -- helpers -----------------------------------------------------------
    def _tag(self, genome: Genome) -> Genome:
        tagged = replace(genome, id=self._next_id)
        self._next_id += 1
        return tagged

    def _eval_params(self) -> dict:
        cfg = self.config
        return {
            "epochs": cfg.epochs_per_eval,
            "eval_every": cfg.eval_every,
            "test_weight": cfg.test_weight,
            "acc_weight": cfg.acc_weight,
            "gen_threshold": cfg.gen_threshold,
            "seed": cfg.seed,
            "es_patience": cfg.early_stop_patience,
            "es_min_delta": cfg.early_stop_min_delta,
            "es_target": cfg.early_stop_target_val_acc,
        }

    def _tournament(self, population: list[Individual]) -> Individual:
        contenders = self.rng.sample(
            population, min(self.config.tournament_size, len(population))
        )
        return max(contenders, key=lambda ind: ind.fitness)

    # Genomes over the memory budget are not trained; they get a fitness below
    # any real score so the search abandons them (rather than risking an OOM).
    _OVER_BUDGET_FITNESS = -1.0

    def _filter_budget(self, pending: list[Individual]) -> list[Individual]:
        """Skip genomes whose estimated activation memory exceeds the budget."""
        budget = self.config.mem_budget_mb
        if budget is None:
            return pending
        batch = len(self.dataset.x_train)
        keep, skipped = [], 0
        for ind in pending:
            mb = estimated_activation_mb(ind.genome, self.dataset.spec, batch)
            if mb > budget:
                ind.fitness = self._OVER_BUDGET_FITNESS
                ind.result = None
                skipped += 1
            else:
                keep.append(ind)
        if skipped:
            print(f"  (skipped {skipped} genome(s) over {budget:.0f} MB memory budget)")
        return keep

    # -- evaluation (sequential or parallel) -------------------------------
    def _evaluate(self, pending: list[Individual], pool: ProcessPoolExecutor | None) -> None:
        pending = self._filter_budget(pending)
        if not pending:
            return
        ec = self._eval_params()
        if pool is None:
            for ind in pending:
                ind.result = _train_one(ind.genome, self.dataset, self.device, ec)
                ind.fitness = _score(ind.result, ec)
            return

        by_id = {ind.genome.id: ind for ind in pending}
        tasks = [(ind.genome, ec) for ind in pending]
        for gid, result, fitness in pool.map(_eval_worker, tasks):
            ind = by_id[gid]
            ind.result = result
            ind.fitness = fitness

    # -- main loop ---------------------------------------------------------
    def run(self) -> Individual:
        cfg = self.config
        population = [
            Individual(
                self._tag(
                    Genome.random(
                        self.rng,
                        self.innov,
                        allow_conv=cfg.allow_conv,
                        conv_pool=cfg.conv_pool,
                    )
                )
            )
            for _ in range(cfg.population_size)
        ]

        pool = None
        if cfg.workers > 1:
            ctx = get_context("spawn")
            pool = ProcessPoolExecutor(
                max_workers=cfg.workers,
                mp_context=ctx,
                initializer=_init_worker,
                initargs=(self.dataset, str(self.device)),
            )
        try:
            for gen in range(cfg.generations):
                t0 = time.time()
                self._evaluate([i for i in population if i.result is None], pool)

                population.sort(key=lambda i: i.fitness, reverse=True)
                gen_best = population[0]
                if self.best is None or gen_best.fitness > self.best.fitness:
                    self.best = gen_best

                self._record(gen, population, time.time() - t0)
                if self.on_generation:
                    self.on_generation(gen, population)

                if gen == cfg.generations - 1:
                    break
                population = self._next_generation(population)
        finally:
            if pool is not None:
                pool.shutdown()

        assert self.best is not None
        return self.best

    def _next_generation(self, population: list[Individual]) -> list[Individual]:
        cfg = self.config
        # Elitism: carry the best individuals over unchanged (and un-re-evaluated).
        next_pop = [
            Individual(ind.genome, ind.fitness, ind.result)
            for ind in population[: cfg.elitism]
        ]
        while len(next_pop) < cfg.population_size:
            if self.rng.random() < cfg.crossover_prob:
                a = self._tournament(population)
                b = self._tournament(population)
                fitter, other = (a, b) if a.fitness >= b.fitness else (b, a)
                child = Genome.crossover(fitter.genome, other.genome, self.rng)
                child = child.mutate(self.rng, self.innov, cfg.mutation_rate)
            else:
                parent = self._tournament(population).genome
                child = parent.mutate(self.rng, self.innov, cfg.mutation_rate)
            next_pop.append(Individual(self._tag(child)))
        return next_pop

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
                f"grok_area={m.grok_area:.3f} acc_area={m.acc_area:.3f} "
                f"val_acc={m.final_val_acc:.1%} drop={m.val_loss_drop:.1f} "
                f"{'GROK' if m.generalised else 'overfit'} "
                if m
                else ""
            )
            + f"({secs:.1f}s) | {best.genome.summary()}"
        )
        if self.logfile:
            with open(self.logfile, "a") as f:
                f.write(json.dumps(entry) + "\n")
