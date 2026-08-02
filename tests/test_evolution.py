"""The evolutionary search loop."""

import os
import tempfile

import torch

from evogrokking.datasets import Dataset, DatasetSpec
from evogrokking.evolution import Evolution, EvolutionConfig


def _fake_dataset(n=40, dim=16, classes=3):
    spec = DatasetSpec("t", "image", input_dim=dim, num_classes=classes)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, dim, generator=g)
    y = torch.randint(0, classes, (n,), generator=g)
    return Dataset(spec, x, y, x.clone(), y.clone())


def test_memory_budget_skips_oversized_genomes():
    ds = _fake_dataset(n=30)
    cfg = EvolutionConfig(
        population_size=4, generations=1, n_hidden=4,
        mem_budget_mb=1e-9, epochs_per_eval=5, workers=1,  # absurdly low budget
    )
    evo = Evolution(ds, cfg, device=torch.device("cpu"))
    best = evo.run()
    # Every genome is over the tiny budget, so all are skipped (never trained).
    assert best.fitness == Evolution._OVER_BUDGET_FITNESS
    assert best.result is None


def test_evolution_runs_end_to_end():
    # A whole (tiny) search: neat-python reproduces, we train and score.
    ds = _fake_dataset()
    out = tempfile.mkdtemp()
    cfg = EvolutionConfig(
        population_size=6, generations=3, epochs_per_eval=15, eval_every=5,
        n_hidden=8, workers=1, seed=0, mem_budget_mb=None,
    )
    evo = Evolution(
        ds, cfg, device=torch.device("cpu"),
        config_path=os.path.join(out, "neat_config.ini"),
    )
    best = evo.run()

    assert len(evo.history) == 3
    assert best.result is not None and best.fitness > float("-inf")
    # The generated neat-python config is kept alongside the run.
    assert os.path.exists(os.path.join(out, "neat_config.ini"))
    # Fitness is the minimise-grokking score of the best individual's run.
    assert abs(best.fitness - best.result.metrics.score()) < 1e-9
    assert best.genome.n_inputs == 16 and best.genome.n_outputs == 3


def test_history_is_recorded_per_generation():
    ds = _fake_dataset()
    out = tempfile.mkdtemp()
    logfile = os.path.join(out, "history.jsonl")
    cfg = EvolutionConfig(
        population_size=4, generations=2, epochs_per_eval=10, eval_every=5,
        n_hidden=4, workers=1, seed=0, mem_budget_mb=None,
    )
    evo = Evolution(ds, cfg, device=torch.device("cpu"), logfile=logfile)
    evo.run()

    assert [h["generation"] for h in evo.history] == [0, 1]
    for entry in evo.history:
        assert "best_fitness" in entry and "mean_fitness" in entry
        assert entry["best_fitness"] >= entry["mean_fitness"]
    # The JSONL log has one line per generation.
    with open(logfile) as f:
        assert sum(1 for _ in f) == 2


def test_no_config_path_does_not_litter_the_cwd():
    # With no run directory to write into, the generated neat config must land in
    # a scratch dir rather than the working directory.
    ds = _fake_dataset(n=20)
    cfg = EvolutionConfig(
        population_size=4, generations=1, epochs_per_eval=5,
        n_hidden=4, workers=1, mem_budget_mb=None,
    )
    before = os.path.exists("neat_config.ini")
    Evolution(ds, cfg, device=torch.device("cpu")).run()
    assert os.path.exists("neat_config.ini") == before
