"""EvoGrokking: evolving neural architectures that generalise *without* grokking.

The package is organised into a few small, composable parts:

* :mod:`evogrokking.datasets`    -- MNIST / FashionMNIST with a shifted training
  class distribution (after Carvalho et al., 2025), plus p-modular addition.
* :mod:`evogrokking.subclasses`  -- the latent-subclass clustering and Equation-1
  under-sampling that create that distribution shift.
* :mod:`evogrokking.metrics`     -- grokking measured from the train/validation
  curves, and the objective that **minimises** it at high final accuracy.
* :mod:`evogrokking.genome`      -- the evolved architecture: a classical NEAT
  DAG of individual neurons (plain MLPs; no convolutions or embeddings).
* :mod:`evogrokking.hyperparams` -- the training recipe, fixed per run and *not*
  evolved.
* :mod:`evogrokking.neat_arch`   -- neat-python integration (reproduction,
  crossover, speciation).
* :mod:`evogrokking.models`      -- compiles a genome into a trainable
  ``nn.Module`` of masked dense matmuls.
* :mod:`evogrokking.train`       -- trains a network with Adam/SGD (CUDA if
  available) and returns its loss curves + grokking metrics.
* :mod:`evogrokking.evolution`   -- the evolutionary architecture search.
* :mod:`evogrokking.experiment`  -- the experiment-running endpoint (CLI).
"""

from evogrokking.genome import Genome
from evogrokking.hyperparams import Hyperparams
from evogrokking.metrics import GrokkingMetrics, grokking_metrics

__all__ = ["Genome", "Hyperparams", "GrokkingMetrics", "grokking_metrics"]
