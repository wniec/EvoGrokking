"""EvoGrokking: evolving neural architectures that maximally induce grokking.

The package is organised into a few small, composable parts:

* :mod:`evogrokking.datasets`  -- MNIST, FashionMNIST and p-modular addition.
* :mod:`evogrokking.metrics`   -- grokking measured as the area between the
  train/validation log-loss curves.
* :mod:`evogrokking.genome`    -- a NEAT-inspired genome encoding architecture
  *and* regularisation strength, with mutation / crossover.
* :mod:`evogrokking.models`    -- builds a trainable ``nn.Module`` from a genome.
* :mod:`evogrokking.train`     -- trains a network with Adam/SGD (CUDA if available)
  and returns its loss curves + grokking metrics.
* :mod:`evogrokking.evolution` -- the evolutionary architecture search.
* :mod:`evogrokking.experiment`-- the experiment-running endpoint (CLI).
"""

from evogrokking.genome import Genome
from evogrokking.metrics import GrokkingMetrics, grokking_metrics

__all__ = ["Genome", "GrokkingMetrics", "grokking_metrics"]
