"""EvoGrokking entry point.

Thin wrapper around the experiment CLI so the project can be run either as
``python main.py ...`` or ``python -m evogrokking.experiment ...``.

    python main.py train  --dataset modadd
    python main.py evolve --dataset modadd --generations 12 --population 24
"""

from evogrokking.experiment import main

if __name__ == "__main__":
    main()
