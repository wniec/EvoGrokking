"""The **fixed** training recipe.

Earlier versions of this project evolved the learning rate, weight decay,
dropout, optimizer, init scale and embedding width alongside the network graph.
That made it impossible to tell whether a good individual owed its behaviour to
its *architecture* or merely to a lucky regularisation setting -- so those knobs
are no longer genes.  They are set once per run (from the CLI) and shared by
every individual in the population, which is what makes an architecture search a
fair comparison between architectures.

Defaults follow the training setup of Carvalho et al., *"Grokking Explained: A
Statistical Phenomenon"* (2025) §5.1: ``lr = 1e-3``, ``weight_decay = 1e-4`` and
an initialisation scaled by 8 (as in Liu et al., "Omnigrok").  The modular task
needs a rather different recipe to move at all, so :meth:`Hyperparams.for_task`
supplies per-task defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

OPTIMIZERS = ("adam", "adamw", "sgd")


@dataclass(frozen=True)
class Hyperparams:
    """Training knobs shared by every individual in a run."""

    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.0
    optimizer: str = "adamw"
    init_scale: float = 8.0
    embed_dim: int = 128  # modular tasks only: token-embedding width

    def __post_init__(self) -> None:
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(
                f"unknown optimizer {self.optimizer!r}; expected one of {OPTIMIZERS}"
            )

    @staticmethod
    def for_task(task: str) -> "Hyperparams":
        """Sensible per-task defaults.

        ``image`` follows the paper's MNIST setup; ``modular`` uses the strong
        weight decay + large learning rate that makes ``(a + b) mod p`` grok at
        all (Power et al., 2022), which is the regime this project starts from.
        """
        if task == "modular":
            return Hyperparams(
                lr=1e-2, weight_decay=1.0, init_scale=1.0, embed_dim=128
            )
        return Hyperparams()

    def with_overrides(self, **kwargs) -> "Hyperparams":
        """Return a copy with the non-``None`` keyword arguments applied.

        Lets the CLI layer pass every flag through unconditionally: flags the
        user did not set arrive as ``None`` and leave the default in place.
        """
        given = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **given) if given else self

    def summary(self) -> str:
        return (
            f"lr={self.lr:.2g} wd={self.weight_decay:.2g} do={self.dropout:.2f} "
            f"opt={self.optimizer} init={self.init_scale:.2g} emb={self.embed_dim}"
        )

    def as_dict(self) -> dict:
        return {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "dropout": self.dropout,
            "optimizer": self.optimizer,
            "init_scale": self.init_scale,
            "embed_dim": self.embed_dim,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Hyperparams":
        fields = {f: d[f] for f in cls.__dataclass_fields__ if f in d}
        return cls(**fields)
