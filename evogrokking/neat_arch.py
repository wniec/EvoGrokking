"""neat-python integration: reproduction is the library's job, not ours.

This is **classical NEAT** -- a node is one neuron, a connection is one scalar
weight -- so almost nothing needs bridging: the node and connection genes are
neat-python's own, and :func:`to_genome` is close to an identity mapping.

Two adjustments are still required.

**Weights are not evolved.**  Connection weights and neuron biases come from
gradient descent (that is what produces the learning curves the grokking metrics
are measured on), so the stock ``weight``, ``bias`` and ``response`` genes are
pinned to constants: zero variance, zero mutation rate.  They never vary, so they
also never contribute to the compatibility distance.  What evolves is the
topology plus each neuron's ``activation`` gene.

**The founding population must share its structure.**  We deliberately start from
a *big densely connected* network rather than NEAT's minimal seed, and
neat-python's ``configure_new`` hands every initial genome its **own** hidden-node
keys.  With ``num_hidden > 0`` that makes the whole population mutually
non-homologous: every genome lands in its own species and crossover degenerates
into copying, because every gene is disjoint.  :class:`ArchGenome` overrides
``configure_new`` so all founding genomes share one set of hidden keys -- they
represent the same founding structure, and are homologous as they should be.

Note on the compatibility threshold: neat-python normalises genomic distance by
gene count, so distances are *mean per-gene* differences.  With a dense start
(tens of thousands of connection genes) a handful of structural differences moves
the distance by very little, so the useful threshold is far below the ~3.0
typical of minimal-seed NEAT -- see ``--compatibility-threshold``.
"""

from __future__ import annotations

import os
from itertools import count
from pathlib import Path
from random import random

from neat.genes import DefaultConnectionGene, DefaultNodeGene
from neat.genome import DefaultGenome, DefaultGenomeConfig

from evogrokking.genome import ACTIVATIONS, ConnGene, Genome, NodeGene


def initial_hidden_keys(config) -> list[int]:
    """The fixed hidden-node keys every founding genome shares."""
    return [config.num_outputs + i for i in range(config.num_hidden)]


class ArchGenome(DefaultGenome):
    """A neat-python genome whose founding population shares its hidden nodes."""

    @classmethod
    def parse_config(cls, param_dict):
        # Our own knob, not one of neat-python's; pop it before the library sees
        # the dict and hang it on the resulting config for `mutate` to read.
        rounds = int(param_dict.pop("structural_mutation_rounds", 1))
        param_dict["node_gene_type"] = DefaultNodeGene
        param_dict["connection_gene_type"] = DefaultConnectionGene
        config = DefaultGenomeConfig(param_dict, section_name=cls.__name__)
        config.structural_mutation_rounds = max(1, rounds)
        return config

    def mutate(self, config):
        """Apply the NEAT structural operators ``structural_mutation_rounds`` times.

        neat-python performs *at most one* add/delete of each kind per genome per
        generation.  That is well judged for a minimal seed, where a genome has
        tens of genes and one change is a large relative move -- but this project
        starts from a big dense network, where a genome can carry tens of
        thousands of connection genes and a single deletion is a ~0.004 % change.
        The search then cannot restructure anything in a reasonable number of
        generations.  Repeating the operators makes the *step size* scale with the
        genome instead of staying fixed at one gene.

        Attribute mutation (activations) is applied once per genome regardless, so
        raising the round count does not silently multiply the activation mutation
        rate along with the structural one.
        """
        for _ in range(config.structural_mutation_rounds):
            if random() < config.node_add_prob:
                self.mutate_add_node(config)
            if random() < config.node_delete_prob:
                self.mutate_delete_node(config)
            if random() < config.conn_add_prob:
                self.mutate_add_connection(config)
            if random() < config.conn_delete_prob:
                self.mutate_delete_connection()

        for cg in self.connections.values():
            cg.mutate(config)
        for ng in self.nodes.values():
            ng.mutate(config)

    def configure_new(self, config):
        """Create a founding genome.

        Identical to :meth:`DefaultGenome.configure_new` except that the initial
        hidden nodes get **deterministic, shared** keys instead of fresh ones per
        genome, so the founding population is homologous (see module docstring).
        """
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key)

        hidden_keys = initial_hidden_keys(config)
        for node_key in hidden_keys:
            self.nodes[node_key] = self.create_node(config, node_key)

        # Later add-node mutations must not collide with the shared keys.
        if config.node_indexer is None:
            config.node_indexer = count(config.num_outputs + config.num_hidden)

        # Wire it up using neat-python's own connectivity helpers, which read
        # self.nodes and therefore pick up the shared hidden keys.
        if "fs_neat" in config.initial_connection:
            if config.initial_connection == "fs_neat_nohidden":
                self.connect_fs_neat_nohidden(config)
            else:
                self.connect_fs_neat_hidden(config)
        elif "full" in config.initial_connection:
            if config.initial_connection == "full_nodirect":
                self.connect_full_nodirect(config)
            else:
                self.connect_full_direct(config)
        elif "partial" in config.initial_connection:
            if config.initial_connection == "partial_nodirect":
                self.connect_partial_nodirect(config)
            else:
                self.connect_partial_direct(config)


# --------------------------------------------------------------------------
# Config file generation.
#
# neat-python is configured from an INI file.  The template lives beside this
# module in ``neat.config`` rather than inline here, so the settings can be read
# (and diffed) as an ordinary config file; :func:`write_config` fills in its
# placeholders and drops the result into the run directory, which also leaves a
# record of the exact search settings alongside the results.
# --------------------------------------------------------------------------
#: The template rendered by :func:`write_config`.
CONFIG_TEMPLATE = Path(__file__).with_name("neat.config")


def write_config(
    path: str,
    *,
    pop_size: int,
    n_inputs: int,
    n_outputs: int,
    n_hidden: int,
    initial_connection: str = "full_direct",
    initial_connection_fraction: float = 1.0,
    structural_mutation_rounds: int = 1,
    elitism: int = 2,
    survival_threshold: float = 0.2,
    compatibility_threshold: float = 0.2,
    max_stagnation: int = 15,
    species_elitism: int = 2,
    conn_add_prob: float = 0.3,
    conn_delete_prob: float = 0.4,
    node_add_prob: float = 0.15,
    node_delete_prob: float = 0.2,
    activation_mutate_rate: float = 0.05,
    activation_default: str = "relu",
    enabled_mutate_rate: float = 0.02,
) -> str:
    """Render :data:`CONFIG_TEMPLATE` to ``path`` and return that path.

    Deletion probabilities default *above* the addition ones: the search starts
    from a big dense network, so the interesting direction is pruning it down to
    the structure that generalises earliest.

    ``activation_default`` is a concrete function rather than ``random`` so the
    founding population is genuinely *one* structure.  Drawing each neuron's
    activation at random would make the founders differ on ~3/4 of their nodes
    before evolution even starts, which alone pushes the genomic distance past
    any sane threshold and shatters the population into one species per genome.
    Activations then diverge through mutation, as structure does.

    ``initial_connection_fraction`` below 1.0 switches the ``full_*`` wiring to
    the matching ``partial_*`` form, so each founding genome gets its **own**
    random subset of the dense connectivity.  That is the cheapest source of
    founding diversity: with full wiring every founder is byte-identical, so the
    first generation evaluates the same network ``pop_size`` times and selection
    has nothing to choose between.  The subsets stay homologous -- same node keys,
    same innovation numbers for shared edges -- so crossover still aligns.
    """
    if not 0.0 < initial_connection_fraction <= 1.0:
        raise ValueError(
            "initial_connection_fraction must be in (0, 1], got "
            f"{initial_connection_fraction}"
        )
    if initial_connection_fraction < 1.0:
        if not initial_connection.startswith("full"):
            raise ValueError(
                "initial_connection_fraction only applies to the full_* wirings, "
                f"got {initial_connection!r}"
            )
        # neat-python spells partial connectivity "partial_direct <fraction>".
        initial_connection = (
            initial_connection.replace("full", "partial")
            + f" {initial_connection_fraction}"
        )
    # ``##`` lines are notes to the template's author, not config: strip them so
    # they never reach the generated file.
    template = "\n".join(
        line
        for line in CONFIG_TEMPLATE.read_text().splitlines()
        if not line.startswith("##")
    )
    text = template.format(
        pop_size=pop_size,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_hidden=n_hidden,
        initial_connection=initial_connection,
        structural_mutation_rounds=structural_mutation_rounds,
        elitism=elitism,
        survival_threshold=survival_threshold,
        compatibility_threshold=compatibility_threshold,
        max_stagnation=max_stagnation,
        species_elitism=species_elitism,
        conn_add_prob=conn_add_prob,
        conn_delete_prob=conn_delete_prob,
        node_add_prob=node_add_prob,
        node_delete_prob=node_delete_prob,
        activation_mutate_rate=activation_mutate_rate,
        activation_default=activation_default,
        enabled_mutate_rate=enabled_mutate_rate,
        activations=" ".join(ACTIVATIONS),
    ).rstrip("\n") + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path


# --------------------------------------------------------------------------
# Translation: neat-python genome -> this project's phenotype description.
# --------------------------------------------------------------------------
def to_genome(neat_genome, n_inputs: int, n_outputs: int) -> Genome:
    """Convert a ``neat-python`` genome into a :class:`~evogrokking.genome.Genome`.

    Node keys carry over unchanged -- both sides use neat-python's convention of
    negative input keys, outputs ``0..n_outputs-1`` and hidden above that.
    """
    nodes = tuple(
        NodeGene(id=int(key), activation=gene.activation)
        for key, gene in sorted(neat_genome.nodes.items())
    )
    conns = tuple(
        ConnGene(
            innovation=int(gene.innovation),
            src=int(src),
            dst=int(dst),
            enabled=bool(gene.enabled),
        )
        for (src, dst), gene in sorted(neat_genome.connections.items())
    )
    return Genome(
        nodes=nodes,
        conns=conns,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        id=int(neat_genome.key),
    )
