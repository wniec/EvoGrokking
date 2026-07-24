"""neat-python integration: reproduction is the library's job, not ours.

This project used to carry a hand-written NEAT implementation (mutation,
innovation-aligned crossover, the lot).  That is now delegated to
`neat-python <https://neat-python.readthedocs.io>`_, which brings the parts the
hand-rolled version never had -- **speciation**, stagnation handling and
fitness sharing -- and is a maintained reference implementation of Stanley &
Miikkulainen (2002).

Two things have to be bridged.

**Nodes carry a width.**  A neat-python node is a single unit with a bias,
response and activation; ours is a whole *layer* (or, in conv mode, a feature map
with a kernel).  :class:`LayerNodeGene` extends the stock node gene with two
extra evolvable attributes -- ``width`` and ``kernel_idx`` -- using neat-python's
documented gene-attribute extension point, so the library mutates and crosses
them over exactly like any built-in attribute.

**Weights are not evolved.**  Connection *weights* come from gradient descent, so
the stock ``weight`` attribute is left in place (``mutate_add_node`` reads it)
but pinned: zero variance, zero mutation rate.  It never varies, so it also never
contributes to the compatibility distance.  Only ``enabled`` matters to us.

The population's genomes are translated into this project's
:class:`~evogrokking.genome.Genome` phenotype by :func:`to_genome` before they are
built and trained.
"""

from __future__ import annotations

import os

from neat.attributes import IntegerAttribute
from neat.genes import DefaultConnectionGene, DefaultNodeGene
from neat.genome import DefaultGenome, DefaultGenomeConfig

from evogrokking.genome import (
    ACTIVATIONS,
    CONV_CH,
    INPUT,
    KERNEL_SIZES,
    MAX_WIDTH,
    MIN_WIDTH,
    OUTPUT,
    ConnGene,
    Genome,
    NodeGene,
)

# neat-python's key convention: inputs are negative, outputs are 0..num_outputs-1
# and hidden nodes count up from num_outputs.  We use a single input "pin" (the
# whole feature vector) and a single output "pin" (the logit vector).
NEAT_INPUT_KEY = -1
NEAT_OUTPUT_KEY = 0


class LayerNodeGene(DefaultNodeGene):
    """A node gene that also carries the layer's ``width`` and kernel size.

    ``width`` is the number of units (linear mode) or channels (conv mode);
    ``kernel_idx`` indexes :data:`~evogrokking.genome.KERNEL_SIZES` so that
    mutation can only ever land on a valid *odd* kernel -- an integer attribute
    over the sizes themselves could drift onto an even value, which would break
    the same-padding shape invariant that lets arbitrary skip connections line
    up.
    """

    _gene_attributes = DefaultNodeGene._gene_attributes + [
        IntegerAttribute("width"),
        IntegerAttribute(
            "kernel_idx",
            # Defaults so a linear-mode config need not mention the kernel at all.
            min_value=0,
            max_value=len(KERNEL_SIZES) - 1,
            mutate_rate=0.1,
            replace_rate=0.05,
            mutate_power=1.0,
        ),
    ]

    def distance(self, other, config):
        """Compatibility distance, extended with the width/kernel genes.

        Widths are compared on a *relative* scale: 16 vs 32 units is a bigger
        architectural difference than 496 vs 512, and an absolute difference
        would otherwise swamp every other term for wide layers.
        """
        wa, wb = max(1, self.width), max(1, other.width)
        extra = abs(wa - wb) / max(wa, wb)
        if self.kernel_idx != other.kernel_idx:
            extra += 1.0
        # super() already scales its own terms by the weight coefficient; scale
        # ours the same way so all node-gene terms stay commensurable.
        return super().distance(other, config) + (
            extra * config.compatibility_weight_coefficient
        )


class ArchGenome(DefaultGenome):
    """A neat-python genome whose nodes are layers (see :class:`LayerNodeGene`)."""

    @classmethod
    def parse_config(cls, param_dict):
        param_dict["node_gene_type"] = LayerNodeGene
        param_dict["connection_gene_type"] = DefaultConnectionGene
        return DefaultGenomeConfig(param_dict, section_name=cls.__name__)


# --------------------------------------------------------------------------
# Config file generation.  neat-python is configured from an INI file, so we
# write one from the CLI options into the run directory -- which also leaves a
# record of the exact search settings alongside the results.
# --------------------------------------------------------------------------
def write_config(
    path: str,
    *,
    pop_size: int,
    conv: bool,
    elitism: int = 2,
    survival_threshold: float = 0.2,
    compatibility_threshold: float = 3.0,
    max_stagnation: int = 15,
    species_elitism: int = 2,
    conn_add_prob: float = 0.5,
    conn_delete_prob: float = 0.2,
    node_add_prob: float = 0.3,
    node_delete_prob: float = 0.15,
    width_mutate_rate: float = 0.2,
    activation_mutate_rate: float = 0.1,
) -> str:
    """Write a neat-python config file and return its path.

    Width bounds depend on the mode: conv nodes are *channels* (kept small,
    because a same-padding map costs ``batch x channels x H x W``), linear nodes
    are units.
    """
    w_min, w_max = CONV_CH if conv else (MIN_WIDTH, MAX_WIDTH)
    # Mutation step: a fraction of the range, so widths drift rather than jump.
    w_power = max(1.0, (w_max - w_min) / 10.0)
    activations = " ".join(ACTIVATIONS)

    text = f"""\
# Generated by evogrokking -- see evogrokking/neat_arch.py.
# Only the *architecture* is evolved here; the training recipe (lr, weight decay,
# dropout, optimizer, init scale) is fixed for the whole run, see
# evogrokking/hyperparams.py.
[NEAT]
fitness_criterion     = max
fitness_threshold     = 1e9
no_fitness_termination = True
pop_size              = {pop_size}
reset_on_extinction   = True

[ArchGenome]
# One input pin (the whole feature vector) and one output pin (the logits).
num_inputs            = 1
num_outputs           = 1
num_hidden            = 0
feed_forward          = True
initial_connection    = full_direct

compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

conn_add_prob         = {conn_add_prob}
conn_delete_prob      = {conn_delete_prob}
node_add_prob         = {node_add_prob}
node_delete_prob      = {node_delete_prob}
single_structural_mutation = False
structural_mutation_surer  = default

# --- layer width (units, or channels in conv mode) ---
width_init_mean       = {(w_min + w_max) // 2}
width_init_stdev      = 0
width_min_value       = {w_min}
width_max_value       = {w_max}
width_mutate_rate     = {width_mutate_rate}
width_mutate_power    = {w_power}
width_replace_rate    = 0.1

# --- conv kernel: an index into KERNEL_SIZES, so only odd sizes are reachable ---
kernel_idx_min_value   = 0
kernel_idx_max_value   = {len(KERNEL_SIZES) - 1}
kernel_idx_mutate_rate = {0.15 if conv else 0.0}
kernel_idx_mutate_power = 1.0
kernel_idx_replace_rate = {0.05 if conv else 0.0}

# --- activation ---
activation_default      = random
activation_options      = {activations}
activation_mutate_rate  = {activation_mutate_rate}

# --- aggregation: our nodes always sum their inputs ---
aggregation_default     = sum
aggregation_options     = sum
aggregation_mutate_rate = 0.0

# --- bias / response: unused. Our layers own their biases and train them by
# --- gradient descent, so these genes are pinned to a constant.
bias_init_mean        = 0.0
bias_init_stdev       = 0.0
bias_max_value        = 0.0
bias_min_value        = 0.0
bias_mutate_power     = 0.0
bias_mutate_rate      = 0.0
bias_replace_rate     = 0.0

response_init_mean    = 1.0
response_init_stdev   = 0.0
response_max_value    = 1.0
response_min_value    = 1.0
response_mutate_power = 0.0
response_mutate_rate  = 0.0
response_replace_rate = 0.0

# --- connection weight: also unused (weights come from gradient descent), but
# --- neat-python's add-node mutation reads it, so it stays pinned at 1.
weight_init_mean      = 1.0
weight_init_stdev     = 0.0
weight_max_value      = 1.0
weight_min_value      = 1.0
weight_mutate_power   = 0.0
weight_mutate_rate    = 0.0
weight_replace_rate   = 0.0

enabled_default       = True
enabled_mutate_rate   = 0.05

[DefaultSpeciesSet]
compatibility_threshold = {compatibility_threshold}

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = {max_stagnation}
species_elitism      = {species_elitism}

[DefaultReproduction]
elitism            = {elitism}
survival_threshold = {survival_threshold}
min_species_size   = 2
"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path


# --------------------------------------------------------------------------
# Translation: neat-python genome -> this project's phenotype description.
# --------------------------------------------------------------------------
def _map_key(key: int) -> int:
    """neat-python node key -> :mod:`evogrokking.genome` node id.

    ``-1`` (the single input pin) becomes ``INPUT``, ``0`` (the single output
    pin) becomes ``OUTPUT``, and hidden keys -- which neat-python numbers from 1
    upwards -- are shifted clear of both.
    """
    if key == NEAT_INPUT_KEY:
        return INPUT
    if key == NEAT_OUTPUT_KEY:
        return OUTPUT
    return key + 1


def to_genome(
    neat_genome, conv: bool = False, conv_pool: int = 1, genome_id: int | None = None
) -> Genome:
    """Convert a ``neat-python`` genome into a :class:`~evogrokking.genome.Genome`.

    Only hidden nodes become :class:`NodeGene`\\s: the input node is the dataset's
    feature vector and the output node is the class logits, so neither has an
    evolvable width or activation.
    """
    nodes = tuple(
        NodeGene(
            id=_map_key(key),
            width=int(gene.width),
            activation=gene.activation,
            kernel_size=KERNEL_SIZES[
                max(0, min(len(KERNEL_SIZES) - 1, int(gene.kernel_idx)))
            ],
        )
        for key, gene in sorted(neat_genome.nodes.items())
        if key != NEAT_OUTPUT_KEY
    )
    conns = tuple(
        ConnGene(
            innovation=int(gene.innovation),
            src=_map_key(src),
            dst=_map_key(dst),
            enabled=bool(gene.enabled),
        )
        for (src, dst), gene in sorted(neat_genome.connections.items())
    )
    gid = neat_genome.key if genome_id is None else genome_id
    return Genome(
        nodes=nodes,
        conns=conns,
        conv=conv,
        conv_pool=conv_pool if conv else 1,
        id=int(gid),
    )
