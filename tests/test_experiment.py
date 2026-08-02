"""The CLI endpoint: argument wiring and the artefacts a run writes."""

import argparse
import os
import tempfile

from evogrokking import datasets, experiment


def _args(**kw):
    defaults = dict(
        dataset="mnist", seed=0, p=97, train_frac=None, train_size=1000, val_size=2000,
        n_subclasses=4, shifted_per_class=1, shift_frac=0.05, feature_epochs=2,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


# --------------------------------------------------------------------------
# Dataset argument wiring
# --------------------------------------------------------------------------
def test_train_frac_applies_to_both_task_families():
    # modadd keeps its historical default; images take an absolute size unless
    # --train-frac is given, in which case it overrides.
    modadd = experiment._dataset_load_kwargs(_args(dataset="modadd"))
    assert modadd["train_frac"] == 0.4 and modadd["p"] == 97

    img = experiment._dataset_load_kwargs(_args(dataset="mnist"))
    assert img["train_size"] == 1000 and "train_frac" not in img

    img_frac = experiment._dataset_load_kwargs(_args(dataset="mnist", train_frac=0.01))
    assert img_frac["train_frac"] == 0.01


def test_shift_knobs_only_reach_shifted_datasets():
    shifted = experiment._dataset_load_kwargs(_args(dataset="mnist"))
    assert shifted["shift_frac"] == 0.05 and shifted["n_subclasses"] == 4

    plain = experiment._dataset_load_kwargs(_args(dataset="mnist_plain"))
    assert "shift_frac" not in plain and "n_subclasses" not in plain

    modadd = experiment._dataset_load_kwargs(_args(dataset="modadd"))
    assert "shift_frac" not in modadd


def test_hyperparams_come_from_task_defaults_plus_overrides():
    ds = datasets.modular_addition(p=7, train_frac=0.5)
    hp_args = _args(lr=None, weight_decay=None, dropout=None, optimizer=None,
                    init_scale=None)
    hp = experiment._hyperparams(hp_args, ds)
    assert hp.weight_decay == 1.0  # the modular task's own default

    overridden = experiment._hyperparams(
        _args(lr=5e-4, weight_decay=None, dropout=None, optimizer=None,
              init_scale=None),
        ds,
    )
    assert overridden.lr == 5e-4 and overridden.weight_decay == 1.0


def test_baseline_genome_is_the_dense_starting_network():
    ds = datasets.modular_addition(p=7, train_frac=0.5)
    g = experiment._baseline_genome(ds, n_hidden=5)
    assert g.n_inputs == ds.spec.input_dim
    assert g.n_outputs == ds.spec.num_classes
    assert len(g.hidden_ids()) == 5
    # Fully wired, including the direct input -> output edges.
    assert g.n_enabled() == (
        ds.spec.input_dim * 5 + 5 * ds.spec.num_classes
        + ds.spec.input_dim * ds.spec.num_classes
    )


# --------------------------------------------------------------------------
# End-to-end artefacts
# --------------------------------------------------------------------------
def test_train_command_writes_curves_and_structure_plots():
    # `train --plot` must produce both visualisations, not just the curves --
    # the same pair `retrain` writes for an evolved genome.
    out_root = tempfile.mkdtemp()
    original_root = experiment.RUNS_ROOT
    experiment.RUNS_ROOT = out_root
    try:
        experiment.main(
            [
                "train", "--dataset", "modadd", "--p", "7", "--hidden", "8",
                "--epochs", "6", "--eval-every", "2", "--plot", "--name", "viz",
            ]
        )
    finally:
        experiment.RUNS_ROOT = original_root

    run = os.path.join(out_root, "viz")
    for name in ("curves.png", "structure.png", "curves.json", "result.json"):
        path = os.path.join(run, name)
        assert os.path.exists(path) and os.path.getsize(path) > 0, name


def test_train_command_records_genome_and_recipe():
    import json

    out_root = tempfile.mkdtemp()
    original_root = experiment.RUNS_ROOT
    experiment.RUNS_ROOT = out_root
    try:
        experiment.main(
            ["train", "--dataset", "modadd", "--p", "7", "--hidden", "4",
             "--epochs", "4", "--eval-every", "2", "--name", "rec"]
        )
    finally:
        experiment.RUNS_ROOT = original_root

    with open(os.path.join(out_root, "rec", "result.json")) as f:
        saved = json.load(f)
    assert set(saved) == {"genome", "hyperparams", "result"}
    assert saved["genome"]["n_inputs"] == 14  # 2p, one-hot
    assert "lr" in saved["hyperparams"] and "embed_dim" not in saved["hyperparams"]
