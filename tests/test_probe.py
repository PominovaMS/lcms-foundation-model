"""Tests for the standalone linear probe (eval/probe.py).

These exercise :func:`fit_probe` directly on synthetic embeddings — no encoder, no
mzML, no GPU — which is the point of it taking plain tensors.
"""

import pytest
import torch

from probe import baseline_metrics, fit_probe, run_retrain_probe
from source.model import MS1Encoder

NUM_CLASSES = 4
D_MODEL = 8


@pytest.fixture
def separable():
    """Embeddings that are trivially linearly separable by class.

    Each class sits on its own one-hot axis, so a working probe should reach a
    high accuracy and a broken one is unambiguous.
    """
    y = torch.arange(NUM_CLASSES).repeat_interleave(5)
    X = torch.zeros(len(y), D_MODEL)
    X[torch.arange(len(y)), y] = 1.0
    return X, y


def test_same_seed_is_reproducible(separable):
    """Two fits with the same seed must agree exactly."""
    X, y = separable
    a = fit_probe(X, y, X, y, NUM_CLASSES, seed=0)
    b = fit_probe(X, y, X, y, NUM_CLASSES, seed=0)
    assert a["val_acc"] == b["val_acc"]
    assert a["val_loss"] == b["val_loss"]


def test_different_seed_changes_the_fit(separable):
    """Different seeds must actually produce different probes.

    Otherwise the reported std would be a constant 0 and would say nothing about
    the noise floor.
    """
    X, y = separable
    a = fit_probe(X, y, X, y, NUM_CLASSES, seed=0, n_epochs=3)
    b = fit_probe(X, y, X, y, NUM_CLASSES, seed=1, n_epochs=3)
    assert a["val_loss"] != b["val_loss"]


def test_learns_separable_data(separable):
    """Sanity check that the probe is capable of solving an easy problem."""
    X, y = separable
    res = fit_probe(X, y, X, y, NUM_CLASSES, seed=0, n_epochs=500)
    assert res["val_acc"] == 1.0
    assert res["n_pred_classes"] == NUM_CLASSES


def test_constant_embeddings_collapse():
    """A degenerate representation must be reported as a collapse.

    With identical embeddings the probe cannot separate anything, so it predicts
    whichever class the bias favours — n_pred_classes == 1 is what makes that
    visible rather than being mistaken for a merely weak encoder.
    """
    y = torch.tensor([0] * 12 + [1, 2, 3])
    X = torch.ones(len(y), D_MODEL)
    res = fit_probe(X, y, X, y, NUM_CLASSES, seed=0, n_epochs=200)
    assert res["n_pred_classes"] == 1
    assert res["val_acc_macro"] <= 1.0 / NUM_CLASSES


def test_probe_epochs_reports_underfitting(separable):
    """probe_epochs pins at n_epochs when min_train_loss is never reached."""
    X, y = separable
    res = fit_probe(X, y, X, y, NUM_CLASSES, seed=0, n_epochs=2, min_train_loss=0.0)
    assert res["probe_epochs"] == 2


def _fake_run_loader(n_runs, n_spectra=2, n_peaks=5):
    """One batch of run-level items, shaped like eval/data.py::run_collate_fn output.

    ``mz_array`` / ``intensity_array`` are lists of per-run (n_spectra, n_peaks)
    tensors, which is what RunDataset + run_collate_fn hand to encode_dataset.
    """
    torch.manual_seed(123)
    return [
        {
            "mz_array": [torch.rand(n_spectra, n_peaks) * 1000 for _ in range(n_runs)],
            "intensity_array": [torch.rand(n_spectra, n_peaks) for _ in range(n_runs)],
            "label": torch.arange(n_runs) % NUM_CLASSES,
        }
    ]


def test_run_retrain_probe_aggregates_repeats():
    """End-to-end: encode once, fit n_repeats probes, report mean + std."""
    model = MS1Encoder(d_model=D_MODEL, nhead=1, dim_feedforward=12, n_layers=1)
    train_loader, val_loader = _fake_run_loader(8), _fake_run_loader(8)

    res = run_retrain_probe(
        model,
        train_loader,
        val_loader,
        d_model=D_MODEL,
        num_classes=NUM_CLASSES,
        n_epochs=5,
        device="cpu",
        seed=0,
        n_repeats=3,
    )

    # every metric carries a matching _std, plus the seed-independent baselines
    for key in ("val_acc", "val_loss", "val_acc_macro", "n_pred_classes"):
        assert key in res and f"{key}_std" in res
    assert res["majority_acc"] == pytest.approx(0.25)
    assert res["random_acc"] == pytest.approx(1 / NUM_CLASSES)


def test_run_retrain_probe_is_reproducible():
    """Same seed, same numbers — the whole point of the change."""
    model = MS1Encoder(d_model=D_MODEL, nhead=1, dim_feedforward=12, n_layers=1)
    loaders = (_fake_run_loader(8), _fake_run_loader(8))
    kwargs = {
        "d_model": D_MODEL,
        "num_classes": NUM_CLASSES,
        "n_epochs": 5,
        "device": "cpu",
        "n_repeats": 2,
    }
    a = run_retrain_probe(model, *loaders, seed=0, **kwargs)
    b = run_retrain_probe(model, *loaders, seed=0, **kwargs)
    assert a == b


def test_majority_baseline_uses_train_modal_class():
    """majority_acc scores the *train* modal class against val.

    This is the abele failure mode in miniature: class 0 dominates train but is
    rare in val, so always predicting it scores below the random rate.
    """
    y_train = torch.tensor([0] * 10 + [1, 2, 3])
    y_val = torch.tensor([0, 1, 1, 2, 2, 3, 3, 3])
    base = baseline_metrics(y_train, y_val, NUM_CLASSES)
    assert base["majority_acc"] == pytest.approx(1 / 8)
    assert base["random_acc"] == pytest.approx(1 / 4)
    assert base["majority_acc"] < base["random_acc"]
