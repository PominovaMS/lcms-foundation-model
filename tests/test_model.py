import pytest
import torch

from source.model import MS1Encoder


@pytest.fixture
def batch():
    """A mass spectrum."""
    spectra = torch.tensor(
        [
            [[100.1, 0.1], [200.2, 0.2], [300.3, 0.3]],
            [[400.4, 0.4], [500, 0.5], [0, 0]],
        ]
    )

    batch_dict = {
        "mz_array": spectra[:, :, 0],
        "intensity_array": spectra[:, :, 1],
    }

    return batch_dict


def test_peaks_mask(batch):
    """Test random peaks mask is created."""
    model = MS1Encoder(d_model=8, nhead=1, dim_feedforward=12, n_layers=1)
    mask = model.get_peaks_mask(batch["intensity_array"])
    assert mask.shape == (2, 3)


def test_forward(batch):
    """Test peaks encoding."""
    model = MS1Encoder(d_model=8, nhead=1, dim_feedforward=12, n_layers=1)
    peak_embs = model.forward(
        mzs=batch["mz_array"], intensities=batch["intensity_array"]
    )
    assert peak_embs.shape == (2, 3, model.d_model)


def test_training_step(batch):
    """Test traning step is performed (basic)."""
    model = MS1Encoder(d_model=8, nhead=1, dim_feedforward=12, n_layers=1)
    loss = model.training_step(batch, 0)
    assert torch.isreal(loss)


TOTAL_STEPS = 100
WARMUP = 10
PEAK_LR = 1e-3


@pytest.fixture
def lr_trace():
    """The LR the one-cycle schedule produces at each of its ``TOTAL_STEPS`` steps."""
    model = MS1Encoder(
        d_model=8,
        nhead=1,
        dim_feedforward=12,
        n_layers=1,
        lr=PEAK_LR,
        warmup_iters=WARMUP,
        total_steps=TOTAL_STEPS,
    )
    cfg = model.configure_optimizers()
    scheduler = cfg["lr_scheduler"]["scheduler"]
    lrs = []
    for _ in range(TOTAL_STEPS):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    return lrs


def test_one_cycle_starts_below_peak(lr_trace):
    """LR starts at lr/div_factor, not at the peak and not at zero."""
    assert lr_trace[0] == pytest.approx(PEAK_LR / 25.0, rel=1e-3)


def test_one_cycle_peaks_at_warmup(lr_trace):
    """LR reaches ~lr at warmup_iters, then never exceeds it."""
    peak_step = lr_trace.index(max(lr_trace))
    assert peak_step == pytest.approx(WARMUP, abs=1)
    assert max(lr_trace) == pytest.approx(PEAK_LR, rel=1e-3)


def test_one_cycle_anneals_monotonically_after_peak(lr_trace):
    """No re-rise after the peak — the failure mode of the old cosine half-period."""
    tail = lr_trace[lr_trace.index(max(lr_trace)) :]
    assert all(b <= a for a, b in zip(tail, tail[1:]))
    assert tail[-1] < PEAK_LR / 1000


def test_one_cycle_keeps_adam_betas():
    """cycle_momentum must stay off: Adam's betas[0] is pinned at 0.9, not cycled."""
    model = MS1Encoder(
        d_model=8,
        nhead=1,
        dim_feedforward=12,
        n_layers=1,
        warmup_iters=WARMUP,
        total_steps=TOTAL_STEPS,
    )
    cfg = model.configure_optimizers()
    optimizer, scheduler = cfg["optimizer"], cfg["lr_scheduler"]["scheduler"]
    for _ in range(TOTAL_STEPS // 2):
        scheduler.step()
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.98)


class _FakeTrainer:
    def __init__(self, estimated):
        self.estimated_stepping_batches = estimated


def _with_trainer(model, estimated):
    """Attach a stand-in trainer; configure_optimizers reads only this one property."""
    model._trainer = _FakeTrainer(estimated)
    return model


def _tiny(**kw):
    return MS1Encoder(d_model=8, nhead=1, dim_feedforward=12, n_layers=1, **kw)


def test_total_steps_prefers_the_trainer_estimate():
    """Lightning knows the real dataloader length; a formula in train.py does not.

    Getting this wrong made OneCycleLR raise "Tried to step N times" mid-run.
    """
    model = _with_trainer(_tiny(warmup_iters=10, total_steps=999), estimated=250)
    scheduler = model.configure_optimizers()["lr_scheduler"]["scheduler"]
    assert scheduler.total_steps == 250


def test_total_steps_falls_back_without_a_trainer():
    model = _tiny(warmup_iters=10, total_steps=250)
    scheduler = model.configure_optimizers()["lr_scheduler"]["scheduler"]
    assert scheduler.total_steps == 250


@pytest.mark.parametrize("estimated", [float("inf"), -1, 0])
def test_total_steps_falls_back_when_the_estimate_is_unusable(estimated):
    """Unsized IterableDataset gives inf, or max_steps (-1) when that isn't set."""
    model = _with_trainer(_tiny(warmup_iters=10, total_steps=250), estimated)
    scheduler = model.configure_optimizers()["lr_scheduler"]["scheduler"]
    assert scheduler.total_steps == 250


def test_explicit_config_override_beats_the_trainer_estimate():
    """`optimizer.total_steps` in the config is a documented escape hatch."""
    model = _with_trainer(
        _tiny(warmup_iters=10, total_steps=777, auto_total_steps=False), estimated=250
    )
    scheduler = model.configure_optimizers()["lr_scheduler"]["scheduler"]
    assert scheduler.total_steps == 777


def test_legacy_schedule_hparam_still_loads():
    """Checkpoints predating OneCycleLR saved the schedule length under the old name.

    ``load_from_checkpoint`` replays saved hparams as kwargs, so the alias is what
    keeps ``eval/probe_checkpoint.py`` able to read them.
    """
    model = MS1Encoder(
        d_model=8, nhead=1, dim_feedforward=12, n_layers=1,
        cosine_schedule_period_iters=500,
    )
    assert model.hparams.total_steps == 500
    assert "cosine_schedule_period_iters" not in model.hparams
