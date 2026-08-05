"""Tests for the curve smoothing in scripts/plot_curves.py.

Only ``ema_smooth`` and ``should_smooth`` are exercised — they are the pieces of the
plotting script with behaviour worth pinning down. Everything else there is matplotlib
calls and I/O.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from plot_curves import ema_smooth, should_smooth  # noqa: E402

WEIGHT = 0.9


def test_zero_weight_is_identity():
    values = [3.0, 1.0, 4.0, 1.0, 5.0]
    assert ema_smooth(values, 0) == values


def test_constant_series_stays_constant():
    """What the bias correction buys.

    A plain EMA starts from 0 and ramps, which on a loss curve reads as a drop that
    never happened. Debiased, a flat series smooths to that same flat value from the
    very first point.
    """
    smoothed = ema_smooth([2.5] * 20, WEIGHT)
    assert all(v == pytest.approx(2.5) for v in smoothed)


def test_first_point_is_the_first_value():
    smoothed = ema_smooth([7.0, 1.0, 1.0], WEIGHT)
    assert smoothed[0] == pytest.approx(7.0)


def test_smoothing_reduces_step_to_step_variation():
    """The whole point: less batch-to-batch jitter, same overall trend."""
    noisy = [10 - 0.1 * i + (1.5 if i % 2 else -1.5) for i in range(100)]
    smoothed = ema_smooth(noisy, WEIGHT)

    def jitter(xs):
        return sum(abs(b - a) for a, b in zip(xs, xs[1:]))

    assert jitter(smoothed) < jitter(noisy) / 10
    # trend preserved: still descending overall
    assert smoothed[-1] < smoothed[0]


def test_length_is_preserved():
    assert len(ema_smooth([1.0] * 37, WEIGHT)) == 37


def test_nan_passes_through_without_poisoning_the_tail():
    """A NaN spike stays visible at its own step instead of eating the rest."""
    values = [1.0, 1.0, float("nan"), 1.0, 1.0]
    smoothed = ema_smooth(values, WEIGHT)
    assert math.isnan(smoothed[2])
    assert all(math.isfinite(v) for i, v in enumerate(smoothed) if i != 2)
    assert smoothed[-1] == pytest.approx(1.0)


# --- should_smooth: which curves get a trend line ------------------------------


@pytest.mark.parametrize("tag", ["val_loss", "val_acc_mz_bin", "online_val_loss"])
@pytest.mark.parametrize("n_points", [50, 500, 100_000])
def test_val_is_never_smoothed_at_any_length(tag, n_points):
    """Val is per-epoch by construction, so length must not buy it a trend line.

    A flat 50-point threshold used to let a 50-epoch ``val_loss`` through, and the EMA
    lag then distorted it by ~30% of its dynamic range.
    """
    assert not should_smooth(tag, "loss", n_points, WEIGHT, min_points=50)


def test_train_of_the_same_length_is_smoothed():
    """The exclusion is about val specifically, not about short series."""
    assert should_smooth("train_loss", "loss", 50, WEIGHT, min_points=50)


def test_short_series_still_gated_by_min_points():
    assert not should_smooth("train_loss", "loss", 49, WEIGHT, min_points=50)


def test_smooth_zero_disables_everything():
    assert not should_smooth("train_loss", "loss", 5000, 0, min_points=50)


def test_lr_panel_is_never_smoothed():
    """The lr panel is deterministic — smoothing would misrepresent the schedule."""
    assert not should_smooth("lr", "lr", 5000, WEIGHT, min_points=50)
