"""Tests for the curve smoothing in scripts/plot_curves.py.

Only ``ema_smooth`` is exercised — it is the one piece of the plotting script with
behaviour worth pinning down. Everything else there is matplotlib calls and I/O.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from plot_curves import ema_smooth  # noqa: E402

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
