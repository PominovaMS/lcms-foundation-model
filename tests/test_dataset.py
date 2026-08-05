"""Tests for the shuffled training loader (source/dataset.py).

``SpectrumIndexDataset`` + ``batch_collate`` exist so a DataLoader can re-draw
batches each epoch over a depthcharge ``SpectrumDataset`` (an IterableDataset, which
PyTorch will not shuffle). These check the batches it produces are interchangeable
with the sequential path's, and that they actually change between epochs.
"""

import polars as pl
import pytest
import torch
from depthcharge.data import SpectrumDataset
from torch.utils.data import DataLoader

from source.dataset import (
    SpectrumIndexDataset,
    batch_collate,
    iterable_steps_per_epoch,
)

N_FILES = 5
PER_FILE = 40
N_TOTAL = N_FILES * PER_FILE
BATCH = 16


@pytest.fixture
def spectra(tmp_path):
    """A small on-disk dataset with variable peak counts and a peak_file column."""
    rows = []
    for f in range(N_FILES):
        for i in range(PER_FILE):
            n_peaks = 3 + (i % 4)  # variable, so padding width actually varies
            idx = f * PER_FILE + i
            rows.append(
                {
                    "peak_file": f"file{f}.mzML",
                    "scan_id": idx,
                    "mz_array": [300.0 + idx + p for p in range(n_peaks)],
                    "intensity_array": [0.5] * n_peaks,
                }
            )
    ds = SpectrumDataset(pl.DataFrame(rows), batch_size=BATCH, path=str(tmp_path / "d.lance"))
    yield ds
    del ds


def _loader(spectra, shuffle, seed=0):
    torch.manual_seed(seed)
    return DataLoader(
        SpectrumIndexDataset(spectra),
        batch_size=BATCH,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=batch_collate(spectra),
    )


def _scan_ids(batch):
    return set(batch["scan_id"].tolist())


def test_batch_structure_matches_the_sequential_path(spectra):
    """A shuffled batch must be a drop-in for what training already consumes."""
    seq = next(iter(DataLoader(spectra, batch_size=None, num_workers=0)))
    shuf = next(iter(_loader(spectra, shuffle=True)))

    assert set(shuf.keys()) == set(seq.keys())
    for key in ("mz_array", "intensity_array"):
        assert shuf[key].dtype == seq[key].dtype
        assert shuf[key].shape[0] == BATCH
    # Padded to the batch's own max, not a fixed width — a fixed width would push
    # extra padded positions through the encoder on every step.
    assert shuf["mz_array"].shape == shuf["intensity_array"].shape
    real_per_spectrum = (shuf["intensity_array"] != 0).sum(dim=1)
    assert shuf["intensity_array"].shape[1] == real_per_spectrum.max().item()


def test_padding_is_zero_so_the_real_peak_mask_still_works(spectra):
    batch = next(iter(_loader(spectra, shuffle=True)))
    real = batch["intensity_array"] != 0
    # Every spectrum has at least 3 real peaks and padding is exactly 0.
    assert real.sum(dim=1).min().item() >= 3
    assert (batch["intensity_array"][~real] == 0).all()


def test_epochs_have_different_batch_composition(spectra):
    """The point of the change: not merely a different order of the same batches."""
    loader = _loader(spectra, shuffle=True)
    first = _scan_ids(next(iter(loader)))
    second = _scan_ids(next(iter(loader)))
    assert first != second


def test_one_epoch_covers_every_spectrum_exactly_once(spectra):
    loader = _loader(spectra, shuffle=True)
    seen = []
    for batch in loader:
        seen.extend(batch["scan_id"].tolist())
    assert sorted(seen) == list(range(N_TOTAL))


def test_final_short_batch_is_not_dropped(spectra):
    """N_TOTAL is not a multiple of an odd batch size — the remainder must survive."""
    odd = 32
    torch.manual_seed(0)
    loader = DataLoader(
        SpectrumIndexDataset(spectra),
        batch_size=odd,
        shuffle=True,
        num_workers=0,
        collate_fn=batch_collate(spectra),
    )
    sizes = [b["scan_id"].shape[0] for b in loader]
    assert sum(sizes) == N_TOTAL
    assert sizes[-1] == N_TOTAL % odd


def test_same_seed_reproduces_both_epochs(spectra):
    def two_epochs(seed):
        loader = _loader(spectra, shuffle=True, seed=seed)
        return [_scan_ids(next(iter(loader))) for _ in range(2)]

    assert two_epochs(123) == two_epochs(123)
    assert two_epochs(123) != two_epochs(456)


def test_no_shuffle_keeps_stored_order(spectra):
    loader = _loader(spectra, shuffle=False)
    seen = []
    for batch in loader:
        seen.extend(batch["scan_id"].tolist())
    assert seen == list(range(N_TOTAL))


def test_len_is_the_exact_steps_per_epoch(spectra):
    """train.py uses len(train_loader) for the LR schedule, so it must be exact."""
    loader = _loader(spectra, shuffle=True)
    assert len(loader) == -(-N_TOTAL // BATCH)
    assert len(list(loader)) == len(loader)


def test_batches_stay_cross_file(spectra):
    """Shuffling must not regress the cross-file property build_lance.py provides."""
    batch = next(iter(_loader(spectra, shuffle=True)))
    assert len(set(batch["peak_file"])) >= N_FILES - 1


# --- steps/epoch for the sequential path -------------------------------------
#
# OneCycleLR raises once stepped past total_steps, so an undercount kills a run
# partway through. `ceil(n_spectra / batch_size)` undercounts because Lance batches
# never span fragments — every mzML file adds a partial batch.


def _multi_fragment(tmp_path, n_files, per_file, batch_size, name):
    def frame(off):
        return pl.DataFrame(
            [
                {
                    "peak_file": f"f{off}.mzML",
                    "scan_id": off + i,
                    "mz_array": [300.0, 400.0],
                    "intensity_array": [0.5, 0.5],
                }
                for i in range(per_file)
            ]
        )

    ds = SpectrumDataset(frame(0), batch_size=batch_size, path=str(tmp_path / name))
    for f in range(1, n_files):
        ds.add_spectra(frame(f * per_file))
    return ds


@pytest.mark.parametrize(
    "n_files,per_file,batch_size",
    [
        (1, 500, 64),  # single fragment: naive formula happens to be right
        (4, 125, 64),  # a few fragments
        (25, 20, 64),  # many small files -> naive formula badly undercounts
    ],
)
def test_iterable_steps_matches_the_real_batch_count(
    tmp_path, n_files, per_file, batch_size
):
    ds = _multi_fragment(tmp_path, n_files, per_file, batch_size, f"m{n_files}.lance")
    actual = sum(1 for _ in DataLoader(ds, batch_size=None, num_workers=0))
    assert iterable_steps_per_epoch(ds, batch_size) == actual
    del ds


def test_naive_formula_undercounts_past_the_old_safety_margin(tmp_path):
    """Pins the bug this replaced, so nobody 'simplifies' it back to a division.

    100 files x 200 spectra at batch 256: each file is under one batch and two don't
    fit in one, so the scan yields 100 batches where ceil(20000/256) predicts 79.
    The old code padded by 2% + 10, i.e. to 90 — still short, so OneCycleLR raised
    partway through the run. The shortfall grows with file count, which is why it
    hit some sweeps and not others.
    """
    batch_size = 256
    ds = _multi_fragment(tmp_path, 100, 200, batch_size, "naive.lance")
    actual = sum(1 for _ in DataLoader(ds, batch_size=None, num_workers=0))
    naive = -(-ds.n_spectra // batch_size)
    old_padded = naive * 1.02 + 10

    assert actual > naive
    assert actual > old_padded, "old margin would have covered it; pick a harsher shape"
    assert iterable_steps_per_epoch(ds, batch_size) == actual
    del ds
