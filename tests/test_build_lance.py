"""Tests for the pre-shuffled dataset builder (scripts/build_lance.py).

These exercise ``shuffle_into`` against a real Lance dataset built from a synthetic
polars frame — no mzML parsing, no GPU. The frame carries a ``peak_file`` column so
the cross-file mixing that motivates the whole change can be asserted directly.
"""

import json
import sys
from pathlib import Path

import lance
import polars as pl
import pytest
from depthcharge.data import SpectrumDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_lance  # noqa: E402
from build_lance import shuffle_into  # noqa: E402

N_FILES = 8
PER_FILE = 250
N_TOTAL = N_FILES * PER_FILE
SEED = 42


@pytest.fixture
def staged(tmp_path):
    """A Lance dataset in file order: all of file 0, then all of file 1, ...

    This is the layout `train.py --data_dir` produces, and the one that makes every
    batch come from a single file.
    """
    rows = []
    for f in range(N_FILES):
        for i in range(PER_FILE):
            rows.append(
                {
                    "peak_file": f"file{f:02d}.mzML",
                    "scan_id": f * PER_FILE + i,
                    "mz_array": [100.0 + i, 200.0 + i, 300.0 + i],
                    "intensity_array": [0.1, 0.2, 0.3],
                }
            )
    ds = SpectrumDataset(pl.DataFrame(rows), batch_size=64, path=str(tmp_path / "stage.lance"))
    yield ds.dataset
    del ds


def _rows_of(path):
    return lance.dataset(str(path)).to_table().to_pylist()


def test_preserves_every_row_exactly_once(staged, tmp_path):
    """Nothing dropped, nothing duplicated — the multiset must be identical."""
    out = tmp_path / "out.lance"
    assert shuffle_into(staged, out, SEED, chunk_size=128) == N_TOTAL

    got = sorted(r["scan_id"] for r in _rows_of(out))
    assert got == list(range(N_TOTAL))


def test_changes_the_order(staged, tmp_path):
    out = tmp_path / "out.lance"
    shuffle_into(staged, out, SEED, chunk_size=128)
    assert [r["scan_id"] for r in _rows_of(out)] != list(range(N_TOTAL))


def test_same_seed_reproduces_the_same_order(staged, tmp_path):
    """What makes a built dataset reusable across hyperparameter runs."""
    a, b = tmp_path / "a.lance", tmp_path / "b.lance"
    shuffle_into(staged, a, SEED, chunk_size=128)
    shuffle_into(staged, b, SEED, chunk_size=128)
    assert [r["scan_id"] for r in _rows_of(a)] == [r["scan_id"] for r in _rows_of(b)]


def test_different_seed_gives_a_different_order(staged, tmp_path):
    a, b = tmp_path / "a.lance", tmp_path / "b.lance"
    shuffle_into(staged, a, SEED, chunk_size=128)
    shuffle_into(staged, b, SEED + 1, chunk_size=128)
    assert [r["scan_id"] for r in _rows_of(a)] != [r["scan_id"] for r in _rows_of(b)]


def test_chunk_size_does_not_change_the_contents(staged, tmp_path):
    """Chunking only bounds RAM; it must not affect which rows land in the output."""
    a, b = tmp_path / "a.lance", tmp_path / "b.lance"
    shuffle_into(staged, a, SEED, chunk_size=64)
    shuffle_into(staged, b, SEED, chunk_size=N_TOTAL)
    assert sorted(r["scan_id"] for r in _rows_of(a)) == sorted(
        r["scan_id"] for r in _rows_of(b)
    )


def test_batches_become_cross_file(staged, tmp_path):
    """The point of the exercise.

    Before: a batch-sized window is one file. After: a window spans many files, so a
    gradient step sees a cross-section of the corpus instead of one LC-MS run.
    """
    batch = 64
    before = {r["peak_file"] for r in _rows_of(staged.uri)[:batch]}
    assert len(before) == 1  # the problem being fixed

    out = tmp_path / "out.lance"
    shuffle_into(staged, out, SEED, chunk_size=128)
    after = {r["peak_file"] for r in _rows_of(out)[:batch]}
    # With 8 files uniformly mixed, a 64-row window should touch essentially all of
    # them; assert well above 1 rather than exactly 8 to stay robust to the RNG.
    assert len(after) >= N_FILES - 1


def _frame(n_files=3, per_file=100):
    return pl.DataFrame(
        [
            {
                "peak_file": f"f{f}.mzML",
                "scan_id": f * per_file + i,
                "mz_array": [100.0 + i, 200.0 + i],
                "intensity_array": [0.5, 0.5],
            }
            for f in range(n_files)
            for i in range(per_file)
        ]
    )


@pytest.fixture
def stage_dir(tmp_path, monkeypatch):
    """A stage dir plus a stubbed ingest, so main() runs without mzML parsing."""
    for split in ("train", "val"):
        (tmp_path / f"{split}_mzml").mkdir()

    def fake_build_dataset(data_dir, preprocessing_fn, batch_size, path=None):
        per_file = 100 if "train" in str(data_dir) else 40  # -> 300 train, 120 val
        ds = SpectrumDataset(
            _frame(n_files=3, per_file=per_file), batch_size=batch_size, path=path
        )
        return ds, ["a.mzML", "b.mzML", "c.mzML"], []

    monkeypatch.setattr(build_lance, "build_dataset", fake_build_dataset)
    return tmp_path


def test_main_writes_both_splits_and_a_sidecar(stage_dir, tmp_path):
    out = tmp_path / "lance"
    assert build_lance.main(["--data_dir", str(stage_dir), "--out", str(out), "--seed", "7"]) == 0

    assert (out / "train.lance").is_dir() and (out / "val.lance").is_dir()
    info = json.loads((out / "dataset_info.json").read_text())
    assert info["seed"] == 7
    assert info["splits"]["train"]["n_spectra"] == 300
    assert info["splits"]["val"]["n_spectra"] == 120
    # The guard train.py checks before training on a prebuilt dataset.
    assert info["preprocessing"]["max_num_peaks"] > 0


def test_main_is_idempotent_and_force_rebuilds(stage_dir, tmp_path):
    out = tmp_path / "lance"
    argv = ["--data_dir", str(stage_dir), "--out", str(out), "--seed", "7"]
    build_lance.main(argv)
    stamp = json.loads((out / "dataset_info.json").read_text())["built_at"]

    build_lance.main(argv)  # no --force: should not rebuild
    assert json.loads((out / "dataset_info.json").read_text())["built_at"] == stamp

    build_lance.main([*argv, "--force"])
    assert (out / "train.lance").is_dir()


def test_sidecar_is_written_last_so_a_partial_build_is_not_trusted(
    stage_dir, tmp_path, monkeypatch
):
    """A failed rebuild must not leave a sidecar describing data that isn't there.

    train.py trusts dataset_info.json to describe the .lance dirs next to it, so a
    stale one surviving a crashed rebuild would be worse than no dataset at all.
    """
    out = tmp_path / "lance"
    argv = ["--data_dir", str(stage_dir), "--out", str(out), "--seed", "7"]
    build_lance.main(argv)
    assert (out / "dataset_info.json").exists()

    def boom(*a, **k):
        raise RuntimeError("ingest exploded")

    monkeypatch.setattr(build_lance, "build_dataset", boom)
    with pytest.raises(RuntimeError):
        build_lance.main([*argv, "--force"])
    assert not (out / "dataset_info.json").exists()


def test_every_window_is_cross_file_not_just_the_first(staged, tmp_path):
    """Guards a specific regression: chunk-internal ordering.

    Reading a chunk's indices sorted is good for locality, but writing them still
    sorted leaves each chunk file-ordered inside, so batches sweep the corpus
    low-id-first and the sawtooth returns with period chunk_size/batch_size. That is
    invisible if you only inspect the first window, so check all of them — and at a
    chunk/batch ratio like production's (50000/1024).
    """
    batch, chunk = 25, 500
    out = tmp_path / "out.lance"
    shuffle_into(staged, out, SEED, chunk_size=chunk)

    rows = _rows_of(out)
    per_window = [
        len({r["peak_file"] for r in rows[i : i + batch]})
        for i in range(0, len(rows) - batch, batch)
    ]
    assert min(per_window) >= N_FILES - 2, (
        f"some window is dominated by too few files: {per_window}"
    )
