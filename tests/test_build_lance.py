"""Tests for the pre-shuffled dataset builder (scripts/build_lance.py).

These exercise ``shuffle_into`` against a real Lance dataset built from a synthetic
polars frame — no mzML parsing, no GPU. The frame carries a ``peak_file`` column so
the cross-file mixing that motivates the whole change can be asserted directly.
"""

import json
import shutil
import sys
from pathlib import Path

import lance
import polars as pl
import pytest
import yaml
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


class _Stage:
    """A stage dir with a stubbed ingest that records how often it actually ran.

    Counting ingests is the whole point: the bug being fixed was a re-run silently
    re-parsing every mzML, which a check on ``built_at`` alone would not have caught.
    """

    def __init__(self, path, calls, corrupt):
        self.path = path
        self.calls = calls
        self.corrupt = corrupt

    def files(self, split):
        return sorted(p.name for p in (self.path / f"{split}_mzml").iterdir())

    def add(self, split, name):
        (self.path / f"{split}_mzml" / name).touch()

    def remove(self, split, name):
        (self.path / f"{split}_mzml" / name).unlink()


@pytest.fixture
def stage_dir(tmp_path, monkeypatch):
    """A stage dir plus a stubbed ingest, so main() runs without mzML parsing."""
    for split in ("train", "val"):
        d = tmp_path / f"{split}_mzml"
        d.mkdir()
        for name in ("a.mzML", "b.mzML", "c.mzML"):
            (d / name).touch()

    calls = []
    corrupt = []

    def fake_build_dataset(data_dir, preprocessing_fn, batch_size, path=None):
        calls.append(str(data_dir))
        present = build_lance.mzml_in(Path(data_dir))
        ingested = [f for f in present if f not in corrupt]
        per_file = 100 if "train" in str(data_dir) else 40  # -> 300 train, 120 val
        ds = SpectrumDataset(
            _frame(n_files=max(len(ingested), 1), per_file=per_file),
            batch_size=batch_size,
            path=path,
        )
        return ds, ingested, [f for f in present if f in corrupt]

    monkeypatch.setattr(build_lance, "build_dataset", fake_build_dataset)
    return _Stage(tmp_path, calls, corrupt)


def _argv(stage, out, **over):
    argv = ["--data_dir", str(stage.path), "--out", str(out), "--seed", "7"]
    for k, v in over.items():
        argv += [f"--{k}", str(v)]
    return argv


def test_main_writes_both_splits_and_a_sidecar(stage_dir, tmp_path):
    out = tmp_path / "lance"
    assert build_lance.main(_argv(stage_dir, out)) == 0

    assert (out / "train.lance").is_dir() and (out / "val.lance").is_dir()
    info = json.loads((out / "dataset_info.json").read_text())
    assert info["seed"] == 7
    assert info["splits"]["train"]["n_spectra"] == 300
    assert info["splits"]["val"]["n_spectra"] == 120
    # The guard train.py checks before training on a prebuilt dataset.
    assert info["preprocessing"]["max_num_peaks"] > 0


# --- reuse rules --------------------------------------------------------------
#
# A sweep re-runs the same stage with different hyperparameters. Re-parsing the whole
# corpus each time is what these guard against — and the counter is on ingests, not
# timestamps, because that is the cost that actually hurt.


def test_unchanged_inputs_are_not_re_ingested(stage_dir, tmp_path):
    out = tmp_path / "lance"
    build_lance.main(_argv(stage_dir, out))
    assert len(stage_dir.calls) == 2  # train + val

    assert build_lance.main(_argv(stage_dir, out)) == 0
    assert len(stage_dir.calls) == 2, "re-ran ingest despite unchanged inputs"


def test_force_rebuilds_unchanged_inputs(stage_dir, tmp_path):
    out = tmp_path / "lance"
    build_lance.main(_argv(stage_dir, out))
    build_lance.main([*_argv(stage_dir, out), "--force"])
    assert len(stage_dir.calls) == 4


@pytest.mark.parametrize("split", ["train", "val"])
def test_added_file_rebuilds(stage_dir, tmp_path, split):
    out = tmp_path / "lance"
    build_lance.main(_argv(stage_dir, out))
    stage_dir.add(split, "d.mzML")

    build_lance.main(_argv(stage_dir, out))
    assert len(stage_dir.calls) == 4
    info = json.loads((out / "dataset_info.json").read_text())
    assert "d.mzML" in info["splits"][split]["files"]


def test_removed_file_rebuilds(stage_dir, tmp_path):
    out = tmp_path / "lance"
    build_lance.main(_argv(stage_dir, out))
    stage_dir.remove("train", "c.mzML")

    build_lance.main(_argv(stage_dir, out))
    assert len(stage_dir.calls) == 4
    info = json.loads((out / "dataset_info.json").read_text())
    assert "c.mzML" not in info["splits"]["train"]["files"]


def test_changed_seed_rebuilds(stage_dir, tmp_path):
    """A different seed means a different stored order, so the data is not the same."""
    out = tmp_path / "lance"
    build_lance.main(_argv(stage_dir, out))
    build_lance.main(["--data_dir", str(stage_dir.path), "--out", str(out), "--seed", "8"])
    assert len(stage_dir.calls) == 4


def test_changed_max_num_peaks_rebuilds(stage_dir, tmp_path):
    """Preprocessing is baked in at ingest, so a new peak cap needs a new dataset."""
    out = tmp_path / "lance"
    build_lance.main(_argv(stage_dir, out))

    cfg = yaml.safe_load(open(build_lance.ROOT / "config.yaml"))
    cfg["data"]["max_num_peaks"] = cfg["data"]["max_num_peaks"] + 1
    alt = tmp_path / "alt.yaml"
    alt.write_text(yaml.safe_dump(cfg))

    build_lance.main(_argv(stage_dir, out, config=alt))
    assert len(stage_dir.calls) == 4


def test_missing_lance_dir_rebuilds(stage_dir, tmp_path):
    """The sidecar alone is not evidence the data is still there."""
    out = tmp_path / "lance"
    build_lance.main(_argv(stage_dir, out))
    shutil.rmtree(out / "train.lance")

    build_lance.main(_argv(stage_dir, out))
    assert len(stage_dir.calls) == 4
    assert (out / "train.lance").is_dir()


def test_unreadable_file_still_on_disk_is_not_a_change(stage_dir, tmp_path):
    """Corrupt files are skipped at ingest but stay on disk.

    Comparing only against the *ingested* list would see them as newly added and
    rebuild on every single run — for any corpus containing one bad file.
    """
    stage_dir.corrupt.append("b.mzML")
    out = tmp_path / "lance"
    build_lance.main(_argv(stage_dir, out))
    info = json.loads((out / "dataset_info.json").read_text())
    assert info["splits"]["train"]["unreadable_files"] == ["b.mzML"]

    build_lance.main(_argv(stage_dir, out))
    assert len(stage_dir.calls) == 2, "an unreadable file forced a needless rebuild"


def test_sidecar_is_written_last_so_a_partial_build_is_not_trusted(
    stage_dir, tmp_path, monkeypatch
):
    """A failed rebuild must not leave a sidecar describing data that isn't there.

    train.py trusts dataset_info.json to describe the .lance dirs next to it, so a
    stale one surviving a crashed rebuild would be worse than no dataset at all.
    """
    out = tmp_path / "lance"
    argv = _argv(stage_dir, out)
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
