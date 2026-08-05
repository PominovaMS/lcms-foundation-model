"""Build a persistent, pre-shuffled Lance dataset from a stage's mzML.

Ingests ``<data_dir>/{train_mzml,val_mzml}`` and writes the spectra out in random
order, so that a plain sequential read during training yields batches that are a
cross-section of files and datasets rather than a slice of one mzML.

Why shuffle here rather than in the DataLoader
----------------------------------------------
Lance scans fragments sequentially and ``train.py`` ingests files in sorted-name
order, so unshuffled training replays an identical sequence every epoch and every
batch of 1024 comes from a single ~195k-spectrum file. That shows up as a loss
sawtooth locked to position within the epoch. Shuffling the rows *on disk* fixes
both, costs nothing at training time (still a sequential scan — no random reads),
and leaves the batch format untouched.

It also makes the dataset **reusable**: without it, every run re-parses the whole
mzML corpus into a temporary Lance DB and throws it away. Build once, then sweep
hyperparameters against byte-identical data.

Note that the on-disk order is fixed, so each epoch sees the same batch composition
— the shuffle is global but not re-drawn per epoch. That is the tradeoff that makes
the data reusable.

Preprocessing (``filter_intensity(max_num_peaks=...)`` then sqrt scaling) is applied
at ingest and therefore baked into the output. ``dataset_info.json`` records it, and
``train.py --lance_dir`` refuses to run if it disagrees with the active config.

Disk
----
The staging and shuffled copies coexist during the build, so it transiently needs
roughly 2x the final size (~11 GB for a 3.5M-spectrum corpus, so budget ~22 GB).
Staging is removed on success.

Usage
-----
    python scripts/build_lance.py \\
        --data_dir /mnt/data/shared/lc_ms_foundation/training_sets/sweep/stage03 \\
        --out      /mnt/data/shared/lc_ms_foundation/training_sets/sweep/stage03/lance \\
        --seed 42

    # then
    cd source && python train.py --lance_dir <...>/stage03/lance --run_name stage03

Writes ``<out>/{train.lance,val.lance,dataset_info.json}``.

Reuse
-----
Re-running is cheap when nothing changed: the build is skipped unless the mzML file
list, the ``--seed``, or the config's ``max_num_peaks`` differs from what
``dataset_info.json`` records (or a ``.lance`` dir has gone missing). That is what
makes a hyperparameter sweep over one stage pay the ingest cost once.

The comparison is by file *name*, so replacing a symlink target with different
content under the same name goes unnoticed — pass ``--force`` for that, or any time
you want an unconditional rebuild.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Make the project root importable (build_dataset lives in source/dataset.py).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lance  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import yaml  # noqa: E402
from depthcharge.data import preprocessing  # noqa: E402

from source.dataset import build_dataset  # noqa: E402

SPLITS = ("train", "val")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--data_dir",
        required=True,
        help="Stage dir holding train_mzml/ and val_mzml/ (see build_stage.py)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output dir for the .lance databases (default: <data_dir>/lance)",
    )
    p.add_argument(
        "--config",
        default=str(ROOT / "config.yaml"),
        help="Config to read data.max_num_peaks and data.batch_size from",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed. The same seed reproduces the same on-disk order.",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Spectra per shuffled write chunk. Bounds peak RAM during the shuffle; "
        "lower it if the build OOMs on peak-dense data.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild unconditionally. Without it the build is skipped when the mzML "
        "file list, seed and max_num_peaks all match what dataset_info.json records.",
    )
    return p.parse_args(argv)


def git_commit() -> str | None:
    """Short commit of this repo, so a dataset can be traced to the code that built it."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def shuffle_into(staging, out_path: Path, seed: int, chunk_size: int) -> int:
    """Write ``staging``'s rows to ``out_path`` in a seeded random order.

    Lance has no shuffle API and ``compact_files`` explicitly preserves insertion
    order, so this permutes explicitly: draw a permutation, then ``take`` it in
    chunks and append each chunk. Chunking is what bounds peak RAM — the whole
    corpus never has to be materialised at once.
    """
    n = staging.count_rows()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    first = True
    for start in range(0, n, chunk_size):
        idx = perm[start : start + chunk_size]
        # Read sorted for locality (lance's own `sample()` sorts for the same reason),
        # then permute the chunk in memory before writing. Both halves matter: a sorted
        # read is much cheaper, but writing in sorted order would leave each chunk
        # internally file-ordered, so consecutive batches would sweep the corpus
        # low-id-first and the sawtooth would come back with a period of
        # chunk_size/batch_size. Cheap to get wrong, hence test_batches_become_cross_file.
        table = staging.take(np.sort(idx).tolist())
        table = table.take(pa.array(rng.permutation(len(table))))
        lance.write_dataset(
            table,
            str(out_path),
            mode="create" if first else "append",
            schema=table.schema,
        )
        first = False
        print(f"  shuffled {min(start + chunk_size, n):,}/{n:,} spectra", flush=True)

    written = lance.dataset(str(out_path)).count_rows()
    if written != n:
        raise SystemExit(
            f"Shuffle lost rows: staged {n}, wrote {written} to {out_path}. "
            f"Refusing to leave a truncated dataset behind."
        )
    return n


def mzml_in(split_dir: Path) -> list[str]:
    """The mzML filenames in a split dir, matching what build_dataset would ingest."""
    return sorted(
        f
        for f in os.listdir(split_dir)
        if f.lower().endswith((".mzml", ".mzml.gz"))
    )


def staleness_reason(info, data_dir: Path, out_dir: Path, seed: int, max_num_peaks: int):
    """Why an existing build can't be reused, or None if it can.

    Existence alone is not enough. A stage dir is a regenerable symlink farm, so its
    contents change whenever `--accessions` or `--limit` change — and silently
    training on a stale corpus is far worse than paying to re-ingest. Checking the
    recorded inputs against what is on disk now is what lets the common case (same
    data, different hyperparameters) skip ingestion safely.

    Note this compares file *names*. Replacing a symlink target with different
    content under the same name is invisible here; use ``--force`` for that.
    """
    if info.get("seed") != seed:
        return f"seed changed ({info.get('seed')} -> {seed})"

    built_peaks = info.get("preprocessing", {}).get("max_num_peaks")
    if built_peaks != max_num_peaks:
        return f"max_num_peaks changed ({built_peaks} -> {max_num_peaks})"

    for split in SPLITS:
        if not (out_dir / f"{split}.lance").is_dir():
            return f"{split}.lance is missing"

        recorded = info.get("splits", {}).get(split)
        if recorded is None:
            return f"sidecar has no record of the {split} split"
        # Corrupt files were skipped at ingest but are still on disk, so the union is
        # what was actually present last time. Comparing against `files` alone would
        # make any corpus with an unreadable file rebuild on every single run.
        before = sorted(recorded.get("files", []) + recorded.get("unreadable_files", []))
        now = mzml_in(data_dir / f"{split}_mzml")
        if before != now:
            added = len(set(now) - set(before))
            removed = len(set(before) - set(now))
            return f"{split} inputs changed (+{added} / -{removed} files)"

    return None


def main(argv=None):
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out) if args.out else data_dir / "lance"
    info_path = out_dir / "dataset_info.json"

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    max_num_peaks = cfg["data"]["max_num_peaks"]
    batch_size = cfg["data"]["batch_size"]

    for split in SPLITS:
        if not (data_dir / f"{split}_mzml").is_dir():
            raise SystemExit(f"Missing {data_dir / f'{split}_mzml'}")

    if info_path.exists() and not args.force:
        with open(info_path) as f:
            existing = json.load(f)
        reason = staleness_reason(existing, data_dir, out_dir, args.seed, max_num_peaks)
        if reason is None:
            counts = "  ".join(
                f"{s}={existing['splits'][s]['n_spectra']:,}" for s in SPLITS
            )
            print(
                f"Reusing {out_dir} — inputs unchanged ({counts} spectra, "
                f"seed {args.seed}, built {existing.get('built_at')}). "
                f"Pass --force to rebuild."
            )
            return 0
        print(f"Rebuilding {out_dir}: {reason}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # A stale info file must not survive a failed rebuild — a later run would then
    # trust a sidecar describing data that no longer matches the .lance dirs.
    info_path.unlink(missing_ok=True)

    preprocessing_fn = [
        preprocessing.filter_intensity(max_num_peaks=max_num_peaks),
        preprocessing.scale_intensity(scaling="root", max_intensity=1.0),
    ]

    info = {
        "seed": args.seed,
        "data_dir": str(data_dir.resolve()),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        # Baked in at ingest: a dataset built under one peak cap cannot be reused
        # under another. train.py --lance_dir checks this.
        "preprocessing": {
            "max_num_peaks": max_num_peaks,
            "intensity_scaling": "root",
            "ms_level": 1,
        },
        "splits": {},
    }

    for split in SPLITS:
        src = data_dir / f"{split}_mzml"
        dst = out_dir / f"{split}.lance"
        print(f"\n=== {split}: ingesting {src} ===", flush=True)

        if dst.exists():
            shutil.rmtree(dst)

        # Stage to a temp dir first: the shuffle needs every file present before it
        # can permute across them, and ingest is per-file streaming by design.
        with tempfile.TemporaryDirectory(prefix=f"lance_stage_{split}_") as tmp:
            staged, ingested, corrupt = build_dataset(
                str(src),
                preprocessing_fn,
                batch_size,
                path=os.path.join(tmp, f"{split}_staging.lance"),
            )
            n_spectra = staged.n_spectra
            print(
                f"  ingested {len(ingested)} file(s), {n_spectra:,} spectra"
                f"{f' ({len(corrupt)} unreadable)' if corrupt else ''}"
            )
            print(f"=== {split}: shuffling into {dst} (seed {args.seed}) ===", flush=True)
            shuffle_into(staged.dataset, dst, args.seed, args.chunk_size)
            # Drop our handle before the TemporaryDirectory is torn down.
            del staged

        info["splits"][split] = {
            "n_spectra": n_spectra,
            "path": dst.name,
            "n_files": len(ingested),
            "files": ingested,
            "unreadable_files": corrupt,
        }

    # Written last, so its presence means the build completed.
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nWrote {out_dir}")
    for split in SPLITS:
        s = info["splits"][split]
        print(f"  {split}.lance  {s['n_spectra']:,} spectra from {s['n_files']} file(s)")
    print(f"  dataset_info.json  (seed {args.seed}, max_num_peaks {max_num_peaks})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
