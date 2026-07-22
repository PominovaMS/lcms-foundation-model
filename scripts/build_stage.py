"""Build cumulative training-set symlink farms from already-ingested PRIDE data.

This is an *offline* symlink builder — it does NOT download or convert anything.
It assumes each dataset is already on disk as::

    <root>/<ACCESSION>/mzml/*.mzML

and produces, for a diversity-scaling experiment, a sequence of cumulative
training sets::

    <out-root>/stage01/{train_mzml,val_mzml}   # accession 1
    <out-root>/stage02/{train_mzml,val_mzml}   # accessions 1..2
    <out-root>/stage03/{train_mzml,val_mzml}   # accessions 1..3
    ...

Each stage dir is directly consumable by training::

    cd source && python train.py --data_dir <out-root>/stage03 --run_name stage03

The train/val split is decided per file name via a stable hash of ``seed:name``
(same scheme as the ingest pipeline), so a file keeps its assignment as more
datasets are added — stages stay comparable and never reshuffle earlier files.

Example
-------
    python scripts/build_stage.py \\
        --root /mnt/data/shared/lc_ms_foundation/pride_data \\
        --accessions PXD014877 PXD012345 PXD067890 \\
        --out-root /mnt/data/shared/lc_ms_foundation/training_sets/sweep \\
        --cumulative --val-frac 0.1 --seed 0
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path


def is_val(name: str, val_frac: float, seed: int) -> bool:
    """Stable per-file split: hashing the name keeps assignments fixed as data grows.

    Mirrors ``scripts/ingest_pxd.py::is_val`` so sets built by either tool agree.
    """
    digest = hashlib.md5(f"{seed}:{name}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < val_frac


def link_into_training(
    mzml_files: list[Path], target: Path, val_frac: float, seed: int, force: bool
) -> tuple[int, int]:
    """Symlink each mzML into target/{train_mzml,val_mzml} with a stable split.

    Mirrors ``scripts/ingest_pxd.py::link_into_training`` (kept dependency-free so
    this builder does not import the download pipeline).
    """
    train_dir = target / "train_mzml"
    val_dir = target / "val_mzml"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    n_train = n_val = 0
    for mzml in mzml_files:
        is_v = is_val(mzml.name, val_frac, seed)
        link = (val_dir if is_v else train_dir) / mzml.name
        if link.is_symlink() or link.exists():
            if force:
                link.unlink()
            # else: leave the existing link in place (idempotent)
        if not (link.is_symlink() or link.exists()):
            os.symlink(mzml.resolve(), link)
        n_val += is_v
        n_train += not is_v
    return n_train, n_val


def collect_mzml(root: Path, accession: str, mzml_subdir: str) -> list[Path]:
    """Return the mzML files for one accession, sorted for determinism."""
    acc_dir = root / accession / mzml_subdir
    if not acc_dir.is_dir():
        raise FileNotFoundError(
            f"No mzML directory for {accession}: {acc_dir} does not exist. "
            f"Ingest it first, or check --root / --mzml-subdir."
        )
    files = sorted(
        p for p in acc_dir.iterdir() if p.suffix.lower() == ".mzml" and p.is_file()
    )
    if not files:
        print(f"WARNING: {acc_dir} contains no .mzML files", file=sys.stderr)
    return files


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--root",
        required=True,
        help="Root holding <ACCESSION>/<mzml-subdir>/*.mzML (e.g. .../pride_data)",
    )
    p.add_argument(
        "--accessions",
        nargs="+",
        required=True,
        metavar="PXD",
        help="Accessions in diversity order; cumulative stages add them left to right",
    )
    p.add_argument(
        "--out-root",
        required=True,
        help="Where stage dirs (or the single set) are created",
    )
    p.add_argument(
        "--cumulative",
        action="store_true",
        help="Emit one stage per prefix (stage01=acc1, stage02=acc1..2, ...). "
        "Without it, build a single set from ALL accessions at --out-root.",
    )
    p.add_argument("--val-frac", type=float, default=0.1, help="Fraction held out for val")
    p.add_argument("--seed", type=int, default=0, help="Seed for the deterministic split")
    p.add_argument(
        "--mzml-subdir",
        default="mzml",
        help="Per-accession subdirectory holding the mzML files (default: mzml)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Replace existing symlinks instead of leaving them in place",
    )
    return p.parse_args(argv)


def build_one_stage(
    stage_dir: Path,
    accessions: list[str],
    root: Path,
    args,
) -> None:
    """Build a single stage dir from the given accessions and record provenance."""
    mzml_files: list[Path] = []
    for acc in accessions:
        mzml_files.extend(collect_mzml(root, acc, args.mzml_subdir))

    n_train, n_val = link_into_training(
        mzml_files, stage_dir, args.val_frac, args.seed, args.force
    )
    # Record which accessions this stage contains (provenance for the sweep).
    (stage_dir / "accessions.txt").write_text("\n".join(accessions) + "\n")
    print(
        f"{stage_dir.name}: {len(accessions)} dataset(s) "
        f"[{', '.join(accessions)}] -> {n_train} train, {n_val} val "
        f"({len(mzml_files)} mzML)"
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.cumulative:
        for i in range(1, len(args.accessions) + 1):
            stage_dir = out_root / f"stage{i:02d}"
            build_one_stage(stage_dir, args.accessions[:i], root, args)
    else:
        build_one_stage(out_root, args.accessions, root, args)

    print(f"Done. Stages under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
