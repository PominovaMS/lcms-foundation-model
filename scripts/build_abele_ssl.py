"""Build a train_mzml/val_mzml symlink farm from the abele SSL split.

Analogue of ``build_stage.py`` for abele: instead of taking whole PRIDE
accessions, it selects the abele mzML files that ``assign_splits`` marks as SSL
(``genus_class == -1`` — the genera NOT held out for the downstream probe) and
lays them out as::

    <out-root>/{train_mzml,val_mzml}

directly consumable by training::

    cd source && python train.py --data_dir <out-root> --run_name abele_ssl

The train/val split within the SSL files uses the same stable per-file hash as
``build_stage.py`` (via ``link_into_training``), so assignments are deterministic.

IMPORTANT: keep ``--n_probe_genera`` / ``--n_ssl_top`` identical to the
``probe_checkpoint.py`` call downstream, so the SSL pretraining files and the
probe genera stay disjoint (no leakage) and the probe split is comparable.

Example
-------
    python scripts/build_abele_ssl.py \\
        --data_dir /mnt/data/shared/lc_ms_foundation/abele_data/mzml \\
        --meta_path /mnt/data/shared/lc_ms_foundation/abele_data/all_abele_metadata.csv \\
        --out-root /mnt/data/shared/lc_ms_foundation/training_sets/abele_ssl \\
        --n_probe_genera 15 --n_ssl_top 3
"""

import argparse
import sys
from pathlib import Path

# Make project root + eval/ importable (assign_splits lives in eval/data.py;
# link_into_training lives alongside this file in scripts/).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import polars as pl

from data import load_metadata, assign_splits  # eval/data.py
from build_stage import link_into_training  # reuse stable split + symlinking


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--data_dir", required=True, help="Directory with abele mzML files"
    )
    p.add_argument("--meta_path", required=True, help="Path to abele metadata CSV")
    p.add_argument(
        "--out-root", required=True, help="Where {train_mzml,val_mzml} are created"
    )
    # Keep these identical to the probe_checkpoint.py call so SSL and probe splits agree.
    p.add_argument("--n_probe_genera", type=int, default=15)
    p.add_argument("--n_ssl_top", type=int, default=3)
    p.add_argument(
        "--val-frac", type=float, default=0.1, help="Fraction of SSL files held for val"
    )
    p.add_argument("--seed", type=int, default=0, help="Seed for the deterministic split")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap SSL mzML files (deterministic, sorted) for a fast smoke test",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Replace existing symlinks instead of leaving them in place",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    meta = load_metadata(args.meta_path)
    meta = assign_splits(
        meta, n_probe_genera=args.n_probe_genera, n_ssl_top=args.n_ssl_top
    )

    # SSL files = the "train" split (genus_class == -1) — genera not used for probing.
    ssl_files = meta.filter(pl.col("split") == "train")["peak_file"].to_list()

    data_dir = Path(args.data_dir)
    mzml_files = sorted(data_dir / f for f in ssl_files if (data_dir / f).exists())
    missing = [f for f in ssl_files if not (data_dir / f).exists()]
    if missing:
        print(
            f"WARNING: {len(missing)}/{len(ssl_files)} SSL files missing from {data_dir}",
            file=sys.stderr,
        )
    if args.limit is not None:
        mzml_files = mzml_files[: args.limit]
    if not mzml_files:
        raise SystemExit(f"No SSL mzML files found in {data_dir}")

    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    n_train, n_val = link_into_training(
        mzml_files, out, args.val_frac, args.seed, args.force
    )
    print(
        f"abele SSL -> {n_train} train, {n_val} val "
        f"({len(mzml_files)} mzML) under {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
