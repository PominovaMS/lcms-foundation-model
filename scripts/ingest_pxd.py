"""Automated ProteomeXchange (PXD) ingestion.

Download a ProteomeXchange dataset by accession, convert Thermo ``.raw`` files to
mzML, and land the result in the shared server repository ready for training.

Example
-------
    python scripts/ingest_pxd.py PXD014877 \
        --glob "*iRT*.raw" --limit 20 \
        --link-into /mnt/data/shared/lc_ms_foundation/training_sets/current \
        --val-frac 0.1 --jobs 4

Prerequisites (see scripts/README.md):
    - ``ppx`` (pip) and ``ThermoRawFileParser`` (bioconda) on PATH.
    - Run on the Linux host where ``/mnt/data/shared/lc_ms_foundation`` is mounted.
"""

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ppx

DEFAULT_ROOT = "/mnt/data/shared/lc_ms_foundation/pride_data"
CONVERTER_CMD = os.environ.get("THERMORAWFILEPARSER", "ThermoRawFileParser")
RAW_SUFFIXES = {".raw"}
MZML_SUFFIXES = {".mzml"}


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
class Manifest:
    """Per-collection JSON tracking download + conversion state for resumability.

    Stored at ``<collection>/manifest.json`` keyed by remote file name. Each entry
    holds ``{size, downloaded, converted, mzml}``. Saved atomically so an interrupted
    run never corrupts it.
    """

    def __init__(self, path: Path, entries: dict | None = None):
        self.path = path
        self.entries = entries or {}

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        if path.exists():
            with open(path) as f:
                return cls(path, json.load(f))
        return cls(path, {})

    def get(self, name: str) -> dict:
        return self.entries.setdefault(name, {})

    def mark(self, name: str, **fields) -> None:
        self.get(name).update(fields)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.entries, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def find_and_list(accession: str, raw_dir: Path):
    """Resolve the project on ProteomeXchange and list its remote files."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    project = ppx.find_project(accession, local=str(raw_dir))
    return project, project.remote_files()


def select_files(remote: list[str], glob: str | None, limit: int | None) -> list[str]:
    """Filter remote file names by glob and cap by limit, deterministically."""
    files = sorted(remote)
    if glob:
        files = [f for f in files if fnmatch.fnmatch(os.path.basename(f), glob)]
    if limit is not None:
        files = files[:limit]
    return files


def download_files(
    project, names: list[str], raw_dir: Path, manifest: Manifest, force: bool
) -> list[Path]:
    """Download the selected files, skipping those already present (ppx + manifest)."""
    todo = []
    for name in names:
        local = raw_dir / os.path.basename(name)
        entry = manifest.get(name)
        if not force and entry.get("downloaded") and local.exists():
            continue
        todo.append(name)

    if todo:
        # ppx downloads into the project's `local` dir and skips size-matched files.
        project.download(todo, force_=force)

    downloaded = []
    for name in names:
        local = raw_dir / os.path.basename(name)
        if local.exists():
            manifest.mark(name, downloaded=True, size=local.stat().st_size)
            downloaded.append(local)
        else:
            print(f"WARNING: expected download missing: {local}", file=sys.stderr)
    return downloaded


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def needs_conversion(files: list[Path]) -> bool:
    """True if any file is a Thermo .raw; False if the collection is already mzML."""
    return any(f.suffix.lower() in RAW_SUFFIXES for f in files)


def convert_one(raw: Path, mzml_dir: Path) -> Path:
    """Convert a single .raw to indexed mzML with ThermoRawFileParser.

    ``-f=2`` = indexed mzML. All MS levels are retained; MS-level filtering happens
    downstream in ``spectra_to_df(ms_level=1)``.
    """
    out = mzml_dir / (raw.stem + ".mzML")
    mzml_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [CONVERTER_CMD, f"-i={raw}", f"-b={out}", "-f=2"],
        check=True,
    )
    return out


def convert_all(
    raws: list[Path], mzml_dir: Path, jobs: int, manifest: Manifest, force: bool
) -> list[Path]:
    """Convert every .raw to mzML, skipping already-converted files. Resumable."""
    # Map local raw path back to its manifest key (remote name).
    key_by_stem = {
        os.path.splitext(os.path.basename(name))[0]: name for name in manifest.entries
    }

    pending = []
    done = []
    for raw in raws:
        out = mzml_dir / (raw.stem + ".mzML")
        name = key_by_stem.get(raw.stem, raw.name)
        entry = manifest.get(name)
        if not force and entry.get("converted") and out.exists():
            done.append(out)
            continue
        pending.append((raw, name))

    def _work(item):
        raw, name = item
        out = convert_one(raw, mzml_dir)
        return name, out

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            for name, out in pool.map(_work, pending):
                manifest.mark(name, converted=True, mzml=str(out))
                done.append(out)
    return done


# --------------------------------------------------------------------------- #
# Training wiring (symlink farm)
# --------------------------------------------------------------------------- #
def is_val(name: str, val_frac: float, seed: int) -> bool:
    """Stable per-file split: hashing the name keeps assignments fixed as data grows."""
    digest = hashlib.md5(f"{seed}:{name}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < val_frac


def link_into_training(
    mzml_files: list[Path], target: Path, val_frac: float, seed: int, force: bool
) -> tuple[int, int]:
    """Symlink each mzML into target/{train_mzml,val_mzml} with a stable file-level split."""
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


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("accession", help="ProteomeXchange accession, e.g. PXD014877")
    p.add_argument("--root", default=DEFAULT_ROOT, help="Collection root directory")
    p.add_argument("--glob", default=None, help="fnmatch filter on remote file names")
    p.add_argument("--limit", type=int, default=None, help="Cap number of files")
    p.add_argument(
        "--all",
        action="store_true",
        help="Opt in to downloading the entire dataset (guardrail for large datasets)",
    )
    p.add_argument("--jobs", type=int, default=1, help="Parallel conversions")
    p.add_argument("--link-into", default=None, help="Training dir to symlink mzml into")
    p.add_argument("--val-frac", type=float, default=0.1, help="Fraction held out for val")
    p.add_argument("--seed", type=int, default=0, help="Seed for the deterministic split")
    p.add_argument("--dry-run", action="store_true", help="List + plan only")
    p.add_argument("--force", action="store_true", help="Ignore manifest, redo everything")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    collection = Path(args.root) / args.accession
    raw_dir = collection / "raw"
    mzml_dir = collection / "mzml"
    manifest = Manifest.load(collection / "manifest.json")

    print(f"Resolving {args.accession} ...")
    project, remote = find_and_list(args.accession, raw_dir)
    print(f"{len(remote)} files available on ProteomeXchange.")

    selected = select_files(remote, args.glob, args.limit)

    # Guardrail: never bulk-download without an explicit narrowing/opt-in flag.
    if not (args.glob or args.limit is not None or args.all):
        print(
            f"\nNo --glob / --limit / --all given. Refusing to download all "
            f"{len(remote)} files (potentially very large).\n"
            "Narrow with --glob/--limit, or pass --all to fetch everything.",
            file=sys.stderr,
        )
        return 2

    print(f"{len(selected)} files selected.")
    if args.dry_run:
        for name in selected:
            print(f"  {name}")
        print("\n(dry run — nothing downloaded or converted)")
        return 0

    downloaded = download_files(project, selected, raw_dir, manifest, args.force)
    print(f"Downloaded/present: {len(downloaded)} files in {raw_dir}")

    if needs_conversion(downloaded):
        raws = [f for f in downloaded if f.suffix.lower() in RAW_SUFFIXES]
        mzml_files = convert_all(raws, mzml_dir, args.jobs, manifest, args.force)
        # Include any non-raw mzML that shipped alongside.
        mzml_files += [f for f in downloaded if f.suffix.lower() in MZML_SUFFIXES]
    else:
        # Dataset already ships mzML; mirror it into the mzml/ dir via symlink.
        mzml_dir.mkdir(parents=True, exist_ok=True)
        mzml_files = []
        for f in downloaded:
            if f.suffix.lower() in MZML_SUFFIXES:
                link = mzml_dir / f.name
                if not (link.exists() or link.is_symlink()):
                    os.symlink(f.resolve(), link)
                mzml_files.append(link)
    print(f"mzML ready: {len(mzml_files)} files in {mzml_dir}")

    if args.link_into:
        n_train, n_val = link_into_training(
            mzml_files, Path(args.link_into), args.val_frac, args.seed, args.force
        )
        print(f"Linked into {args.link_into}: {n_train} train, {n_val} val")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
