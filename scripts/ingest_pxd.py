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
import ftplib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
def find_and_list(accession: str, raw_dir: Path, timeout: float = 60.0):
    """Resolve the project on ProteomeXchange and list its remote files.

    ``timeout`` is the per-socket-operation timeout (seconds) handed to ppx's
    FTP client. ppx defaults to 10s, which trips constantly on large files from
    slow PRIDE mirrors — bump it well up.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    project = ppx.find_project(accession, local=str(raw_dir), timeout=timeout)
    return project, project.remote_files()


def select_files(
    remote: list[str],
    glob: str | None,
    limit: int | None,
    reverse: bool = False,
) -> list[str]:
    """Filter remote file names by glob and cap by limit, deterministically.

    With ``reverse`` the sorted list is walked from the end *before* ``limit`` is
    applied, so the capped slice is the last ``limit`` files instead of the first.
    Running one job forward and one reversed lets two processes ingest opposite
    ends of the same large dataset concurrently. They only overlap if the dataset
    has fewer than ``2 * limit`` matching files; in the overlap both jobs would
    fetch the same files, but the on-disk mzML checks in ``converted_mzml`` make
    that idempotent (redundant, not corrupting).
    """
    files = sorted(remote)
    if glob:
        files = [f for f in files if fnmatch.fnmatch(os.path.basename(f), glob)]
    if reverse:
        files = list(reversed(files))
    if limit is not None:
        files = files[:limit]
    return files


def _download_with_resume(
    project,
    todo: list[str],
    force: bool,
    max_reconnects: int,
    attempts: int,
) -> None:
    """Call ``project.download`` repeatedly, resuming until it completes.

    ppx opens partial downloads in append mode and issues an FTP ``REST``, so a
    torn-down transfer resumes from the last byte on the next call — both across
    ppx's internal reconnects and across our outer attempts here. A slow mirror
    (e.g. a 2.4 GB file at ~25 kb/s) blows past ppx's built-in reconnect budget
    mid-file and raises ``ftplib.error_temp``; we just call again and it picks up
    where it left off. Only the first attempt honours ``force`` (a truncating
    ``wb+`` open) — every retry uses ``force_=False`` so we resume, not restart.

    We also catch ``ftplib.error_perm``: PRIDE intermittently answers the ``SIZE``
    probe (which ppx issues *outside* its own reconnect wrapper, ftp.py:147) with
    ``550 Could not get file size``. ppx's own ``_with_reconnects`` treats
    ``error_perm`` as retryable, so mirroring that here is consistent — a genuinely
    missing file just exhausts ``attempts`` and re-raises as before, only slower.
    """
    # ppx hardcodes max_reconnects=10 on its FTPParser; reach in and raise it.
    parser = getattr(project, "_parser", None)
    if parser is not None and hasattr(parser, "max_reconnects"):
        parser.max_reconnects = max_reconnects

    for attempt in range(1, attempts + 1):
        try:
            # ppx downloads into the project's `local` dir and skips size-matched
            # files, so completed files are no-ops on subsequent attempts.
            project.download(todo, force_=force and attempt == 1)
            return
        except (ftplib.error_temp, ftplib.error_perm, EOFError, OSError) as err:
            if attempt == attempts:
                raise
            print(
                f"WARNING: download attempt {attempt}/{attempts} failed "
                f"({type(err).__name__}: {err}); resuming ...",
                file=sys.stderr,
            )
            time.sleep(min(30, 2 ** attempt))


def stream_ingest(
    project,
    names: list[str],
    raw_dir: Path,
    mzml_dir: Path,
    manifest: Manifest,
    force: bool,
    jobs: int,
    prune_raw: bool,
    max_reconnects: int = 50,
    attempts: int = 20,
) -> list[Path]:
    """Download, convert, and (optionally) prune each file as a streaming pipeline.

    Files are downloaded one at a time (ppx uses a single FTP connection), but each
    raw is handed to a background pool of ``jobs`` converters the instant it lands —
    so the next download overlaps the running conversions. With ``prune_raw`` each
    raw is deleted the moment its mzML is verified on disk.

    The point of streaming (vs. download-all-then-convert-all) is failure and disk
    safety: if a download dies on file N, files 1..N-1 are already converted and
    pruned rather than lost, and raw disk usage stays bounded to the few in flight
    instead of holding the whole batch. Fully resumable via the manifest / mzML checks.
    """
    mzml_dir.mkdir(parents=True, exist_ok=True)
    results: list[Path] = []
    have: set[str] = set()

    def _add(path: Path) -> None:
        if path.name not in have:
            have.add(path.name)
            results.append(path)

    def _finish(name: str, out: Path) -> None:
        manifest.mark(name, converted=True, mzml=str(out))
        _add(out)
        if prune_raw and converted_mzml(name, mzml_dir, manifest):
            raw = raw_dir / os.path.basename(name)
            if raw.exists() and raw.suffix.lower() in RAW_SUFFIXES:
                raw.unlink()
                manifest.mark(name, raw_removed=True)

    def _drain(future) -> None:
        name = pending.pop(future)
        try:
            _finish(name, future.result())
        except Exception as err:  # a bad raw shouldn't sink the whole batch
            print(f"WARNING: conversion failed for {name}: {err}", file=sys.stderr)

    pending: dict = {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        try:
            for name in names:
                # 1) Already converted on an earlier run (raw may be pruned).
                existing = converted_mzml(name, mzml_dir, manifest) if not force else None
                if existing:
                    _add(existing)
                    continue

                local = raw_dir / os.path.basename(name)
                entry = manifest.get(name)

                # 2) Download unless the raw is already present on disk.
                if force or not (entry.get("downloaded") and local.exists()):
                    _download_with_resume(
                        project, [name], force, max_reconnects, attempts
                    )
                if not local.exists():
                    print(f"WARNING: expected download missing: {local}", file=sys.stderr)
                    continue
                manifest.mark(name, downloaded=True, size=local.stat().st_size)

                # 3) mzML that shipped directly: symlink into mzml_dir, no conversion.
                if local.suffix.lower() in MZML_SUFFIXES:
                    link = mzml_dir / local.name
                    if not (link.exists() or link.is_symlink()):
                        os.symlink(local.resolve(), link)
                    _add(link)
                    continue

                # 4) Raw: convert in the background so the next download overlaps it.
                pending[pool.submit(convert_one, local, mzml_dir)] = name

                # Reap finished conversions promptly so their raws get pruned now.
                for future in [f for f in pending if f.done()]:
                    _drain(future)
        finally:
            # Finalize every conversion we started, even if a download raised —
            # so partial progress is converted + pruned before we report failure.
            for future in as_completed(list(pending)):
                _drain(future)

    return results


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def converted_mzml(name: str, mzml_dir: Path, manifest: Manifest) -> Path | None:
    """Return the mzML for a remote file if it exists on disk, else ``None``.

    The check is at the *mzML level*: the output file must actually be present and
    non-empty, not merely flagged ``converted`` in the manifest. This is the safety
    gate for removing a raw — we only ever delete a .raw whose real output we can see.
    """
    entry = manifest.get(name)
    candidates = []
    if entry.get("mzml"):
        candidates.append(Path(entry["mzml"]))
    stem = os.path.splitext(os.path.basename(name))[0]
    candidates.append(mzml_dir / (stem + ".mzML"))
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


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
        "--reverse",
        action="store_true",
        help="Ingest from the end of the sorted file list (pair with a forward "
        "job to cover a large dataset from both ends concurrently)",
    )
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
    p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-operation FTP socket timeout in seconds (ppx default is 10)",
    )
    p.add_argument(
        "--max-reconnects",
        type=int,
        default=50,
        help="FTP reconnects ppx attempts mid-file before raising (ppx default is 10)",
    )
    p.add_argument(
        "--download-attempts",
        type=int,
        default=20,
        help="Outer resume attempts if a download still fails after all reconnects",
    )
    p.add_argument(
        "--prune-raw",
        action="store_true",
        help="After conversion, delete .raw files whose mzML is verified on disk",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    collection = Path(args.root) / args.accession
    raw_dir = collection / "raw"
    mzml_dir = collection / "mzml"
    manifest = Manifest.load(collection / "manifest.json")

    print(f"Resolving {args.accession} ...")
    project, remote = find_and_list(args.accession, raw_dir, args.timeout)
    print(f"{len(remote)} files available on ProteomeXchange.")

    selected = select_files(remote, args.glob, args.limit, args.reverse)

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

    # Stream each file through download -> convert -> (prune) so progress survives a
    # mid-batch download failure and raw disk usage stays bounded.
    mzml_files = stream_ingest(
        project,
        selected,
        raw_dir,
        mzml_dir,
        manifest,
        args.force,
        args.jobs,
        args.prune_raw,
        args.max_reconnects,
        args.download_attempts,
    )
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
