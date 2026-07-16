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
def find_and_list(accession: str, raw_dir: Path, timeout: float = 60.0):
    """Resolve the project on ProteomeXchange and list its remote files.

    ``timeout`` is the per-socket-operation timeout (seconds) handed to ppx's
    FTP client. ppx defaults to 10s, which trips constantly on large files from
    slow PRIDE mirrors — bump it well up.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    project = ppx.find_project(accession, local=str(raw_dir), timeout=timeout)
    return project, project.remote_files()


def select_files(remote: list[str], glob: str | None, limit: int | None) -> list[str]:
    """Filter remote file names by glob and cap by limit, deterministically."""
    files = sorted(remote)
    if glob:
        files = [f for f in files if fnmatch.fnmatch(os.path.basename(f), glob)]
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
        except (ftplib.error_temp, EOFError, OSError) as err:
            if attempt == attempts:
                raise
            print(
                f"WARNING: download attempt {attempt}/{attempts} failed "
                f"({type(err).__name__}: {err}); resuming ...",
                file=sys.stderr,
            )
            time.sleep(min(30, 2 ** attempt))


def download_files(
    project,
    names: list[str],
    raw_dir: Path,
    mzml_dir: Path,
    manifest: Manifest,
    force: bool,
    max_reconnects: int = 50,
    attempts: int = 20,
) -> list[Path]:
    """Download the selected files, skipping those already present (ppx + manifest).

    Files whose mzML already exists on disk are skipped too: their raw is no longer
    needed, so a run after ``--prune-raw`` never re-downloads a raw we deleted on purpose.
    """
    todo = []
    for name in names:
        local = raw_dir / os.path.basename(name)
        entry = manifest.get(name)
        if not force and converted_mzml(name, mzml_dir, manifest):
            continue
        if not force and entry.get("downloaded") and local.exists():
            continue
        todo.append(name)

    if todo:
        _download_with_resume(project, todo, force, max_reconnects, attempts)

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


def prune_raws(
    raw_dir: Path, mzml_dir: Path, manifest: Manifest, names: list[str]
) -> int:
    """Delete .raw files whose converted mzML is verified on disk. Returns count removed.

    Never touches a raw without a real mzML output (see :func:`converted_mzml`), so a
    failed or partial conversion always keeps its source. Idempotent: a raw already gone
    is simply skipped. The manifest records ``raw_removed`` so re-runs know why it's absent.
    """
    removed = 0
    for name in names:
        if not converted_mzml(name, mzml_dir, manifest):
            continue  # no verified mzML -> keep the raw
        raw = raw_dir / os.path.basename(name)
        if raw.exists() and raw.suffix.lower() in RAW_SUFFIXES:
            raw.unlink()
            manifest.mark(name, raw_removed=True)
            removed += 1
    return removed


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

    downloaded = download_files(
        project,
        selected,
        raw_dir,
        mzml_dir,
        manifest,
        args.force,
        args.max_reconnects,
        args.download_attempts,
    )
    print(f"Downloaded/present: {len(downloaded)} raw/mzML files in {raw_dir}")

    mzml_dir.mkdir(parents=True, exist_ok=True)
    mzml_files: list[Path] = []
    have: set[str] = set()

    def _add(path: Path) -> None:
        if path.name not in have:
            mzml_files.append(path)
            have.add(path.name)

    # 1) Convert freshly-downloaded .raw files (writes into mzml_dir).
    raws = [f for f in downloaded if f.suffix.lower() in RAW_SUFFIXES]
    for out in convert_all(raws, mzml_dir, args.jobs, manifest, args.force):
        _add(out)

    # 2) Mirror any mzML that shipped directly: it downloads into raw_dir, so
    #    symlink it into mzml_dir to keep mzml_dir the single source of truth.
    for f in downloaded:
        if f.suffix.lower() in MZML_SUFFIXES:
            link = mzml_dir / f.name
            if not (link.exists() or link.is_symlink()):
                os.symlink(f.resolve(), link)
            _add(link)

    # 3) Re-attach files converted on an earlier run whose raw may since be pruned
    #    (they never appear in `downloaded`, but their mzML still lives in mzml_dir).
    for name in selected:
        existing = converted_mzml(name, mzml_dir, manifest)
        if existing:
            _add(existing)

    print(f"mzML ready: {len(mzml_files)} files in {mzml_dir}")

    if args.link_into:
        n_train, n_val = link_into_training(
            mzml_files, Path(args.link_into), args.val_frac, args.seed, args.force
        )
        print(f"Linked into {args.link_into}: {n_train} train, {n_val} val")

    if args.prune_raw:
        removed = prune_raws(raw_dir, mzml_dir, manifest, selected)
        print(f"Pruned {removed} raw files (converted mzML verified on disk).")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
