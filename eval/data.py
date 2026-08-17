"""Shared data loading, splitting, and DataLoader creation for eval experiments."""

import json
import logging
import os

import numpy as np
import polars as pl
import torch
from depthcharge.data import SpectrumDataset, spectra_to_df, preprocessing
from torch.utils.data import DataLoader

from source.dataset import LanceMapDataset, RunDataset

logger = logging.getLogger(__name__)


def stride(n: int, k: int) -> list[int]:
    """``k`` indices evenly spaced across ``range(n)`` (deterministic, no RNG).

    Mirrors ``scripts/mass_dist.py::stride`` (kept local so ``eval/`` does not
    depend on ``scripts/``).
    """
    if k <= 0 or n <= k:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))


def load_metadata(meta_path: str) -> pl.DataFrame:
    """
    Load metadata CSV and normalise column names.
    Adds `peak_file` column (data_file + ".mzML").
    """
    meta_df = pl.read_csv(meta_path)
    meta_df = meta_df.rename(
        {
            "characteristics[organism]": "organism",
            "comment[data file]": "data_file",
        }
    )
    meta_df = meta_df.with_columns((pl.col("data_file") + ".mzML").alias("peak_file"))
    return meta_df


def _split_counts(meta_df: pl.DataFrame, split: str) -> dict[str, int]:
    """Number of files per genus in one split."""
    counts = meta_df.filter(pl.col("split") == split).group_by("genus").agg(
        pl.len().alias("n")
    )
    return dict(zip(counts["genus"].to_list(), counts["n"].to_list()))


def _cap_probe_files(
    meta_df: pl.DataFrame, group_cols: list[str], cap: int
) -> pl.DataFrame:
    """Mark all but ``cap`` evenly-strided files per group as ``split="unused"``.

    Only rows already in a probe split are considered — the SSL ("train") split is
    never capped. Files are ordered by ``peak_file`` and sampled with :func:`stride`
    rather than truncated, so the kept files spread across the acquisition order
    instead of clustering in whatever batch happens to sort first.
    """
    probe_splits = ["probe_train", "probe_val"]
    probe = meta_df.filter(pl.col("split").is_in(probe_splits))

    groups: dict[tuple, list[str]] = {}
    for row in probe.select([*group_cols, "peak_file"]).iter_rows():
        groups.setdefault(row[:-1], []).append(row[-1])

    keep: set[str] = set()
    for files in groups.values():
        files = sorted(files)
        keep.update(files[i] for i in stride(len(files), cap))

    return meta_df.with_columns(
        pl.when(
            pl.col("split").is_in(probe_splits)
            & ~pl.col("peak_file").is_in(list(keep))
        )
        .then(pl.lit("unused"))
        .otherwise(pl.col("split"))
        .alias("split")
    )


def assign_splits(
    meta_df: pl.DataFrame,
    n_probe_genera: int = 15,
    min_species_per_genus: int = 2,
    n_ssl_top: int = 3,
    probe_all: bool = False,
    max_files_per_species: int | None = 3,
    max_files_per_genus: int | None = None,
) -> pl.DataFrame:
    """
    Deterministic split of files into SSL train and probe (train/val).

    With ``probe_all=True`` the probe covers ALL eligible genera (nothing is
    reserved for SSL); ``n_probe_genera`` and ``n_ssl_top`` are then ignored. Use
    this when the encoder is pretrained elsewhere (e.g. on PRIDE), so there is no
    need to hold abele genera out for co-trained SSL.

    Eligible genera (≥ min_species, not "food") are sorted by size (desc).
    - The top n_ssl_top largest → always SSL  (e.g. Pseudomonas, Staphylococcus, Bacillus)
    - The next n_probe_genera → probe (mid-sized, good for classification)
    - Everything else (remaining eligible + ineligible + "food") → SSL
    - Species within probe genera are sorted alphabetically and alternated
      between train (even index) and val (odd index).
    - genus_class is assigned 0..n-1 for probe genera (alphabetical by genus).
    - SSL files get genus_class = -1.

    Probe class balance
    -------------------
    The species alternation above implicitly assumes species carry comparable
    numbers of files. In abele they do not: 80 of the 87 probe species have 3
    files, but *Escherichia coli* has 48 — and being alphabetically first it always
    lands in probe_train. Uncapped, that single species is 25% of probe_train but
    only 2.3% of probe_val, so a probe that collapses onto it scores *below* the
    random rate. ``max_files_per_species`` (default 3, the modal count) caps each
    species within each probe split, which drops the train-vs-val total variation
    from 0.232 to 0.082. ``max_files_per_genus`` additionally caps each class, for
    an exactly balanced probe at the cost of roughly half the files. Set either to
    ``None`` or ``0`` to disable. Capping applies to the probe splits ONLY — the
    SSL split keeps every file. Files dropped by a cap get ``split = "unused"``
    (never ``"train"``, which would leak probe genera into the SSL corpus).

    The split is fully deterministic (no randomness). Ties in genus file count
    are broken alphabetically by genus name.
    """
    # --- count species & files per genus ---
    genus_stats = (
        meta_df.group_by("genus")
        .agg(
            pl.col("organism").n_unique().alias("n_species"),
            pl.len().alias("n_files"),
        )
        .sort(["n_files", "genus"], descending=[True, False])
    )

    # --- eligible = ≥ min_species AND not "food" ---
    eligible = genus_stats.filter(
        (pl.col("n_species") >= min_species_per_genus) & (pl.col("genus") != "food")
    )
    n_eligible = len(eligible)

    if probe_all:
        # Probe on every eligible genus; reserve nothing for SSL.
        ssl_top = set()
        probe_genera = eligible["genus"].to_list()
        ssl_remaining = set()
        n_probe_genera = len(probe_genera)
        if n_probe_genera == 0:
            raise ValueError(f"No eligible genera to probe ({n_eligible} eligible).")
    else:
        # Clamp n_ssl_top and n_probe_genera to available eligible genera
        n_ssl_top = min(n_ssl_top, n_eligible)
        n_probe_genera = min(n_probe_genera, n_eligible - n_ssl_top)
        if n_probe_genera <= 0:
            raise ValueError(
                f"No genera left for probe: {n_eligible} eligible, {n_ssl_top} reserved for SSL top."
            )

        # Top n_ssl_top → SSL, next n_probe_genera → probe, rest → SSL
        ssl_top = set(eligible.head(n_ssl_top)["genus"].to_list())
        probe_genera = eligible.slice(n_ssl_top, n_probe_genera)["genus"].to_list()
        ssl_remaining = set(eligible.slice(n_ssl_top + n_probe_genera)["genus"].to_list())

    # Ineligible genera (< min_species, or "food") → always SSL
    ineligible = genus_stats.filter(
        (pl.col("n_species") < min_species_per_genus) | (pl.col("genus") == "food")
    )
    ssl_genera_ineligible = set(ineligible["genus"].to_list())

    all_ssl_genera = ssl_top | ssl_remaining | ssl_genera_ineligible

    # --- assign genus_class (0..n-1 for probe, alphabetical by genus name) ---
    probe_genera_sorted = sorted(probe_genera)
    genus_to_class = {g: i for i, g in enumerate(probe_genera_sorted)}

    # --- assign splits at the species level within probe genera ---
    # For each probe genus, sort species alphabetically and alternate train/val
    species_to_split = {}
    probe_genus_summary = []

    for genus in probe_genera_sorted:
        genus_df = meta_df.filter(pl.col("genus") == genus)
        species_list = sorted(genus_df["organism"].unique().to_list())
        n_train, n_val = 0, 0
        for idx, species in enumerate(species_list):
            if idx % 2 == 0:
                species_to_split[species] = "probe_train"
                n_train += 1
            else:
                species_to_split[species] = "probe_val"
                n_val += 1
        probe_genus_summary.append(
            (genus, len(genus_df), len(species_list), n_train, n_val)
        )

    # --- build split + genus_class columns ---
    def _get_split(row):
        genus = row["genus"]
        organism = row["organism"]
        if genus in all_ssl_genera:
            return "train"
        return species_to_split.get(organism, "train")

    def _get_genus_class(genus):
        return genus_to_class.get(genus, -1)

    meta_df = meta_df.with_columns(
        pl.struct(["genus", "organism"])
        .map_elements(_get_split, return_dtype=pl.Utf8)
        .alias("split")
    )
    meta_df = meta_df.with_columns(
        pl.col("genus")
        .map_elements(_get_genus_class, return_dtype=pl.Int64)
        .alias("genus_class")
    )

    # --- cap files per species / per genus within the probe splits ---
    n_probe_before = meta_df.filter(
        pl.col("split").is_in(["probe_train", "probe_val"])
    ).height
    if max_files_per_species:
        meta_df = _cap_probe_files(meta_df, ["split", "organism"], max_files_per_species)
    if max_files_per_genus:
        meta_df = _cap_probe_files(meta_df, ["split", "genus"], max_files_per_genus)

    # --- log summary ---
    ssl_df = meta_df.filter(pl.col("split") == "train")
    ssl_genus_counts = (
        ssl_df.group_by("genus").agg(pl.len().alias("n")).sort("n", descending=True)
    )
    top_ssl = ssl_genus_counts.head(5)["genus"].to_list()
    logger.info(
        f"SSL: {ssl_genus_counts.shape[0]} genera, {ssl_df.shape[0]} files "
        f"(top: {', '.join(top_ssl)}, ...)"
    )

    # Per-genus file counts AFTER capping, so the log reflects what the probe sees.
    n_train_files = _split_counts(meta_df, "probe_train")
    n_val_files = _split_counts(meta_df, "probe_val")
    n_probe_files = sum(n_train_files.values()) + sum(n_val_files.values())
    n_dropped = n_probe_before - n_probe_files
    logger.info(
        f"Probe: {n_probe_genera} genera, {n_probe_files} files "
        f"({sum(n_train_files.values())} train / {sum(n_val_files.values())} val)"
    )
    for genus, _n_files, n_sp, n_train, n_val in probe_genus_summary:
        cls = genus_to_class[genus]
        logger.info(
            f"  [{cls}] {genus:<25s} ({n_sp:>2d} sp: {n_train} train / {n_val} val) "
            f"— files: {n_train_files.get(genus, 0):>3d} train / "
            f"{n_val_files.get(genus, 0):>3d} val"
        )
    if n_dropped:
        logger.info(
            f"Capped {n_dropped}/{n_probe_before} probe files to 'unused' "
            f"(max_files_per_species={max_files_per_species}, "
            f"max_files_per_genus={max_files_per_genus}) to keep the probe "
            f"train/val class distributions comparable."
        )

    return meta_df


def _open_mzml_cache(cache_dir: str, data_dir: str, max_num_peaks: int) -> None:
    """Create ``cache_dir`` if needed and validate its ``cache_info.json``.

    Parsing bakes the preprocessing in, and cache entries are keyed by mzML
    basename, so a cache dir is only valid for ONE (data_dir, max_num_peaks)
    pair. Reusing it across either would serve spectra that look fine and are
    not what the config asked for — hence a hard refusal rather than a warning.
    Mirrors the ``dataset_info.json`` check ``train.py --lance_dir`` makes.
    """
    os.makedirs(cache_dir, exist_ok=True)
    info_path = os.path.join(cache_dir, "cache_info.json")
    info = {
        "data_dir": os.path.realpath(data_dir),
        "max_num_peaks": max_num_peaks,
        "intensity_scaling": "root",
        "ms_level": 1,
    }

    if not os.path.exists(info_path):
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        return

    with open(info_path) as f:
        recorded = json.load(f)

    for key in ("data_dir", "max_num_peaks", "intensity_scaling", "ms_level"):
        if recorded.get(key) != info[key]:
            raise SystemExit(
                f"mzML cache {cache_dir} was built with {key}="
                f"{recorded.get(key)!r}, but this run wants {info[key]!r}. "
                f"Use a separate cache dir for it (one dir per dataset / peak "
                f"cap), or delete {cache_dir} to rebuild."
            )


def load_mzml_data(
    data_dir: str,
    peak_files: list[str],
    max_num_peaks: int,
    cache_dir: str | None = None,
) -> dict:
    """
    Load mzML files from data_dir via depthcharge.
    Returns dict mapping filename → polars DataFrame of MS1 spectra.
    Logs any files listed in peak_files but missing from data_dir.

    With ``cache_dir`` set, each parsed file is persisted as
    ``<cache_dir>/<peak_file>.parquet`` and reloaded from there next time —
    the XML parse dominates the runtime of an eval, and its output depends only
    on (file, max_num_peaks, the preprocessing below). The cache is per file, so
    changing the split settings reuses everything already parsed and only reads
    the newly-included files.
    """
    existing = []
    missing = []
    for f in peak_files:
        if os.path.exists(os.path.join(data_dir, f)):
            existing.append(f)
        else:
            missing.append(f)

    if missing:
        logger.warning(
            f"{len(missing)}/{len(peak_files)} mzML files not found in {data_dir}:"
        )
        for f in missing:
            logger.warning(f"  missing: {f}")

    if cache_dir:
        _open_mzml_cache(cache_dir, data_dir, max_num_peaks)
        logger.info(f"mzML cache: {cache_dir}")
    logger.info(f"Loading {len(existing)}/{len(peak_files)} mzML files...")

    preprocessing_fn = [
        preprocessing.filter_intensity(max_num_peaks=max_num_peaks),
        preprocessing.scale_intensity(scaling="root", max_intensity=1.0),
    ]
    dfs = {}
    n_cached = 0
    for peak_file in existing:
        cache_path = (
            os.path.join(cache_dir, os.path.basename(peak_file) + ".parquet")
            if cache_dir
            else None
        )
        if cache_path and os.path.exists(cache_path):
            dfs[peak_file] = pl.read_parquet(cache_path)
            n_cached += 1
            continue

        df = spectra_to_df(
            os.path.join(data_dir, peak_file),
            metadata_df=None,
            ms_level=1,
            preprocessing_fn=preprocessing_fn,
            valid_charge=None,
            custom_fields=None,
            progress=True,
        )
        dfs[peak_file] = df

        if cache_path:
            # Write-then-rename: a killed job must not leave a truncated parquet
            # that a later run reads as valid, and an --array submit has several
            # tasks writing the same entries at once (identical bytes, so the
            # last rename winning is fine — but a shared tmp path is not).
            tmp_path = f"{cache_path}.{os.getpid()}.tmp"
            df.write_parquet(tmp_path)
            os.replace(tmp_path, cache_path)

    if cache_dir:
        logger.info(
            f"Loaded {len(dfs)} files — {n_cached} from cache, "
            f"{len(dfs) - n_cached} parsed"
        )
    return dfs


def get_needed_files(
    meta_df: pl.DataFrame, data_dir: str, n_ssl_files: int | None = None
) -> list[str]:
    """
    Return only the peak_files that will actually be used, so only these
    need to be loaded from disk.

    1. Checks which metadata files exist in data_dir (logs missing ones).
    2. Selects all existing probe files (always needed).
    3. Selects existing SSL train files, capped to n_ssl_files if set
       (deterministic: sorted alphabetically).
    4. Returns the union.
    """
    all_files = meta_df["peak_file"].to_list()
    existing = set()
    missing = []
    for f in all_files:
        if os.path.exists(os.path.join(data_dir, f)):
            existing.add(f)
        else:
            missing.append(f)

    if missing:
        logger.warning(
            f"{len(missing)}/{len(all_files)} mzML files not found in {data_dir}:"
        )
        for f in missing:
            logger.warning(f"  missing: {f}")

    # Probe files — always needed (skip missing)
    probe_files = [
        f
        for f in meta_df.filter(pl.col("split").is_in(["probe_train", "probe_val"]))[
            "peak_file"
        ].to_list()
        if f in existing
    ]

    # SSL train files — cap after filtering to existing
    ssl_files = [
        f
        for f in meta_df.filter(pl.col("split") == "train")["peak_file"].to_list()
        if f in existing
    ]
    if n_ssl_files is not None and len(ssl_files) > n_ssl_files:
        ssl_files = sorted(ssl_files)[:n_ssl_files]
        logger.info(
            f"SSL train capped to {n_ssl_files} files "
            f"(of {len(meta_df.filter(pl.col('split') == 'train'))} in metadata)"
        )

    needed = sorted(set(probe_files) | set(ssl_files))
    logger.info(
        f"Files to load: {len(needed)} "
        f"({len(ssl_files)} SSL train + {len(probe_files)} probe)"
    )
    return needed


def run_collate_fn(rows):
    """Collate function for RunDataset: keeps mz/intensity as lists of tensors."""
    keys = rows[0].keys()
    batch = {}
    for key in keys:
        if key in ("mz_array", "intensity_array"):
            batch[key] = [torch.tensor(r[key]) for r in rows]
        else:
            batch[key] = torch.tensor([r[key] for r in rows])
    return batch


def build_probe_dataloaders(dfs: dict, meta_df: pl.DataFrame, config):
    """
    Build only the run-level probe DataLoaders (no SSL datasets).

    Use this for evaluating an already-pretrained checkpoint: it skips the
    spectrum-level SSL datasets (and their temporary Lance DB), so only the
    probe_train / probe_val runs need to be loaded from disk.

    Filters meta_df to only files present in dfs.

    Returns:
        probe_train_loader – run-level probe training (shuffled)
        probe_val_loader   – run-level probe evaluation
    """
    batch_size = config.data.batch_size
    seq_len = config.data.max_num_peaks

    # filter to files that were actually loaded
    loaded_files = list(dfs.keys())
    meta_df = meta_df.filter(pl.col("peak_file").is_in(loaded_files))

    run_labels = dict(zip(meta_df["peak_file"], meta_df["genus_class"]))

    def _make_probe_dataset(split_name):
        files = meta_df.filter(pl.col("split") == split_name)["peak_file"].to_list()
        return RunDataset(
            [dfs[f] for f in files],
            run_labels=run_labels,
            seq_len=seq_len,
        )

    probe_train_dataset = _make_probe_dataset("probe_train")
    probe_val_dataset = _make_probe_dataset("probe_val")

    probe_train_loader = DataLoader(
        probe_train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True,
        collate_fn=run_collate_fn,
    )
    probe_val_loader = DataLoader(
        probe_val_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        collate_fn=run_collate_fn,
    )

    return probe_train_loader, probe_val_loader


def build_dataloaders(dfs: dict, meta_df: pl.DataFrame, config):
    """
    Build all four DataLoaders for an eval experiment.

    Filters meta_df to only files present in dfs. File selection and SSL
    capping should be done upstream via get_needed_files().

    Returns:
        train_loader       – SSL pretraining (spectra-level, shuffled)
        val_loader         – SSL validation (spectra-level, not shuffled)
        probe_train_loader – run-level probe training (shuffled)
        probe_val_loader   – run-level probe evaluation
    """
    batch_size = config.data.batch_size
    seq_len = config.data.max_num_peaks

    # filter to files that were actually loaded
    loaded_files = list(dfs.keys())
    meta_df = meta_df.filter(pl.col("peak_file").is_in(loaded_files))

    # --- SSL datasets (spectrum-level, stored in Lance) ---
    def _make_ssl_dataset(split_names):
        files = meta_df.filter(pl.col("split").is_in(split_names))[
            "peak_file"
        ].to_list()
        df = pl.concat([dfs[f] for f in files], how="vertical")
        df = df.join(meta_df, on="peak_file", how="left")
        stream = SpectrumDataset(
            df.select(["mz_array", "intensity_array", "genus_class"]),
            batch_size=256,
        )
        dataset = LanceMapDataset(str(stream.path), seq_len=seq_len)
        # Prevent the SpectrumDataset (and its temp Lance DB) from being
        # garbage-collected while the LanceMapDataset still needs the files.
        dataset._spectrum_dataset_ref = stream
        return dataset

    train_dataset = _make_ssl_dataset(["train"])
    val_dataset = _make_ssl_dataset(["probe_train", "probe_val"])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=0, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=0, shuffle=False
    )

    # --- Probe datasets (run-level) ---
    probe_train_loader, probe_val_loader = build_probe_dataloaders(
        dfs, meta_df, config
    )

    return train_loader, val_loader, probe_train_loader, probe_val_loader
