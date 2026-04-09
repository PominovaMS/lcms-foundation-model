"""Shared data loading, splitting, and DataLoader creation for eval experiments."""

import logging
import os

import polars as pl
import torch
from depthcharge.data import SpectrumDataset, spectra_to_df, preprocessing
from torch.utils.data import DataLoader

from source.dataset import LanceMapDataset, RunDataset

logger = logging.getLogger(__name__)


def load_metadata(meta_path: str) -> pl.DataFrame:
    """
    Load metadata CSV and normalise column names.
    Adds `peak_file` column (data_file + ".mzML").
    """
    meta_df = pl.read_csv(meta_path)
    meta_df = meta_df.rename({
        "characteristics[organism]": "organism",
        "comment[data file]": "data_file",
    })
    meta_df = meta_df.with_columns(
        (pl.col("data_file") + ".mzML").alias("peak_file")
    )
    return meta_df


def assign_splits(
    meta_df: pl.DataFrame,
    n_probe_genera: int = 15,
    min_species_per_genus: int = 2,
    n_ssl_top: int = 3,
) -> pl.DataFrame:
    """
    Deterministic split of files into SSL train and probe (train/val).

    Eligible genera (≥ min_species, not "food") are sorted by size (desc).
    - The top n_ssl_top largest → always SSL  (e.g. Pseudomonas, Staphylococcus, Bacillus)
    - The next n_probe_genera → probe (mid-sized, good for classification)
    - Everything else (remaining eligible + ineligible + "food") → SSL
    - Species within probe genera are sorted alphabetically and alternated
      between train (even index) and val (odd index).
    - genus_class is assigned 0..n-1 for probe genera (alphabetical by genus).
    - SSL files get genus_class = -1.

    The split is fully deterministic (no randomness). Ties in genus file count
    are broken alphabetically by genus name.
    """
    # --- count species & files per genus ---
    genus_stats = (
        meta_df
        .group_by("genus")
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
        probe_genus_summary.append((genus, len(genus_df), len(species_list), n_train, n_val))

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

    # --- log summary ---
    ssl_df = meta_df.filter(pl.col("split") == "train")
    ssl_genus_counts = (
        ssl_df.group_by("genus").agg(pl.len().alias("n"))
        .sort("n", descending=True)
    )
    top_ssl = ssl_genus_counts.head(5)["genus"].to_list()
    logger.info(
        f"SSL: {ssl_genus_counts.shape[0]} genera, {ssl_df.shape[0]} files "
        f"(top: {', '.join(top_ssl)}, ...)"
    )

    logger.info(f"Probe: {n_probe_genera} genera, "
                f"{meta_df.filter(pl.col('split') != 'train').shape[0]} files")
    for genus, n_files, n_sp, n_train, n_val in probe_genus_summary:
        cls = genus_to_class[genus]
        logger.info(
            f"  [{cls}] {genus:<25s} ({n_files:>3d} files, {n_sp:>2d} sp) "
            f"— train: {n_train} sp, val: {n_val} sp"
        )

    return meta_df


def load_mzml_data(data_dir: str, peak_files: list[str], max_num_peaks: int) -> dict:
    """
    Load mzML files from data_dir via depthcharge.
    Returns dict mapping filename → polars DataFrame of MS1 spectra.
    Logs any files listed in peak_files but missing from data_dir.
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

    logger.info(f"Loading {len(existing)}/{len(peak_files)} mzML files...")

    preprocessing_fn = [
        preprocessing.filter_intensity(max_num_peaks=max_num_peaks),
        preprocessing.scale_intensity(scaling="root", max_intensity=1.0),
    ]
    dfs = {}
    for peak_file in existing:
        dfs[peak_file] = spectra_to_df(
            os.path.join(data_dir, peak_file),
            metadata_df=None,
            ms_level=1,
            preprocessing_fn=preprocessing_fn,
            valid_charge=None,
            custom_fields=None,
            progress=True,
        )
    return dfs


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


def build_dataloaders(dfs: dict, meta_df: pl.DataFrame, config, n_ssl_files: int | None = None):
    """
    Build all four DataLoaders for an eval experiment.

    Filters meta_df to only files present in dfs (handles missing mzMLs).
    If n_ssl_files is set, caps the SSL training set to that many files
    (deterministic: sorted alphabetically by filename).

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
    def _make_ssl_dataset(split_names, max_files=None):
        files = meta_df.filter(pl.col("split").is_in(split_names))["peak_file"].to_list()
        if max_files is not None and len(files) > max_files:
            # Deterministic cap: sort alphabetically, take first N
            files = sorted(files)[:max_files]
            logger.info(f"SSL train capped to {max_files}/{len(meta_df.filter(pl.col('split').is_in(split_names)))} files")
        df = pl.concat([dfs[f] for f in files], how="vertical")
        df = df.join(meta_df, on="peak_file", how="left")
        stream = SpectrumDataset(
            df.select(["mz_array", "intensity_array", "genus_class"]),
            batch_size=256,
        )
        return LanceMapDataset(str(stream.path), seq_len=seq_len)

    train_dataset = _make_ssl_dataset(["train"], max_files=n_ssl_files)
    val_dataset = _make_ssl_dataset(["probe_train", "probe_val"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    # --- Probe datasets (run-level) ---
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

    return train_loader, val_loader, probe_train_loader, probe_val_loader
