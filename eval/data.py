"""Shared data loading, splitting, and DataLoader creation for eval experiments."""

import json
import logging
import os

import numpy as np
import polars as pl
import torch
from depthcharge.data import spectra_to_df, preprocessing
from torch.utils.data import DataLoader

from source.dataset import RunDataset

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


def _log_class_composition(meta_df: pl.DataFrame, label: str, cls: int) -> None:
    """Log one class's train/val file counts, broken down by species.

    The per-species breakdown is the point: a genus can be one heavily replicated
    species or forty species with three files each, and the two make ``val_acc``
    mean very different things. Nothing in the repo records this, so the split log
    is where it surfaces.
    """
    rows = meta_df.filter(pl.col("split").is_in(["probe_train", "probe_val"]))
    n_train = rows.filter(pl.col("split") == "probe_train").height
    n_val = rows.filter(pl.col("split") == "probe_val").height
    species = sorted(rows["organism"].unique().to_list())
    logger.info(
        f"  [{cls}] {label:<25s} {n_train + n_val:>4d} files "
        f"({n_train} train / {n_val} val), {len(species)} species"
    )
    for sp in species:
        sp_rows = rows.filter(pl.col("organism") == sp)
        logger.info(
            f"        {sp:<38s} "
            f"{sp_rows.filter(pl.col('split') == 'probe_train').height:>3d} train / "
            f"{sp_rows.filter(pl.col('split') == 'probe_val').height:>3d} val"
        )


def _log_label_counts(
    counts: pl.DataFrame,
    selected: list[str],
    label_col: str,
    n_classes: int,
    label_offset: int,
    top: int = 10,
) -> None:
    """Log the largest labels and mark the selected ones.

    The runner-up matters: it is the input to ``--label_offset``, and at
    ``label_col="organism"`` these per-species file counts are written down
    nowhere else in the repo.
    """
    shown = min(top, counts.height)
    scope = (
        f"top {shown} of {counts.height}" if shown < counts.height else f"all {shown}"
    )
    logger.info(
        f"Label counts ({label_col}), {scope} — * = selected "
        f"(n_classes={n_classes}, label_offset={label_offset}):"
    )
    for label, n_files in counts.head(top).iter_rows():
        mark = "*" if label in selected else " "
        logger.info(f"  {mark} {label:<38s} {n_files:>4d}")
    if counts.height > top:
        logger.info(f"    ... and {counts.height - top} more")


def assign_splits(
    meta_df: pl.DataFrame,
    n_classes: int = 2,
    label_col: str = "genus",
    val_frac: float = 0.3,
    label_offset: int = 0,
) -> pl.DataFrame:
    """Deterministic probe split over the ``n_classes`` most abundant labels.

    Takes the ``n_classes`` labels with the most mzML files (ties broken
    alphabetically) and splits **each label's own files** into probe train/val in
    ``val_frac`` proportion. Files of every other label are marked ``"unused"`` and
    read by nothing.

    ``label_col`` is ``"genus"`` (the default) or ``"organism"`` for species.
    ``genus == "food"`` is always excluded: it is a sample type, not an organism.

    ``label_offset`` skips that many labels from the top before selecting, so
    ``label_offset=1, n_classes=2`` probes the second- and third-largest labels.
    The default 0 is the plain top-N. Use it when the largest label dwarfs the
    rest and ``majority_acc`` is uncomfortably high — at the cost of fewer files,
    so a coarser ``val_acc``. The selection is recorded in the results CSV's
    ``class_names``, so the offset itself needs no column.

    What this deliberately does NOT do
    ----------------------------------
    Nothing is reserved for SSL pretraining — that runs on PRIDE, and abele is
    purely a downstream evaluation set. There is no ``"train"`` split.

    Classes are not balanced or capped. On abele the two largest genera are
    Pseudomonas (312 files) and Staphylococcus (136), so ``majority_acc`` is ~0.70
    rather than ``random_acc``'s 0.50 — read ``val_acc_macro`` and
    ``n_pred_classes`` alongside ``val_acc``. Because the split is proportional,
    train and val carry the same class distribution, which makes ``majority_acc``
    a well-behaved collapse threshold.

    Train and val share species by construction, so this measures "same species,
    unseen run", not cross-species generalisation. That is the intent — it is a
    lower-variance sanity check on whether the encoder carries organism
    information at all — but it is why these numbers are not comparable with those
    from the older species-alternating split.

    Returns ``meta_df`` plus ``split`` (``"probe_train"`` / ``"probe_val"`` /
    ``"unused"``) and ``label_class`` (``0..n_classes-1`` for selected labels,
    ``-1`` otherwise). Fully deterministic: no randomness anywhere.
    """
    eligible = meta_df.filter(pl.col("genus") != "food")

    counts = (
        eligible.group_by(label_col)
        .agg(pl.len().alias("n_files"))
        .sort(["n_files", label_col], descending=[True, False])
    )
    if counts.height < label_offset + n_classes:
        raise ValueError(
            f"Only {counts.height} {label_col} values available, need "
            f"{label_offset + n_classes} (n_classes={n_classes} at "
            f"label_offset={label_offset}): {sorted(counts[label_col].to_list())}"
        )

    # Alphabetical class indices, so they do not shift when file counts do.
    selected = sorted(counts.slice(label_offset, n_classes)[label_col].to_list())
    _log_label_counts(counts, selected, label_col, n_classes, label_offset)
    label_to_class = {label: i for i, label in enumerate(selected)}

    # --- proportional train/val split inside each selected label ---
    val_files: set[str] = set()
    for label in selected:
        files = sorted(
            eligible.filter(pl.col(label_col) == label)["peak_file"].to_list()
        )
        n = len(files)
        if n < 2:
            # A one-file class cannot have both sides; leave it in train so the
            # probe at least sees the class. (Falling through with k=0 would put it
            # in val instead — stride(n, 0) means "no cap", i.e. every index.)
            continue
        # Clamped so a small class still yields at least one file on each side.
        k = min(max(1, round(n * val_frac)), n - 1)
        # stride(), not a prefix: val samples across the acquisition order instead
        # of taking whichever contiguous batch happens to sort first.
        val_files.update(files[i] for i in stride(n, k))

    meta_df = meta_df.with_columns(
        pl.col(label_col)
        .replace_strict(label_to_class, default=-1, return_dtype=pl.Int64)
        .alias("label_class")
    )
    meta_df = meta_df.with_columns(
        pl.when(pl.col("label_class") < 0)
        .then(pl.lit("unused"))
        .when(pl.col("peak_file").is_in(list(val_files)))
        .then(pl.lit("probe_val"))
        .otherwise(pl.lit("probe_train"))
        .alias("split")
    )

    n_train = meta_df.filter(pl.col("split") == "probe_train").height
    n_val = meta_df.filter(pl.col("split") == "probe_val").height
    logger.info(
        f"Probe: {n_classes} {label_col} classes (label_offset={label_offset}), "
        f"{n_train + n_val} files ({n_train} train / {n_val} val); "
        f"{meta_df.filter(pl.col('split') == 'unused').height} files unused"
    )
    for label in selected:
        cls = label_to_class[label]
        _log_class_composition(
            meta_df.filter(pl.col(label_col) == label), label, cls
        )
        n_cls_val = meta_df.filter(
            (pl.col(label_col) == label) & (pl.col("split") == "probe_val")
        ).height
        if n_cls_val < 3:
            logger.warning(
                f"Class [{cls}] {label} has only {n_cls_val} val file(s); val "
                f"accuracy is coarsely quantised at that size."
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

    run_labels = dict(zip(meta_df["peak_file"], meta_df["label_class"]))

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
