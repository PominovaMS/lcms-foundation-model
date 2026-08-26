"""t-SNE visualization of run-level embeddings from a trained MS1Encoder.

Loads a checkpoint, computes CLS embeddings for all spectra in each run
(from parquet files), mean-pools to get run-level embeddings, and plots
a t-SNE scatter colored by genus.

Usage:
    # First run (computes embeddings and caches to .npz):
    python eval/tsne_embeddings.py \\
        --ckpt_path /path/to/epoch=42-step=37324.ckpt \\
        --data_dir /mnt/data/shared/lc_ms_foundation/abele_data/parquet \\
        --meta_path /mnt/data/shared/lc_ms_foundation/abele_data/filtered_abele_metadata.csv \\
        --embeddings_path /path/to/run_embeddings.npz \\
        --output tsne_genus.png

    # Re-plot from cache with different coloring:
    python eval/tsne_embeddings.py \\
        --embeddings_path /path/to/run_embeddings.npz \\
        --color_by suffix \\
        --output tsne_suffix.png

Options for --color_by:
    genus   - color by genus from metadata (default)
    suffix  - color by filename suffix (_10, _30, or other)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from tqdm import tqdm

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from source.model import MS1Encoder
from eval.data import load_metadata

logger = logging.getLogger(__name__)


def load_parquet_runs(data_dir: str) -> dict[str, pl.DataFrame]:
    """Load all parquet files from data_dir. Returns {filename: DataFrame}."""
    parquet_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
    dfs = {}
    for pf in tqdm(parquet_files, desc="Loading parquet files"):
        dfs[pf] = pl.read_parquet(os.path.join(data_dir, pf))
    return dfs


def build_run_genus_map(
    parquet_filenames: list[str], meta_df: pl.DataFrame
) -> tuple[dict[str, str], list[str]]:
    """Map parquet filename -> genus label via data_file join.

    Parquet filenames are like 'BBM_334_P110_25_DDA_001_R2.parquet'.
    Metadata data_file is like 'BBM_437_P110_31_MIA_026' (no extension).
    Match by stripping .parquet from filename and looking up data_file.

    Returns (mapping dict, list of unmatched filenames).
    """
    data_file_to_genus = dict(
        zip(meta_df["data_file"].to_list(), meta_df["genus"].to_list())
    )
    mapping = {}
    missing = []
    for pf in parquet_filenames:
        data_file = pf.removesuffix(".parquet")
        if data_file in data_file_to_genus:
            mapping[pf] = data_file_to_genus[data_file]
        else:
            missing.append(pf)
    return mapping, missing


def pad_or_truncate(seq, seq_len: int) -> np.ndarray:
    """Pad or truncate a 1D array to seq_len."""
    arr = np.array(seq, dtype=np.float32)
    if len(arr) < seq_len:
        return np.pad(arr, (0, seq_len - len(arr)))
    return arr[:seq_len]


def compute_run_embedding(
    model: MS1Encoder,
    run_df: pl.DataFrame,
    seq_len: int,
    batch_size: int,
) -> np.ndarray:
    """Compute run-level embedding by mean-pooling CLS embeddings.

    Follows the pattern from eval/callbacks.py _encode_run().
    Batches the forward pass to limit CPU memory.
    """
    mz_arrays = run_df["mz_array"].to_numpy()
    intensity_arrays = run_df["intensity_array"].to_numpy()

    mz_padded = np.stack([pad_or_truncate(s, seq_len) for s in mz_arrays])
    int_padded = np.stack([pad_or_truncate(s, seq_len) for s in intensity_arrays])

    n_spectra = len(mz_padded)
    all_cls_embs = []

    for start in range(0, n_spectra, batch_size):
        end = min(start + batch_size, n_spectra)
        mz_batch = torch.tensor(mz_padded[start:end])
        int_batch = torch.tensor(int_padded[start:end])

        with torch.no_grad():
            cls_emb, _ = model.forward(mz_batch, int_batch)
        all_cls_embs.append(cls_emb)

    all_cls = torch.cat(all_cls_embs, dim=0)  # (n_spectra, d_model)
    run_emb = all_cls.mean(dim=0)  # (d_model,)
    return run_emb.numpy()


def compute_all_embeddings(
    model: MS1Encoder,
    dfs: dict[str, pl.DataFrame],
    run_genus_map: dict[str, str],
    seq_len: int,
    batch_size: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Compute embeddings for all runs that have genus labels.

    Returns (embeddings array, genus list, filename list).
    """
    embeddings = []
    genera = []
    filenames = []

    for parquet_file, genus in tqdm(
        run_genus_map.items(), desc="Computing run embeddings"
    ):
        if parquet_file not in dfs:
            continue
        run_df = dfs[parquet_file]
        if len(run_df) == 0:
            continue

        emb = compute_run_embedding(model, run_df, seq_len, batch_size)
        embeddings.append(emb)
        genera.append(genus)
        filenames.append(parquet_file)

    return np.stack(embeddings), genera, filenames


def run_tsne(
    embeddings: np.ndarray, perplexity: float = 30.0, random_state: int = 42
) -> np.ndarray:
    """Run t-SNE, auto-clamping perplexity if needed."""
    n = embeddings.shape[0]
    effective_perplexity = min(perplexity, max(1.0, n - 1))
    if effective_perplexity != perplexity:
        logger.warning(
            f"Clamped perplexity from {perplexity} to {effective_perplexity} "
            f"(n_samples={n})"
        )

    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
    )
    return tsne.fit_transform(embeddings)


def get_suffix_label(filename: str) -> str:
    """Classify a parquet filename by its suffix (_10, _30, or other)."""
    stem = filename.removesuffix(".parquet")
    if stem.endswith("_10"):
        return "_10"
    elif stem.endswith("_30"):
        return "_30"
    else:
        return "other"


def plot_tsne(
    coords: np.ndarray,
    labels: list[str],
    output_path: str,
    title: str = "t-SNE of LC-MS Run Embeddings",
    figsize: tuple[float, float] = (12, 8),
    dpi: int = 150,
) -> None:
    """Scatter plot of t-SNE coordinates colored by label."""
    unique_labels = sorted(set(labels))
    n_labels = len(unique_labels)

    if n_labels <= 20:
        cmap = matplotlib.colormaps["tab20"]
    else:
        cmap = matplotlib.colormaps["gist_ncar"]

    label_to_color = {g: cmap(i / max(n_labels - 1, 1)) for i, g in enumerate(unique_labels)}
    labels_arr = np.array(labels)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for lbl in unique_labels:
        mask = labels_arr == lbl
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            color=label_to_color[lbl],
            s=30,
            alpha=0.7,
            edgecolors="none",
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=label_to_color[lbl],
               markersize=8, linestyle="None", label=lbl)
        for lbl in unique_labels
    ]
    ncol = 2 if n_labels > 20 else 1
    fontsize = 6 if n_labels > 20 else 8
    ax.legend(
        handles=handles,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=fontsize,
        ncol=ncol,
    )

    fig.savefig(output_path, bbox_inches="tight")
    logger.info(f"Saved plot to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="t-SNE visualization of run-level embeddings"
    )
    parser.add_argument(
        "--ckpt_path",
        default=None,
        help="Path to MS1Encoder checkpoint (not needed when loading cached embeddings)",
    )
    parser.add_argument(
        "--data_dir",
        default="/mnt/data/shared/lc_ms_foundation/abele_data/parquet",
        help="Directory containing parquet files (one per run)",
    )
    parser.add_argument(
        "--meta_path",
        default="/mnt/data/shared/lc_ms_foundation/abele_data/filtered_abele_metadata.csv",
        help="Path to metadata CSV",
    )
    parser.add_argument(
        "--output",
        default="tsne_plot.png",
        help="Output image path",
    )
    parser.add_argument(
        "--seq_len", type=int, default=200, help="Max peaks per spectrum"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Spectra per forward pass"
    )
    parser.add_argument(
        "--perplexity", type=float, default=30.0, help="t-SNE perplexity"
    )
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--figsize",
        default="12,8",
        help="Figure size as 'W,H'",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--color_by",
        choices=["genus", "suffix"],
        default="genus",
        help="Color runs by 'genus' (from metadata) or 'suffix' (_10, _30, other from filename)",
    )
    parser.add_argument(
        "--embeddings_path",
        default="run_embeddings.npz",
        help="Path to save/load precomputed embeddings (.npz). "
        "If the file exists, skips model loading and embedding computation.",
    )
    args = parser.parse_args()

    figsize = tuple(float(x) for x in args.figsize.split(","))
    args.embeddings_path = os.path.abspath(args.embeddings_path)

    if os.path.exists(args.embeddings_path):
        # Load cached embeddings
        logger.info(f"Loading cached embeddings from {args.embeddings_path}")
        data = np.load(args.embeddings_path, allow_pickle=True)
        embeddings = data["embeddings"]
        genera = data["genera"].tolist()
        filenames = data["filenames"].tolist()
    else:
        # 1. Load model
        if args.ckpt_path is None:
            parser.error("--ckpt_path is required when no cached embeddings exist")
        logger.info(f"Loading checkpoint: {args.ckpt_path}")
        model = MS1Encoder.load_from_checkpoint(args.ckpt_path, map_location="cpu")
        model.eval()
        model.cpu()
        logger.info(f"Model loaded (d_model={model.d_model})")

        # 2. Load parquet data
        dfs = load_parquet_runs(args.data_dir)
        logger.info(f"Loaded {len(dfs)} parquet files")

        # 3. Load metadata and build genus mapping
        meta_df = load_metadata(args.meta_path)
        run_genus_map, missing = build_run_genus_map(list(dfs.keys()), meta_df)
        logger.info(f"Matched {len(run_genus_map)} runs to genus labels")
        if missing:
            logger.warning(f"{len(missing)} parquet files have no metadata match")

        # 4. Compute run embeddings
        embeddings, genera, filenames = compute_all_embeddings(
            model, dfs, run_genus_map, args.seq_len, args.batch_size
        )

        # Save embeddings to disk
        np.savez(
            args.embeddings_path,
            embeddings=embeddings,
            genera=np.array(genera),
            filenames=np.array(filenames),
        )
        logger.info(f"Saved embeddings to {args.embeddings_path}")

    logger.info(
        f"Embeddings: {embeddings.shape} " f"({len(set(genera))} unique genera)"
    )

    # 5. t-SNE
    logger.info("Running t-SNE...")
    coords = run_tsne(
        embeddings, perplexity=args.perplexity, random_state=args.random_state
    )

    # 6. Plot
    if args.color_by == "suffix":
        labels = [get_suffix_label(f) for f in filenames]
        title = "t-SNE of LC-MS Run Embeddings (colored by gradient length)"
    else:
        labels = genera
        title = "t-SNE of LC-MS Run Embeddings (colored by genus)"
    plot_tsne(coords, labels, args.output, title=title, figsize=figsize, dpi=args.dpi)
    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
