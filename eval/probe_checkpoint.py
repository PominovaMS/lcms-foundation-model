"""Evaluate a *pretrained* MS1Encoder checkpoint with a downstream linear probe.

Loads a checkpoint (e.g. one pretrained on a PRIDE training set), freezes the
encoder, trains a fresh linear probe on the abele organism-classification task, and
reports validation accuracy. No SSL training happens here — it only probes the
representation the checkpoint already learned.

The default task is the **2 largest abele genera** (Pseudomonas 312 files vs
Staphylococcus 136), each genus's files split proportionally into train/val by
``assign_splits``. The classes are deliberately left unbalanced, so the bar to beat
is ``majority_acc`` (~0.70), NOT ``random_acc`` (0.50) — read
``probe_val_acc_macro`` and ``n_pred_classes`` alongside ``probe_val_acc``.

``--label_offset`` shifts the selection down the ranking: ``--label_offset 1``
probes the second- and third-largest labels instead, which lowers ``majority_acc``
when the largest label dwarfs the rest, at the cost of fewer files.

Keep ``--probe_label`` / ``--n_classes`` / ``--val_frac`` / ``--label_offset``
identical across runs so the probe split is fixed and stages are comparable. The
first three and the resulting ``class_names`` are recorded in ``--results_csv`` so
a row is self-describing. Point ``--results_csv`` at the same file across stages to
accumulate the scaling curve.

NOTE: the CSV header changed with the split rewrite (the five ``n_probe_genera`` /
``n_ssl_top`` / ``probe_all`` / ``max_files_per_*`` columns became ``probe_label`` /
``n_classes`` / ``val_frac`` / ``class_names``, and ``ssl_epoch`` / ``ssl_step`` were
added). Point ``--results_csv`` at a fresh file; rows written under the old split are
not comparable with these anyway.

``ssl_epoch`` / ``ssl_step`` are read out of the checkpoint, so probing several
checkpoints of ONE pretraining run (``train.py --save_every_n_epochs``) into one CSV
gives probe accuracy as a function of pretraining epoch without needing a distinct
``--run_name`` per row.

Accuracy is reported as a mean over ``--probe_repeats`` seeded probe fits. Compare
runs only when their gap exceeds ``probe_val_acc_std``; below that it is probe
initialisation noise, not the encoder.

Usage:
    python eval/probe_checkpoint.py \\
        --ckpt_path /path/to/last.ckpt \\
        --data_dir /mnt/data/shared/lc_ms_foundation/abele_data/mzml \\
        --meta_path /path/to/all_abele_metadata.csv \\
        --config config.yaml \\
        --probe_label genus --n_classes 2 \\
        --results_csv sweep.csv
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import yaml

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl
import torch

from source.model import MS1Encoder
from source.config import (
    ExperimentConfig,
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from data import (
    load_metadata,
    load_mzml_data,
    assign_splits,
    build_probe_dataloaders,
)
from probe import run_retrain_probe

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path) as f:
        d = yaml.safe_load(f)
    return ExperimentConfig(
        name=d["name"],
        data=DataConfig(**d["data"]),
        model=ModelConfig(**d["model"]),
        optimizer=OptimizerConfig(**d["optimizer"]),
        training=TrainingConfig(**d["training"]),
    )


def append_result(results_csv: str, row: dict) -> None:
    """Append one result row, writing a header first if the file is new.

    The first five columns keep their names and positions, so ``plot_sweep.py``
    (which reads only ``run_name`` and ``probe_val_acc``) is unaffected by the
    split-flag columns changing.
    """
    fieldnames = [
        "run_name",
        "ckpt_path",
        # which point of PRETRAINING this checkpoint is — read out of the checkpoint
        # itself, so an epoch sweep's rows are distinguishable under one run_name
        "ssl_epoch",
        "ssl_step",
        "n_probe_classes",
        "probe_val_acc",
        "probe_val_loss",
        # spread across seeded repeats — a gap smaller than this is noise
        "probe_val_acc_std",
        # collapse diagnostics
        "probe_val_acc_macro",
        "probe_val_acc_macro_std",
        "majority_acc",
        "random_acc",
        "n_pred_classes",
        "probe_epochs",
        # settings that must match for two rows to be comparable
        "probe_seed",
        "probe_repeats",
        "probe_lr",
        "probe_n_epochs",
        "probe_min_train_loss",
        "probe_label",
        "n_classes",
        "val_frac",
        "class_names",
        "n_probe_train_files",
        "n_probe_val_files",
    ]
    is_new = not os.path.exists(results_csv) or os.path.getsize(results_csv) == 0
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def infer_run_name(ckpt_path: str) -> str:
    """The pretraining --run_name a checkpoint belongs to, from its path.

    train.py writes ``<run_name>/checkpoints/last.ckpt`` and, under
    ``--save_every_n_epochs``, ``<run_name>/checkpoints/epochs/epochNNN-*.ckpt``, so
    the run name is not at a fixed depth. Skip the structural dirs instead of
    counting parents, or every periodic checkpoint is labelled "checkpoints".
    """
    for parent in Path(ckpt_path).resolve().parents:
        if parent.name not in ("checkpoints", "epochs"):
            return parent.name
    return Path(ckpt_path).stem


def main():
    parser = argparse.ArgumentParser(
        description="Downstream linear-probe eval of a pretrained checkpoint"
    )
    parser.add_argument(
        "--ckpt_path", required=True, help="Path to a pretrained MS1Encoder checkpoint"
    )
    parser.add_argument(
        "--data_dir", required=True, help="Directory with abele mzML files"
    )
    parser.add_argument("--meta_path", required=True, help="Path to metadata CSV")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--mzml_cache_dir",
        default=None,
        help="Persist the parsed spectra here (one parquet per mzML) and reload "
        "them on later runs instead of re-parsing — the mzML parse dominates the "
        "runtime of a probe. Preprocessing is baked in at parse time and entries "
        "are keyed by file name, so a cache dir is valid for exactly one "
        "--data_dir at one max_num_peaks; the run aborts if either disagrees "
        "with what the dir was built from. Use one dir per dataset.",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="Label for this row in the results CSV (default: inferred from ckpt dir)",
    )
    parser.add_argument(
        "--results_csv",
        default=None,
        help="If set, append a result row (run_name, ckpt, acc, loss) to this CSV",
    )
    # Split control — keep fixed across stages for a comparable probe split.
    parser.add_argument(
        "--probe_label",
        choices=["genus", "species"],
        default="genus",
        help="What the probe classifies. Default 'genus': the 2 largest abele "
        "genera are Pseudomonas (312 files) and Staphylococcus (136), against a "
        "modal 3 files per species elsewhere in the set.",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=2,
        help="Number of classes: the N most abundant labels by file count, ties "
        "broken alphabetically.",
    )
    parser.add_argument(
        "--label_offset",
        type=int,
        default=0,
        help="Skip this many labels from the top before selecting. 0 (default) is "
        "the plain top-N; --label_offset 1 --n_classes 2 probes the second- and "
        "third-largest labels, which lowers majority_acc when the largest label "
        "dwarfs the rest, at the cost of fewer files. Keep it fixed across a "
        "sweep, like --probe_label / --n_classes / --val_frac.",
    )
    parser.add_argument(
        "--val_frac",
        type=float,
        default=0.3,
        help="Fraction of EACH class's files held out for probe validation. The "
        "split is proportional, so train and val carry the same class "
        "distribution and share species.",
    )
    # Probe hyperparameters.
    parser.add_argument("--probe_lr", type=float, default=1e-2)
    parser.add_argument("--probe_n_epochs", type=int, default=100)
    parser.add_argument("--probe_min_train_loss", type=float, default=0.3)
    parser.add_argument(
        "--probe_seed",
        type=int,
        default=0,
        help="Seed for the probe initialisation. Repeat i uses probe_seed + i.",
    )
    parser.add_argument(
        "--probe_repeats",
        type=int,
        default=3,
        help="Probe fits over the same cached embeddings; the spread across them "
        "is the noise floor for comparing two checkpoints. Default 3.",
    )
    parser.add_argument(
        "--encode_chunk_size",
        type=int,
        default=512,
        help="Spectra per forward pass when embedding a run. A whole LC-MS run in "
        "one forward needs tens of GB and OOMs the GPU; chunking is exact, so this "
        "changes only whether the numbers compute, never what they are. Lower it "
        "if the encoder still OOMs.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device (default: cuda if available else cpu)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or infer_run_name(args.ckpt_path)

    config = load_config(args.config)

    # Load and split metadata (deterministic — same split every run).
    label_col = "organism" if args.probe_label == "species" else "genus"
    meta_df = load_metadata(args.meta_path)
    meta_df = assign_splits(
        meta_df,
        n_classes=args.n_classes,
        label_col=label_col,
        val_frac=args.val_frac,
        label_offset=args.label_offset,
    )

    # Load ONLY the selected classes' files; everything else is split="unused".
    probe_files = meta_df.filter(
        pl.col("split").is_in(["probe_train", "probe_val"])
    )["peak_file"].to_list()
    class_names = ";".join(
        sorted(meta_df.filter(pl.col("label_class") >= 0)[label_col].unique().to_list())
    )
    logger.info(
        f"Downstream probe: {args.n_classes} {args.probe_label} classes "
        f"({class_names}) / {len(probe_files)} files."
    )
    dfs = load_mzml_data(
        args.data_dir,
        probe_files,
        config.data.max_num_peaks,
        cache_dir=args.mzml_cache_dir,
    )

    probe_train_loader, probe_val_loader = build_probe_dataloaders(
        dfs, meta_df, config
    )
    logger.info(
        f"Probe runs — train: {len(probe_train_loader.dataset)}, "
        f"val: {len(probe_val_loader.dataset)}"
    )

    num_probe_classes = meta_df.filter(meta_df["label_class"] >= 0)[
        "label_class"
    ].n_unique()

    logger.info(f"Loading checkpoint: {args.ckpt_path}")
    # Lightning stores the epoch/step it was written at; read them rather than parsing
    # the filename, which differs between the best-val_loss and the periodic callbacks.
    ckpt_meta = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    ssl_epoch = ckpt_meta.get("epoch", "")
    ssl_step = ckpt_meta.get("global_step", "")
    del ckpt_meta
    model = MS1Encoder.load_from_checkpoint(args.ckpt_path, map_location=device)
    model.eval()
    model.to(device)

    res = run_retrain_probe(
        model,
        probe_train_loader,
        probe_val_loader,
        d_model=config.model.d_model,
        num_classes=num_probe_classes,
        target_key="label",
        lr=args.probe_lr,
        n_epochs=args.probe_n_epochs,
        min_train_loss=args.probe_min_train_loss,
        device=device,
        seed=args.probe_seed,
        n_repeats=args.probe_repeats,
        chunk_size=args.encode_chunk_size,
    )

    logger.info(
        f"[{run_name}] ssl_epoch={ssl_epoch} ssl_step={ssl_step}"
    )
    logger.info(
        f"[{run_name}] probe_val_acc={res['val_acc']:.4f}±{res['val_acc_std']:.4f}  "
        f"macro={res['val_acc_macro']:.4f}±{res['val_acc_macro_std']:.4f}  "
        f"(majority={res['majority_acc']:.4f}, random={res['random_acc']:.4f}, "
        f"{num_probe_classes} classes, "
        f"{res['n_pred_classes']:.1f}/{num_probe_classes} classes predicted, "
        f"{res['probe_epochs']:.0f} probe epochs)"
    )

    if args.results_csv:
        append_result(
            args.results_csv,
            {
                "run_name": run_name,
                "ckpt_path": args.ckpt_path,
                "ssl_epoch": ssl_epoch,
                "ssl_step": ssl_step,
                "n_probe_classes": num_probe_classes,
                "probe_val_acc": f"{res['val_acc']:.6f}",
                "probe_val_loss": f"{res['val_loss']:.6f}",
                "probe_val_acc_std": f"{res['val_acc_std']:.6f}",
                "probe_val_acc_macro": f"{res['val_acc_macro']:.6f}",
                "probe_val_acc_macro_std": f"{res['val_acc_macro_std']:.6f}",
                "majority_acc": f"{res['majority_acc']:.6f}",
                "random_acc": f"{res['random_acc']:.6f}",
                "n_pred_classes": f"{res['n_pred_classes']:.1f}",
                "probe_epochs": f"{res['probe_epochs']:.1f}",
                "probe_seed": args.probe_seed,
                "probe_repeats": args.probe_repeats,
                "probe_lr": args.probe_lr,
                "probe_n_epochs": args.probe_n_epochs,
                "probe_min_train_loss": args.probe_min_train_loss,
                "probe_label": args.probe_label,
                "n_classes": args.n_classes,
                "val_frac": args.val_frac,
                "class_names": class_names,
                "n_probe_train_files": len(probe_train_loader.dataset),
                "n_probe_val_files": len(probe_val_loader.dataset),
            },
        )
        logger.info(f"Appended result to {args.results_csv}")


if __name__ == "__main__":
    main()
