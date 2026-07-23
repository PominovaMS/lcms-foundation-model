"""Evaluate a *pretrained* MS1Encoder checkpoint with a downstream linear probe.

Loads a checkpoint (e.g. one pretrained on a PRIDE training set), freezes the
encoder, trains a fresh linear probe on the abele genus-classification task, and
reports validation accuracy. Unlike ``eval/retrain_eval.py`` this does NOT run any
SSL training — it only probes the representation the checkpoint already learned.

Keep ``--n_probe_genera`` / ``--n_ssl_top`` identical across runs so the abele
probe split (from ``assign_splits``) is fixed and stages are comparable. Point
``--results_csv`` at the same file across stages to accumulate the scaling curve.

Usage:
    python eval/probe_checkpoint.py \\
        --ckpt_path /path/to/last.ckpt \\
        --data_dir /mnt/data/shared/lc_ms_foundation/abele_data/mzml \\
        --meta_path /path/to/all_abele_metadata.csv \\
        --config config.yaml \\
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
    """Append one result row, writing a header first if the file is new."""
    fieldnames = [
        "run_name",
        "ckpt_path",
        "n_probe_classes",
        "probe_val_acc",
        "probe_val_loss",
    ]
    is_new = not os.path.exists(results_csv) or os.path.getsize(results_csv) == 0
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


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
    parser.add_argument("--n_probe_genera", type=int, default=15)
    parser.add_argument("--n_ssl_top", type=int, default=3)
    parser.add_argument(
        "--probe_all",
        action="store_true",
        help="Probe on ALL eligible abele genera instead of the 15 mid-sized ones "
        "(ignores --n_probe_genera / --n_ssl_top). Richer eval, but slower and not "
        "comparable to the default 15-genera numbers.",
    )
    # Probe hyperparameters (match eval/retrain_eval.py defaults).
    parser.add_argument("--probe_lr", type=float, default=1e-2)
    parser.add_argument("--probe_n_epochs", type=int, default=100)
    parser.add_argument("--probe_min_train_loss", type=float, default=0.3)
    parser.add_argument(
        "--device",
        default=None,
        help="torch device (default: cuda if available else cpu)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or Path(args.ckpt_path).resolve().parent.parent.name

    config = load_config(args.config)

    # Load and split metadata (deterministic — same split every run).
    meta_df = load_metadata(args.meta_path)
    meta_df = assign_splits(
        meta_df,
        n_probe_genera=args.n_probe_genera,
        n_ssl_top=args.n_ssl_top,
        probe_all=args.probe_all,
    )

    # Load ONLY the probe files (skip the abele SSL corpus — not needed here).
    # NOTE: the "SSL: ... genera" line above is assign_splits' abele-internal split.
    # Here the encoder was pretrained on PRIDE, so those genus_class=-1 files are
    # NOT used — only the probe_train/probe_val files below are loaded.
    probe_files = meta_df.filter(
        pl.col("split").is_in(["probe_train", "probe_val"])
    )["peak_file"].to_list()
    n_probe_genera = meta_df.filter(pl.col("genus_class") >= 0)["genus_class"].n_unique()
    logger.info(
        f"Downstream probe: {n_probe_genera} genera / {len(probe_files)} files "
        f"({'all eligible' if args.probe_all else 'mid-sized subset'}). "
        f"The abele 'SSL' (genus_class=-1) files above are NOT used — "
        f"pretraining was on PRIDE."
    )
    dfs = load_mzml_data(args.data_dir, probe_files, config.data.max_num_peaks)

    probe_train_loader, probe_val_loader = build_probe_dataloaders(
        dfs, meta_df, config
    )
    logger.info(
        f"Probe runs — train: {len(probe_train_loader.dataset)}, "
        f"val: {len(probe_val_loader.dataset)}"
    )

    num_probe_classes = meta_df.filter(meta_df["genus_class"] >= 0)[
        "genus_class"
    ].n_unique()

    logger.info(f"Loading checkpoint: {args.ckpt_path}")
    model = MS1Encoder.load_from_checkpoint(args.ckpt_path, map_location=device)
    model.eval()
    model.to(device)

    val_acc, val_loss = run_retrain_probe(
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
    )

    logger.info(
        f"[{run_name}] probe_val_acc={val_acc:.4f}  probe_val_loss={val_loss:.4f} "
        f"({num_probe_classes} classes)"
    )

    if args.results_csv:
        append_result(
            args.results_csv,
            {
                "run_name": run_name,
                "ckpt_path": args.ckpt_path,
                "n_probe_classes": num_probe_classes,
                "probe_val_acc": f"{val_acc:.6f}",
                "probe_val_loss": f"{val_loss:.6f}",
            },
        )
        logger.info(f"Appended result to {args.results_csv}")


if __name__ == "__main__":
    main()
