"""SSL pretraining with retrain-style downstream evaluation.

After each SSL training epoch:
  - The linear probe is re-initialized from scratch
  - Trained until loss < min_train_loss OR for max n_probe_epochs
  - Evaluated on the probe validation set

Usage:
    python eval/retrain_eval.py \\
        --data_dir /mnt/data/shared/lc_ms_foundation/abele_data/mzml \\
        --meta_path /path/to/all_abele_metadata.csv \\
        --config config.yaml \\
        --n_probe_genera 15
"""

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytorch_lightning as L

from source.model import MS1Encoder
from source.config import (
    ExperimentConfig,
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from callbacks import FineTuner
from data import (
    load_metadata,
    load_mzml_data,
    assign_splits,
    get_needed_files,
    build_dataloaders,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


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


def main():
    parser = argparse.ArgumentParser(
        description="SSL pretrain + retrain downstream eval"
    )
    parser.add_argument(
        "--data_dir", required=True, help="Directory containing mzML files"
    )
    parser.add_argument("--meta_path", required=True, help="Path to metadata CSV")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    # Split control
    parser.add_argument(
        "--n_probe_genera",
        type=int,
        default=15,
        help="Number of genera for downstream probe task (default: 15)",
    )
    parser.add_argument(
        "--n_ssl_top",
        type=int,
        default=3,
        help="Number of largest genera always reserved for SSL (default: 3)",
    )
    parser.add_argument(
        "--n_ssl_files",
        type=int,
        default=None,
        help="Cap SSL training files to this many (default: use all)",
    )
    # Probe hyperparameters
    parser.add_argument("--probe_lr", type=float, default=1e-2)
    parser.add_argument("--probe_n_epochs", type=int, default=100)
    parser.add_argument("--probe_min_train_loss", type=float, default=0.3)
    parser.add_argument(
        "--probe_eval_every",
        type=int,
        default=1,
        help="Retrain and evaluate probe every N SSL epochs (default: 1)",
    )
    parser.add_argument("--ssl_max_epochs", type=int, default=500)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Stop if val_loss doesn't improve for N epochs (default: disabled)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Load and split metadata
    meta_df = load_metadata(args.meta_path)
    meta_df = assign_splits(
        meta_df, n_probe_genera=args.n_probe_genera, n_ssl_top=args.n_ssl_top
    )

    # Only load files that will actually be used
    peak_files = get_needed_files(meta_df, args.data_dir, n_ssl_files=args.n_ssl_files)
    dfs = load_mzml_data(args.data_dir, peak_files, config.data.max_num_peaks)

    # Build DataLoaders
    train_loader, val_loader, probe_train_loader, probe_val_loader = build_dataloaders(
        dfs, meta_df, config
    )
    logging.info(f"SSL batches — train: {len(train_loader)}, val: {len(val_loader)}")
    logging.info(
        f"Probe runs — train: {len(probe_train_loader.dataset)}, val: {len(probe_val_loader.dataset)}"
    )

    # Number of probe classes = number of probe genera
    num_probe_classes = meta_df.filter(meta_df["genus_class"] >= 0)[
        "genus_class"
    ].n_unique()

    model = MS1Encoder(
        d_model=config.model.d_model,
        nhead=config.model.nhead,
        dim_feedforward=config.model.dim_feedforward,
        n_layers=config.model.n_layers,
        dropout=config.model.dropout,
        n_bins=config.model.n_bins,
        bin_mz_min=config.model.bin_mz_min,
        bin_mz_max=config.model.bin_mz_max,
        masked_peaks_fraction=config.model.masked_peaks_fraction,
        lr=config.optimizer.lr,
        warmup_iters=config.optimizer.warmup_iters,
        cosine_schedule_period_iters=config.optimizer.cosine_schedule_period_iters,
    )

    root_dir = os.path.join(config.training.checkpoint_path, "foundation_model")
    os.makedirs(root_dir, exist_ok=True)
    logger = L.loggers.TensorBoardLogger(
        os.path.join(root_dir, "lightning_logs"),
        name=config.name,
    )

    retrain_finetuner = FineTuner(
        encoder_output_dim=config.model.d_model,
        num_classes=num_probe_classes,
        target_key="label",
        probe_train_loader=probe_train_loader,
        probe_val_loader=probe_val_loader,
        lr=args.probe_lr,
        n_epochs=args.probe_n_epochs,
        min_train_loss=args.probe_min_train_loss,
        eval_every_n_epochs=args.probe_eval_every,
    )

    callbacks = [retrain_finetuner]
    if args.early_stopping_patience is not None:
        callbacks.append(
            L.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.early_stopping_patience,
                mode="min",
            )
        )

    trainer = L.Trainer(
        logger=logger,
        default_root_dir=root_dir,
        callbacks=callbacks,
        accelerator=config.training.accelerator,
        devices=config.training.devices,
        max_epochs=args.ssl_max_epochs,
        gradient_clip_val=config.training.gradient_clip_val,
        num_sanity_val_steps=2,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, val_dataloaders=[val_loader])


if __name__ == "__main__":
    main()
