"""Train the model."""  # Should this script be outside of the source folder (as a main entry point)?

import argparse
import os
import yaml
import pytorch_lightning as L
from depthcharge.data import SpectrumDataset, spectra_to_df, preprocessing
from torch.utils.data import DataLoader
from model import MS1Encoder
from config import (
    ExperimentConfig,
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        config = ExperimentConfig(
            name=config_dict["name"],
            data=DataConfig(**config_dict["data"]),
            model=ModelConfig(**config_dict["model"]),
            optimizer=OptimizerConfig(**config_dict["optimizer"]),
            training=TrainingConfig(**config_dict["training"]),
        )
    return config


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, help="The path to the training data.")
parser.add_argument(
    "--config", default="../config.yaml", help="Path to configuration file"
)
parser.add_argument(
    "--run_name",
    default=None,
    help="Run name for logs + checkpoints (default: config.name). Use a distinct "
    "name per training set so stages don't overwrite each other.",
)
parser.add_argument(
    "--max_epochs",
    type=int,
    default=None,
    help="Override config.training.max_epochs (e.g. a small value for a smoke test).",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Stop after this many optimizer steps (overrides --max_epochs). Use this to "
    "train every diversity stage to the SAME number of steps for a fair comparison.",
)
args = parser.parse_args()

# Load configuration
config = load_config(args.config)

# Extract configuration values
BATCH_SIZE = config.data.batch_size
CHECKPOINT_PATH = config.training.checkpoint_path
RUN_NAME = args.run_name or config.name
# --max_steps takes precedence: cap by steps and leave epochs unlimited (-1) so the
# step budget is the sole stopping criterion. Otherwise stop on epochs as before.
if args.max_steps is not None:
    MAX_STEPS = args.max_steps
    MAX_EPOCHS = -1
else:
    MAX_STEPS = -1
    MAX_EPOCHS = (
        args.max_epochs if args.max_epochs is not None else config.training.max_epochs
    )

# Load training data
train_data_dir = os.path.join(args.data_dir, "train_mzml")
val_data_dir = os.path.join(args.data_dir, "val_mzml")

preprocessing_fn = [
    preprocessing.filter_intensity(max_num_peaks=config.data.max_num_peaks),
    preprocessing.scale_intensity(scaling="root", max_intensity=1.0),
]

def build_dataset(data_dir, preprocessing_fn, batch_size):
    """Build a SpectrumDataset by streaming one mzML into Lance at a time.

    Appending per file (depthcharge's `add_spectra`, mode="append") keeps only a
    single file's DataFrame in RAM, so peak memory scales with one file instead of
    the whole corpus — the previous `pl.concat` of every file did not scale to a
    full dataset (hundreds of ~195k-spectra files).
    """
    # Only feed mzML to the parser. depthcharge dispatches a parser by extension, and
    # its MGF parser is MS2-only ("ms_level 1 is currently not supported") — so a stray
    # .mgf (or any non-mzML) symlinked into the dir would crash MS1 ingestion. Match the
    # `.mzml` convention used by the stage builders and peak_stats.py.
    all_entries = sorted(os.listdir(data_dir))
    mzml_files = [f for f in all_entries if f.lower().endswith((".mzml", ".mzml.gz"))]
    skipped = [f for f in all_entries if f not in mzml_files]
    if skipped:
        print(f"Skipping {len(skipped)} non-mzML file(s) in {data_dir}: {skipped}")

    ds = None
    for mzml_file in mzml_files:
        df = spectra_to_df(
            os.path.join(data_dir, mzml_file),
            metadata_df=None,
            ms_level=1,
            preprocessing_fn=preprocessing_fn,
            valid_charge=None,
            custom_fields=None,
            progress=True,
        )
        if ds is None:
            ds = SpectrumDataset(df, batch_size=batch_size)
        else:
            ds.add_spectra(df)
        del df  # free this file before loading the next
    if ds is None:
        raise SystemExit(f"No mzML files in {data_dir}")
    return ds


train_dataset = build_dataset(train_data_dir, preprocessing_fn, BATCH_SIZE)
val_dataset = build_dataset(val_data_dir, preprocessing_fn, BATCH_SIZE)
print("N train spectra", train_dataset.n_spectra)
print("N val spectra:", val_dataset.n_spectra)

train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)

root_dir = os.path.join(CHECKPOINT_PATH, "foundation_model")
os.makedirs(root_dir, exist_ok=True)

logger = L.loggers.TensorBoardLogger(
    os.path.join(root_dir, "lightning_logs"),
    name=RUN_NAME,
)

# Save checkpoints under a deterministic per-run directory (NOT logger.log_dir,
# which appends a version_N/ subdir) so downstream tooling can find `last.ckpt` at
# a stable path: <root>/lightning_logs/<run_name>/checkpoints/last.ckpt.
# `save_last=True` gives that stable file to probe downstream.
ckpt_dir = os.path.join(root_dir, "lightning_logs", RUN_NAME, "checkpoints")
checkpoint_callback = L.callbacks.ModelCheckpoint(
    dirpath=ckpt_dir,
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_last=True,
    filename="{epoch}-{step}-{val_loss:.4f}",
)

# TODO: set reasonable hyperparameters and move them to constants/config
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

trainer = L.Trainer(
    #     resume_from_checkpoint=ckpt_path,
    logger=logger,
    default_root_dir=root_dir,
    callbacks=[checkpoint_callback],
    accelerator=config.training.accelerator,
    devices=config.training.devices,
    max_epochs=MAX_EPOCHS,
    max_steps=MAX_STEPS,
    gradient_clip_val=config.training.gradient_clip_val,
    num_sanity_val_steps=2,
)

# Train the model
trainer.fit(model, train_loader, val_dataloaders=[val_loader])
