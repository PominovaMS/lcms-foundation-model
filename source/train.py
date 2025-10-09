"""Train the model."""  # Should this script be outside of the source folder (as a main entry point)?

import argparse
import os
import yaml
import polars as pl
import pytorch_lightning as L
from depthcharge.data import SpectrumDataset, spectra_to_df
from torch.utils.data import DataLoader
from model import MS1Encoder
from config import ExperimentConfig, DataConfig, ModelConfig, OptimizerConfig, TrainingConfig


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
        config = ExperimentConfig(
            name=config_dict['name'],
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            optimizer=OptimizerConfig(**config_dict['optimizer']),
            training=TrainingConfig(**config_dict['training'])
        )
    return config


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True, help="The path to the training data.")
parser.add_argument(
    "--config", default="../config.yaml", help="Path to configuration file"
)
args = parser.parse_args()

# Load configuration
config = load_config(args.config)

# Extract configuration values
BATCH_SIZE = config.data.batch_size
CHECKPOINT_PATH = config.training.checkpoint_path

# Load training data
train_data_dir = os.path.join(args.data_dir, "train_mzml")
val_data_dir = os.path.join(args.data_dir, "val_mzml")

dfs = [
    spectra_to_df(
        os.path.join(train_data_dir, mzml_file),
        metadata_df=None,
        ms_level=1,
        preprocessing_fn=None,
        valid_charge=None,
        custom_fields=None,
        progress=True,
    )
    for mzml_file in os.listdir(train_data_dir)
]
train_df = pl.concat(dfs, how="vertical")

dfs = [
    spectra_to_df(
        os.path.join(val_data_dir, mzml_file),
        metadata_df=None,
        ms_level=1,
        preprocessing_fn=None,
        valid_charge=None,
        custom_fields=None,
        progress=True,
    )
    for mzml_file in os.listdir(val_data_dir)
]
val_df = pl.concat(dfs, how="vertical")

# Rescale Intensities in the dataframe to [0, 1].
train_df = train_df.with_columns(
    pl.col("intensity_array") / pl.col("intensity_array").list.max()
)
val_df = val_df.with_columns(
    pl.col("intensity_array") / pl.col("intensity_array").list.max()
)

train_dataset = SpectrumDataset(train_df, batch_size=2)
val_dataset = SpectrumDataset(val_df, batch_size=2)
print("N train spectra", train_dataset.n_spectra)
print("N val spectra:", val_dataset.n_spectra)

train_dataset.batch_size = BATCH_SIZE
train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
val_dataset.batch_size = BATCH_SIZE
val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)

root_dir = os.path.join(CHECKPOINT_PATH, "foundation_model")
os.makedirs(root_dir, exist_ok=True)

logger = L.loggers.TensorBoardLogger(
    os.path.join(root_dir, "lightning_logs"),
    name=config.name,
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
    callbacks=[],  # [ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
    accelerator=config.training.accelerator,
    devices=config.training.devices,
    max_epochs=config.training.max_epochs,
    gradient_clip_val=config.training.gradient_clip_val,
    num_sanity_val_steps=2,
)

# Train the model
trainer.fit(model, train_loader, val_dataloaders=[val_loader])
