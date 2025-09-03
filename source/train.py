"""Train the model."""  # Should this script be outside of the source folder (as a main entry point)?

import os
import polars as pl
import depthcharge as dc
import pytorch_lightning as L
from torch.utils.data import DataLoader
from model import MS1Encoder

# Path to the training data
DATA_DIR = "../data"
# Training batch size
BATCH_SIZE = 20
# Directory to save model checkpoints and logs
CHECKPOINT_PATH = "./train_checkpoints"

# Load training data
ms1_dfs = [
    dc.data.spectra_to_df(
        os.path.join(DATA_DIR, mzml_file),
        metadata_df=None,
        ms_level=1,
        preprocessing_fn=None,
        valid_charge=None,
        custom_fields=None,
        progress=True,
    )
    for mzml_file in os.listdir(DATA_DIR)
]
ms1_df = pl.concat(ms1_dfs, how="vertical")

# Rescale Intensities in the dataframe to [0, 1].
# (more logical would be to do it in the dataset,
# but unclear how to implement it in LanceDataset)
ms1_df = ms1_df.with_columns(
    pl.col("intensity_array") / pl.col("intensity_array").list.max()
)

# Split into train/val (dummy split for now)
train_mzmls = ["b1945_293T_proteinID_09B_QE3_122212.mzML"]
train_idx = ms1_df["peak_file"].is_in(train_mzmls)
val_idx = ~train_idx

train_df = ms1_df.filter(train_idx)
val_df = ms1_df.filter(val_idx)
train_dataset = dc.data.SpectrumDataset(train_df, batch_size=2)
val_dataset = dc.data.SpectrumDataset(val_df, batch_size=2)
print("N train spectra", train_dataset.n_spectra)
print("N val spectra:", val_dataset.n_spectra)


train_dataset.batch_size = BATCH_SIZE
train_loader = DataLoader(train_dataset, batch_size=None, num_workers=1)
val_dataset.batch_size = BATCH_SIZE
val_loader = DataLoader(val_dataset, batch_size=None, num_workers=1)


root_dir = os.path.join(CHECKPOINT_PATH, "foundation_model")
os.makedirs(root_dir, exist_ok=True)

logger = L.loggers.TensorBoardLogger(
    os.path.join(root_dir, "lightning_logs"),
    name="test_ms1_model",
)

# TODO: set reasonable hyperparameters and move them to constants/config
model = MS1Encoder(
    d_model=12,  # 8,
    nhead=1,  # 8,
    dim_feedforward=24,  # 512,
    n_layers=1,  # ,
    dropout=0.1,
    n_bins=2000,
    bin_mz_min=0,
    bin_mz_max=2000,
    masked_peaks_fraction=0.3,
    lr=5e-4,
    warmup_iters=1000,
    cosine_schedule_period_iters=32000,
)

trainer = L.Trainer(
    #     resume_from_checkpoint=ckpt_path,
    logger=logger,
    default_root_dir=root_dir,
    callbacks=[],  # [ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
    accelerator="auto",  # "gpu",
    devices=1,
    max_epochs=50,
    gradient_clip_val=5,
    num_sanity_val_steps=2,
)

# Train the model
trainer.fit(model, train_loader, val_dataloaders=[val_loader])
