"""Train the model."""  # Should this script be outside of the source folder (as a main entry point)?

import argparse
import json
import math
import os
import yaml
import pytorch_lightning as L
from depthcharge.data import SpectrumDataset, preprocessing
from torch.utils.data import DataLoader
from dataset import (
    SpectrumIndexDataset,
    batch_collate,
    build_dataset,
    iterable_steps_per_epoch,
)
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
source_group = parser.add_mutually_exclusive_group(required=True)
source_group.add_argument(
    "--data_dir",
    help="Directory of mzML to ingest, holding train_mzml/ and val_mzml/. Ingested "
    "into a temporary Lance DB in FILE ORDER (unshuffled) and discarded when the run "
    "ends. Prefer --lance_dir: it is shuffled and reusable.",
)
source_group.add_argument(
    "--lance_dir",
    help="Directory holding a prebuilt, pre-shuffled dataset from "
    "scripts/build_lance.py (train.lance, val.lance, dataset_info.json). Skips "
    "ingestion entirely, so every run over the same data starts from identical "
    "spectra in identical order.",
)
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
parser.add_argument(
    "--lr",
    type=float,
    default=None,
    help="Override config.optimizer.lr (peak LR after warmup). Use this to sweep LR "
    "from a job script without editing the shared config.",
)
parser.add_argument(
    "--warmup_iters",
    type=int,
    default=None,
    help="Override config.optimizer.warmup_iters (linear warmup length in optimizer "
    "steps). Raise it alongside --lr for large/heterogeneous training sets.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Seeds the per-epoch batch shuffle (and torch/numpy generally), so a run is "
    "reproducible.",
)
parser.add_argument(
    "--no_shuffle",
    action="store_true",
    help="Read the training data in stored order instead of re-drawing batches each "
    "epoch. Reproduces pre-shuffle runs.",
)
parser.add_argument(
    "--save_every_n_epochs",
    type=int,
    default=0,
    help="Also keep a weights-only checkpoint every N epochs, under "
    "checkpoints/epochs/, so the encoder can be probed as a function of "
    "pretraining epoch. 0 (default) = off; only best-val_loss and last.ckpt are "
    "kept, which is what every earlier run has.",
)
args = parser.parse_args()

# Seed before anything stochastic: the batch shuffle, the peak masking, and the model
# init all draw from these generators.
L.seed_everything(args.seed, workers=True)

# Load configuration
config = load_config(args.config)

# Optimizer overrides: CLI wins over the config file so a job script can sweep LR
# without editing config.yaml (which every other experiment also reads).
LR = args.lr if args.lr is not None else config.optimizer.lr
WARMUP_ITERS = (
    args.warmup_iters if args.warmup_iters is not None else config.optimizer.warmup_iters
)

# Extract configuration values
BATCH_SIZE = config.data.batch_size
ACCUMULATE_GRAD_BATCHES = config.training.accumulate_grad_batches
PRECISION = config.training.precision
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
if args.lance_dir:
    # Prebuilt, pre-shuffled dataset (scripts/build_lance.py). Rows are already in
    # random order on disk, so the sequential Lance scan yields batches that are a
    # cross-section of files and datasets — no loader-side shuffling needed.
    info_path = os.path.join(args.lance_dir, "dataset_info.json")
    try:
        with open(info_path) as f:
            dataset_info = json.load(f)
    except FileNotFoundError:
        raise SystemExit(
            f"No dataset_info.json in {args.lance_dir}. Build the dataset with "
            f"scripts/build_lance.py, which writes it alongside the .lance dirs."
        )
    # Preprocessing is applied at INGEST, so it is baked into the stored spectra.
    # Training on a dataset built under a different peak cap would silently be a
    # different experiment, so refuse rather than let it through.
    built_peaks = dataset_info.get("preprocessing", {}).get("max_num_peaks")
    if built_peaks != config.data.max_num_peaks:
        raise SystemExit(
            f"Preprocessing mismatch: {info_path} was built with "
            f"max_num_peaks={built_peaks}, but the config asks for "
            f"{config.data.max_num_peaks}. Rebuild with scripts/build_lance.py "
            f"(preprocessing is baked in at ingest and cannot be changed after)."
        )
    train_dataset = SpectrumDataset.from_lance(
        os.path.join(args.lance_dir, "train.lance"), BATCH_SIZE
    )
    val_dataset = SpectrumDataset.from_lance(
        os.path.join(args.lance_dir, "val.lance"), BATCH_SIZE
    )
    print(
        f"Prebuilt dataset {args.lance_dir}  shuffle_seed={dataset_info.get('seed')}  "
        f"built={dataset_info.get('built_at')}  "
        f"source={dataset_info.get('data_dir')}"
    )
else:
    # Direct-ingest path: mzML into a temporary Lance DB, in sorted-filename order,
    # rebuilt every run. Per-epoch shuffling still applies on top, so this is only
    # the pathological one-file-per-batch case when combined with --no_shuffle.
    print(
        "NOTE: --data_dir re-parses every mzML into a temporary Lance DB and discards "
        "it at the end of the run, and stores them in file order. Per-epoch batch "
        "shuffling still applies, so batches are still cross-file — but with "
        "--no_shuffle they would not be. scripts/build_lance.py + --lance_dir builds "
        "once and reuses it."
    )
    preprocessing_fn = [
        preprocessing.filter_intensity(max_num_peaks=config.data.max_num_peaks),
        preprocessing.scale_intensity(scaling="root", max_intensity=1.0),
    ]
    train_dataset, _, _ = build_dataset(
        os.path.join(args.data_dir, "train_mzml"), preprocessing_fn, BATCH_SIZE
    )
    val_dataset, _, _ = build_dataset(
        os.path.join(args.data_dir, "val_mzml"), preprocessing_fn, BATCH_SIZE
    )

print("N train spectra", train_dataset.n_spectra)
print("N val spectra:", val_dataset.n_spectra)
print(
    f"batch shuffling: {'OFF (stored order)' if args.no_shuffle else 'per-epoch'}  "
    f"seed={args.seed}"
)

if args.no_shuffle:
    # Sequential scan in stored order. With a --lance_dir dataset that order is
    # already random, so batches stay cross-file; they just don't change per epoch.
    # With --data_dir it is file order, which is the sawtooth case.
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=0)
else:
    # Re-draw batches every epoch. DataLoader's RandomSampler does the shuffling;
    # SpectrumIndexDataset only exists because SpectrumDataset is an IterableDataset
    # and PyTorch refuses `shuffle=True` on those. One lance `take` per step.
    train_loader = DataLoader(
        SpectrumIndexDataset(train_dataset),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # the dataset holds a live Lance handle; not fork-safe
        collate_fn=batch_collate(train_dataset),
    )
# Validation is NEVER reshuffled. `validation_step` seeds its mask generator with
# `42 + batch_idx` so masks are identical across epochs — which only holds if batch k
# is the same spectra every time. Shuffling here would add noise to val loss and make
# epoch-to-epoch comparisons meaningless.
val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)

# Resolve the length of the one-cycle LR schedule. OneCycleLR raises once it is
# stepped past total_steps, so this has to be exact, not approximately right.
#
# The model prefers `trainer.estimated_stepping_batches` (which knows the real
# dataloader length, accumulate_grad_batches, limit_train_batches and max_steps).
# What is computed here is the fallback for when that is unavailable — an unsized
# IterableDataset — plus the startup diagnostics.
if args.no_shuffle:
    # NOT ceil(n_spectra / BATCH_SIZE): Lance batches never span fragments, so each
    # mzML file / write chunk adds a partial batch. That formula undercounted by up
    # to ~27% on many-file sets, which is what made OneCycleLR raise mid-run.
    batches_per_epoch = iterable_steps_per_epoch(train_dataset, BATCH_SIZE)
    derivation = "exact (per-fragment sum)"
else:
    # Map-style loader: len() is the true batches/epoch.
    batches_per_epoch = len(train_loader)
    derivation = "exact (len(train_loader))"

# OneCycleLR is stepped per OPTIMIZER step, not per batch, so gradient accumulation
# shortens the schedule. Lightning still steps on an epoch's trailing partial
# accumulation window, hence ceil — and ceil is also the safe rounding: overshooting
# only leaves the anneal slightly unfinished, while undershooting makes OneCycleLR
# raise mid-run.
steps_per_epoch = math.ceil(batches_per_epoch / ACCUMULATE_GRAD_BATCHES)

if MAX_STEPS != -1:
    total_steps = MAX_STEPS
    derivation = "exact (--max_steps)"
else:
    total_steps = steps_per_epoch * MAX_EPOCHS
total_steps_sched = total_steps

# An explicit config value overrides the derivation, and also stops the model from
# preferring the trainer's estimate over it.
auto_total_steps = config.optimizer.total_steps is None
if not auto_total_steps:
    total_steps_sched = config.optimizer.total_steps
    derivation = "from config (override)"
print(
    f"steps/epoch={steps_per_epoch}  total_steps={total_steps}  "
    f"schedule_total_steps={total_steps_sched} ({derivation})"
)
print(
    f"precision={PRECISION}  batch_size={BATCH_SIZE}  "
    f"accumulate_grad_batches={ACCUMULATE_GRAD_BATCHES}  "
    f"effective_batch={BATCH_SIZE * ACCUMULATE_GRAD_BATCHES}  "
    f"batches/epoch={batches_per_epoch}"
)
print(
    f"model: d_model={config.model.d_model} nhead={config.model.nhead} "
    f"head_dim={config.model.d_model // config.model.nhead} "
    f"dim_feedforward={config.model.dim_feedforward} "
    f"(ff/d={config.model.dim_feedforward / config.model.d_model:g}) "
    f"n_layers={config.model.n_layers} n_bins={config.model.n_bins}"
)
print(
    f"lr={LR}{' (CLI)' if args.lr is not None else ' (config)'}  "
    f"warmup_iters={WARMUP_ITERS}"
    f"{' (CLI)' if args.warmup_iters is not None else ' (config)'}  "
    f"pct_start={WARMUP_ITERS / total_steps_sched:.4g}"
)

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
callbacks = [checkpoint_callback]

# --save_every_n_epochs: keep an unpruned series of checkpoints so probe_checkpoint.py
# can be pointed at each one and produce probe accuracy as a function of pretraining
# epoch. The callback above cannot do this — it keeps only the single best val_loss.
#
# Its own subdirectory, so these files can never be confused with (or take the name of)
# the `last.ckpt` path the slurm scripts hardcode. save_top_k=-1 is required: with
# every_n_epochs alone, the default save_top_k=1 overwrites each file with the next.
#
# save_weights_only drops the two Adam moments, cutting ~925 MB to ~310 MB per file at
# d_model=1024 / n_layers=9 / ff=2048. load_from_checkpoint still works (hyper_parameters
# is saved either way), which is all the probe needs; these files cannot be RESUMED from,
# so last.ckpt remains the resume point.
if args.save_every_n_epochs > 0:
    callbacks.append(
        L.callbacks.ModelCheckpoint(
            dirpath=os.path.join(ckpt_dir, "epochs"),
            every_n_epochs=args.save_every_n_epochs,
            save_top_k=-1,
            save_weights_only=True,
            # Without auto_insert_metric_name=False Lightning expands each {name} to
            # "name={value}", giving "epochepoch=000-stepstep=00000004.ckpt".
            auto_insert_metric_name=False,
            filename="epoch{epoch:03d}-step{step:08d}",
        )
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
    lr=LR,
    warmup_iters=WARMUP_ITERS,
    total_steps=total_steps_sched,
    auto_total_steps=auto_total_steps,
    div_factor=config.optimizer.div_factor,
    final_div_factor=config.optimizer.final_div_factor,
)

trainer = L.Trainer(
    #     resume_from_checkpoint=ckpt_path,
    logger=logger,
    default_root_dir=root_dir,
    callbacks=callbacks,
    accelerator=config.training.accelerator,
    devices=config.training.devices,
    precision=PRECISION,
    accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
    max_epochs=MAX_EPOCHS,
    max_steps=MAX_STEPS,
    gradient_clip_val=config.training.gradient_clip_val,
    num_sanity_val_steps=2,
)

# Train the model
trainer.fit(model, train_loader, val_dataloaders=[val_loader])
