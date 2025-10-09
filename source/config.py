from dataclasses import dataclass
from typing import Optional

@dataclass
class DataConfig:
    train_dir: str
    val_dir: str
    batch_size: int

@dataclass
class ModelConfig:
    d_model: int
    nhead: int
    dim_feedforward: int
    n_layers: int
    dropout: float
    n_bins: int
    bin_mz_min: int
    bin_mz_max: int
    masked_peaks_fraction: float

@dataclass
class OptimizerConfig:
    lr: float
    warmup_iters: int
    cosine_schedule_period_iters: int

@dataclass
class TrainingConfig:
    checkpoint_path: str
    max_epochs: int
    gradient_clip_val: float
    accelerator: str
    devices: int

@dataclass
class ExperimentConfig:
    name: str
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig

