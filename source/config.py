from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    train_dir: str
    val_dir: str
    batch_size: int
    max_num_peaks: int


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
    lr: float  # peak LR of the one-cycle schedule
    warmup_iters: int
    # Length of the one-cycle LR schedule, in optimizer steps. None (yaml `null`)
    # means derive it from the run's stop criterion; an explicit int overrides that.
    total_steps: Optional[int] = None
    div_factor: float = 25.0  # initial LR = lr / div_factor
    final_div_factor: float = 1e4  # final LR = initial LR / final_div_factor


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
