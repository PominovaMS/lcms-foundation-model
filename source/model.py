import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as L
from depthcharge.encoders import PeakEncoder
from depthcharge.transformers import SpectrumTransformerEncoder


class MS1Encoder(L.LightningModule):
    def __init__(
        self,
        d_model=128,
        nhead=8,
        dim_feedforward=512,
        n_layers=4,
        dropout=0.1,
        n_bins=2000,
        bin_mz_min=0,
        bin_mz_max=2000,
        masked_peaks_fraction=0.3,
        lr=5e-4,
        warmup_iters=1000,
        cosine_schedule_period_iters=32000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.n_layers = n_layers
        self.dropout = dropout

        self.n_bins = n_bins
        self.bin_mz_min = bin_mz_min
        self.bin_mz_max = bin_mz_max
        self.masked_peaks_fraction = masked_peaks_fraction

        self.peak_encoder = PeakEncoder(
            d_model=self.d_model,  # size of embedding vectors
            min_mz_wavelength=0.001,
            max_mz_wavelength=10000,
            min_intensity_wavelength=1e-06,
            max_intensity_wavelength=1,
            learnable_wavelengths=False,
        )

        self.encoder = SpectrumTransformerEncoder(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,  # 1024,
            n_layers=self.n_layers,
            dropout=self.dropout,
            peak_encoder=self.peak_encoder,
        )

        # lol we don't have SpectrumTransformer here

        self.head_mz = nn.Sequential(
            nn.Linear(d_model, n_bins),
        )  # outputs n_bins logits for each peak
        self.head_I = nn.Sequential(
            nn.Linear(d_model, 1),
        )  # outputs float I value for each peak

        # losses
        self.loss_mz_bin = nn.CrossEntropyLoss(reduction="mean")
        self.loss_I = nn.MSELoss(reduction="mean")

        # metrics
        # TODO: add accuracy for mz bin prediction
        # TODO: add MAE for I prediction

    def get_peaks_mask(self, intensities):
        # TODO: may want the mask to be proportional to I?
        mask = torch.rand_like(intensities) < self.masked_peaks_fraction

        # ensure at least some peaks are masked
        if mask.long().sum() == 0:
            mask = self.get_peaks_mask(intensities)
        return mask

    def get_mz_bins(self, mz):
        return (
            ((mz - self.bin_mz_min) / (self.bin_mz_max - self.bin_mz_min) * self.n_bins)
            .floor()
            .long()
        )

    def forward(
        self,
        mzs: torch.Tensor,
        intensities: torch.Tensor,
    ):
        peak_embs, _ = self.encoder(mz_array=mzs, intensity_array=intensities)
        # drop global token
        peak_embs = peak_embs[:, 1:, :]
        return peak_embs

    # def ssl_step(self):
    #     # TODO: move here the repeated part of training & validation parts
    #     return

    def training_step(self, batch, batch_idx):
        mz = batch["mz_array"]
        I = batch["intensity_array"]

        # sample peak masks
        masks = self.get_peaks_mask(I)

        # prepare targets (bins & I of masked peaks)
        target_mz, target_I = mz[masks], I[masks]
        # transform mz into bins (target classes C \in [0, n_bins - 1])
        target_mz_bins = self.get_mz_bins(target_mz)

        # mask input peaks with 0 (before encoding)
        masked_mz = mz * (1 - masks.float())
        masked_I = I * (1 - masks.float())
        # get embeddings for all peaks
        peak_embs = self.forward(masked_mz, masked_I)
        # select only embeddings of masked peaks
        masked_peak_embs = peak_embs[masks]
        # predict masked peaks binned mz & I
        pred_mz_bins = self.head_mz(masked_peak_embs)
        pred_I = self.head_I(masked_peak_embs).squeeze(dim=-1)

        loss_mz_bin = self.loss_mz_bin(pred_mz_bins, target_mz_bins)
        loss_I = self.loss_I(pred_I, target_I)
        loss = loss_mz_bin + loss_I
        self.log("train_loss_mz_bin", loss_mz_bin.item())
        self.log("train_loss_I", loss_I.item())
        self.log("train_loss", loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        # Note: validation is now always partially random (mask sampling)
        # so not identical between epochs. May want to change it (how?).
        mz = batch["mz_array"]
        I = batch["intensity_array"]

        # sample peak masks
        masks = self.get_peaks_mask(I)

        # prepare targets (bins & I of masked peaks)
        target_mz, target_I = mz[masks], I[masks]
        # transform mz into bins (target classes C \in [0, n_bins - 1])
        target_mz_bins = self.get_mz_bins(target_mz)

        # mask input peaks with 0 (before encoding)
        masked_mz = mz * (1 - masks.float())
        masked_I = I * (1 - masks.float())
        # get embeddings for all peaks
        peak_embs = self.forward(masked_mz, masked_I)
        # select only embeddings of masked peaks
        masked_peak_embs = peak_embs[masks]
        # predict masked peaks binned mz & I
        pred_mz_bins = self.head_mz(masked_peak_embs)
        pred_I = self.head_I(masked_peak_embs).squeeze(dim=-1)

        loss_mz_bin = self.loss_mz_bin(pred_mz_bins, target_mz_bins)
        loss_I = self.loss_I(pred_I, target_I)
        loss = loss_mz_bin + loss_I
        self.log("val_loss_mz_bin", loss_mz_bin.item())
        self.log("val_loss_I", loss_I.item())
        self.log("val_loss", loss.item())
        return loss

    def configure_optimizers(
        self,
    ):
        """TODO."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.98)
        )
        self.lr_scheduler = CosineWarmupScheduler(
            optimizer,
            self.hparams.warmup_iters,
            self.hparams.cosine_schedule_period_iters,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "cosine_warmup",
            },
        }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.log("lr", self.lr_scheduler.get_last_lr()[0])


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Impelementation borrowed from https://github.com/Noble-Lab/casanovo.
    Learning rate scheduler with linear warm-up followed by cosine
    shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup_iters : int
        The number of iterations for the linear warm-up of the learning
        rate.
    cosine_schedule_period_iters : int
        The number of iterations for the cosine half period of the
        learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        cosine_schedule_period_iters: int,
    ):
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (
            1 + np.cos(np.pi * epoch / self.cosine_schedule_period_iters)
        )
        if epoch <= self.warmup_iters:
            lr_factor *= epoch / self.warmup_iters
        return lr_factor
