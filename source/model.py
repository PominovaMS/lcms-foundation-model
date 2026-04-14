import numpy as np
import pandas as pd  # DEBUG
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as L
from depthcharge.encoders import PeakEncoder, PositionalEncoder
from depthcharge.transformers import SpectrumTransformerEncoder
from .scheduler import CosineWarmupScheduler

# from IPython.display import clear_output # DEBUG
# pd.set_option('display.max_rows', 500) # DEBUG


# <01/01/26 TODO: update model
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
        mask_proportional=True,
        mz_label_sigma=0.0,
        lr=5e-4,
        warmup_iters=1000,
        cosine_schedule_period_iters=32000,
        optimizer_type="adam",
        weight_decay=0.0,
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
        self.mask_proportional = mask_proportional
        self.mz_label_sigma = mz_label_sigma

        # peak_encoder (that is passed to the SpectrumTransformerEncoder)
        # changed to also apply (add) positional encodings
        self.peak_encoder = nn.Sequential(
            PeakEncoder(
                d_model=self.d_model,
                min_mz_wavelength=0.001,
                max_mz_wavelength=10000,
                min_intensity_wavelength=1e-06,
                max_intensity_wavelength=1,
                learnable_wavelengths=False,
            ),
            # PositionalEncoder(
            #     d_model=self.d_model,
            #     min_wavelength=1,
            #     max_wavelength=10000,
            # ),
        )

        self.encoder = SpectrumTransformerEncoder(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,  # 1024,
            n_layers=self.n_layers,
            dropout=self.dropout,
            peak_encoder=self.peak_encoder,
        )

        self.head_mz = nn.Sequential(
            nn.Linear(d_model, n_bins),
        )  # outputs n_bins logits for each peak
        self.head_I = nn.Sequential(
            nn.Linear(d_model, 1),
        )  # outputs float I value for each peak

        # losses
        self.loss_mz_bin = nn.CrossEntropyLoss(reduction="mean", ignore_index=-1)
        self.loss_I = nn.MSELoss(reduction="mean")
        # metrics
        self.train_accuracy_mz_bin = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.n_bins, ignore_index=-1
        )
        self.val_accuracy_mz_bin = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=self.n_bins, ignore_index=-1
        )
        self.train_mae_I = torchmetrics.regression.MeanAbsoluteError()
        self.val_mae_I = torchmetrics.regression.MeanAbsoluteError()

    def get_peaks_mask(self, intensities, proportional=False, generator=None):
        if proportional:
            k = int(intensities.size(1) * self.masked_peaks_fraction)
            mask = torch.zeros_like(intensities, dtype=torch.bool)
            # FIXME: assume we have no zero rows (= spectra with no peaks)

            # compute sampling weights w
            I_mean = intensities.sum(dim=1) / (intensities != 0).sum(
                dim=1
            )  # mean I of non-zero peaks
            w = (
                intensities + (intensities == 0).float() * I_mean[:, None]
            )  # weight 0s by mean I
            # sample k indices without replacement, weighted by w
            idx = torch.multinomial(
                w, num_samples=k, replacement=False, generator=generator
            )
            mask = mask.scatter(
                dim=1, index=idx.to(dtype=torch.int64), value=True
            )  # value to write into mask (True)

        else:
            mask = (
                torch.rand(
                    intensities.shape,
                    device=intensities.device,
                    dtype=intensities.dtype,
                    generator=generator,
                )
                < self.masked_peaks_fraction
            )
        return mask

    def get_mz_bins(self, mz):
        # every peak with mz > bin_mz_max will belong to max bin
        mz = mz.clamp(0, self.bin_mz_max - 1)
        mz_binned = (
            ((mz - self.bin_mz_min) / (self.bin_mz_max - self.bin_mz_min) * self.n_bins)
            .floor()
            .long()
        )
        mz_binned[mz < self.bin_mz_min] = -1
        return mz_binned

    def get_soft_targets(self, target_mz_bins):
        """Create Gaussian soft target distributions centered on true bins."""
        bins = torch.arange(self.n_bins, device=target_mz_bins.device, dtype=torch.float)
        centers = target_mz_bins.float().unsqueeze(-1)  # (N, 1)
        soft = torch.exp(-((bins - centers) ** 2) / (2 * self.mz_label_sigma**2))
        soft = soft / soft.sum(dim=-1, keepdim=True)
        return soft

    def compute_mz_loss(self, pred_mz_bins, target_mz_bins):
        """Compute m/z loss using either hard or soft targets."""
        if self.mz_label_sigma > 0:
            valid = target_mz_bins != -1
            if valid.sum() == 0:
                return torch.tensor(0.0, device=pred_mz_bins.device)
            soft_targets = self.get_soft_targets(target_mz_bins[valid])
            log_probs = F.log_softmax(pred_mz_bins[valid], dim=-1)
            return -(soft_targets * log_probs).sum(dim=-1).mean()
        else:
            return self.loss_mz_bin(pred_mz_bins, target_mz_bins)

    def forward(
        self,
        mzs: torch.Tensor,
        intensities: torch.Tensor,
    ):
        embs, _ = self.encoder(mz_array=mzs, intensity_array=intensities)
        cls_emb = embs[:, 0, :]    # (batch, d_model) — spectrum-level embedding
        peak_embs = embs[:, 1:, :]  # (batch, n_peaks, d_model)
        return cls_emb, peak_embs

    # def ssl_step(self):
    #     # TODO: move here the repeated part of training & validation parts
    #     return

    def training_step(self, batch, batch_idx):
        mz = batch["mz_array"]
        I = batch["intensity_array"]

        # sample peak masks
        masks = self.get_peaks_mask(I, proportional=self.mask_proportional)

        # prepare targets (bins & I of masked peaks)
        target_mz, target_I = mz[masks], I[masks]
        # transform mz into bins (target classes C \in [0, n_bins - 1])
        target_mz_bins = self.get_mz_bins(target_mz)

        # mask input peaks with 0 (before encoding)
        masked_mz = mz * (1 - masks.float())
        # masked_I = I * (1 - masks.float()) # FIX: not mask intensities, only mz

        # get embeddings for all peaks
        _cls_emb, peak_embs = self.forward(masked_mz, I)
        # select only embeddings of masked peaks
        masked_peak_embs = peak_embs[masks]
        # predict masked peaks binned mz & I
        pred_mz_bins = self.head_mz(masked_peak_embs)
        pred_I = self.head_I(masked_peak_embs).squeeze(dim=-1)

        loss_mz_bin = self.compute_mz_loss(pred_mz_bins, target_mz_bins)
        loss_I = self.loss_I(pred_I, target_I)
        loss = loss_mz_bin  # + loss_I
        self.log("train_loss_mz_bin", loss_mz_bin.item())
        # self.log("train_loss_I", loss_I.item())
        self.log("train_loss", loss.item())
        # Accuracy metric for mz bin prediction
        acc_mz_bin = self.train_accuracy_mz_bin(pred_mz_bins, target_mz_bins)
        self.log(
            "train_acc_mz_bin",
            acc_mz_bin.item(),
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        # MAE metric for intensity prediction
        # mae_I = self.train_mae_I(pred_I, target_I)
        # self.log("train_mae_I", mae_I.item(), prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # Seed mask sampling for deterministic validation across epochs
        # (each batch gets a different but reproducible mask)
        mz = batch["mz_array"]
        I = batch["intensity_array"]

        # sample peak masks
        generator = torch.Generator(device=I.device)
        generator.manual_seed(42 + batch_idx)
        masks = self.get_peaks_mask(
            I, proportional=self.mask_proportional, generator=generator
        )

        # prepare targets (bins & I of masked peaks)
        target_mz, target_I = mz[masks], I[masks]
        # transform mz into bins (target classes C \in [0, n_bins - 1])
        target_mz_bins = self.get_mz_bins(target_mz)

        # mask input peaks with 0 (before encoding)
        masked_mz = mz * (1 - masks.float())
        # masked_I = I * (1 - masks.float()) # FIX: not mask intensities, only mz

        # get embeddings for all peaks
        _cls_emb, peak_embs = self.forward(masked_mz, I)
        # select only embeddings of masked peaks
        masked_peak_embs = peak_embs[masks]
        # predict masked peaks binned mz & I
        pred_mz_bins = self.head_mz(masked_peak_embs)
        pred_I = self.head_I(masked_peak_embs).squeeze(dim=-1)

        loss_mz_bin = self.compute_mz_loss(pred_mz_bins, target_mz_bins)
        # loss_I = self.loss_I(pred_I, target_I)
        loss = loss_mz_bin  # + loss_I
        self.log("val_loss_mz_bin", loss_mz_bin.item())
        # self.log("val_loss_I", loss_I.item())
        self.log("val_loss", loss.item())
        # Accuracy metric for mz bin prediction
        acc_mz_bin = self.val_accuracy_mz_bin(pred_mz_bins, target_mz_bins)
        self.log(
            "val_acc_mz_bin",
            acc_mz_bin.item(),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        # MAE metric for intensity prediction
        # mae_I = self.val_mae_I(pred_I, target_I)
        # self.log("val_mae_I", mae_I.item(), prog_bar=True, on_step=False, on_epoch=True)

        # DEBUG outputs
        # i = 0
        # mz_i, I_i = mz[i], I[i]
        # mask_i = masks[i]
        # target_mz_i, target_I_i = mz_i[masks_i], I_i[masks_i]
        # target_mz_bins = self.get_mz_bins(target_mz)

        n = 30
        mz_bins_true, I_true = (
            target_mz_bins[:n].cpu().numpy(),
            target_I[:n].cpu().numpy(),
        )
        mz_bins_pred, I_pred = (
            pred_mz_bins[:n].argmax(dim=1).cpu().numpy(),
            pred_I[:n].cpu().numpy(),
        )
        sample_df = np.column_stack(
            (
                mz_bins_true.ravel(),
                I_true.ravel(),
                mz_bins_pred.ravel(),
                # I_pred.ravel()
            )
        )
        sample_df = pd.DataFrame(
            sample_df,
            columns=[
                "mz_bins_true",
                "I_true",
                "mz_bins_pred",
                # "I_pred"
            ],
        )
        print(sample_df.to_string())

        return loss

    def configure_optimizers(
        self,
    ):
        """TODO."""
        if self.hparams.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                betas=(0.9, 0.98),
                weight_decay=self.hparams.weight_decay,
            )
        else:
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
