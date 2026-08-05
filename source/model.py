import warnings

import numpy as np
import pandas as pd  # DEBUG
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as L
from depthcharge.encoders import PeakEncoder, PositionalEncoder
from depthcharge.transformers import SpectrumTransformerEncoder

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
        lr=5e-4,
        warmup_iters=1000,
        total_steps=32000,
        auto_total_steps=True,
        div_factor=25.0,
        final_div_factor=1e4,
        cosine_schedule_period_iters=None,
    ):
        super().__init__()
        if cosine_schedule_period_iters is not None:
            # Checkpoints trained before the OneCycleLR switch saved the LR schedule
            # length under this name. `load_from_checkpoint` replays saved hparams as
            # keyword arguments, so accepting the old name is what keeps those
            # checkpoints loadable (eval/probe_checkpoint.py reads them).
            # noqa is for ruff, which can't see that the reassignment is consumed:
            # `save_hyperparameters` reads it back out of the frame locals below.
            total_steps = cosine_schedule_period_iters  # noqa: F841
        # `save_hyperparameters` reads the current frame locals, so it picks up the
        # remapped `total_steps` above; the legacy name is dropped so hparams keep a
        # single source of truth for the schedule length.
        self.save_hyperparameters(ignore=["cosine_schedule_period_iters"])

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
        # Padding peaks have intensity exactly 0 (real peaks are strictly positive
        # after intensity scaling), so `intensities != 0` is the real-peak mask.
        is_real = intensities != 0
        if proportional:
            k = int(intensities.size(1) * self.masked_peaks_fraction)
            mask = torch.zeros_like(intensities, dtype=torch.bool)

            # Sampling weights: real peaks weighted by intensity, padding given a
            # tiny epsilon. `torch.multinomial(replacement=False)` requires >= k
            # strictly-positive weights per row, so the epsilon guarantees the call
            # never errors when a spectrum has fewer than k real peaks. Because the
            # epsilon is negligible next to real intensities, padding is only ever
            # drawn once the real peaks are exhausted — and we drop it below.
            w = intensities + (~is_real).float() * 1e-9
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
        # Never mask padding: drop any padding positions that were selected so they
        # don't leak into the targets/loss as spurious bin-0 predictions.
        mask &= is_real
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
        masks = self.get_peaks_mask(I, proportional=self.mask_proportional)

        # prepare targets (bins & I of masked peaks)
        target_mz, target_I = mz[masks], I[masks]
        # transform mz into bins (target classes C \in [0, n_bins - 1])
        target_mz_bins = self.get_mz_bins(target_mz)

        # mask input peaks with 0 (before encoding)
        masked_mz = mz * (1 - masks.float())
        # masked_I = I * (1 - masks.float()) # FIX: not mask intensities, only mz

        # get embeddings for all peaks
        # peak_embs = self.forward(masked_mz, masked_I)
        peak_embs = self.forward(masked_mz, I)  # FIX: not mask intensities, only mz
        # select only embeddings of masked peaks
        masked_peak_embs = peak_embs[masks]
        # predict masked peaks binned mz & I
        pred_mz_bins = self.head_mz(masked_peak_embs)
        pred_I = self.head_I(masked_peak_embs).squeeze(dim=-1)

        loss_mz_bin = self.loss_mz_bin(pred_mz_bins, target_mz_bins)
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
        # peak_embs = self.forward(masked_mz, masked_I)
        peak_embs = self.forward(masked_mz, I)  # FIX: not mask intensities, only mz
        # select only embeddings of masked peaks
        masked_peak_embs = peak_embs[masks]
        # predict masked peaks binned mz & I
        pred_mz_bins = self.head_mz(masked_peak_embs)
        pred_I = self.head_I(masked_peak_embs).squeeze(dim=-1)

        loss_mz_bin = self.loss_mz_bin(pred_mz_bins, target_mz_bins)
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

    def _resolve_total_steps(self):
        """How many optimizer steps the run will actually take.

        ``OneCycleLR`` raises once stepped past ``total_steps``, so this has to be
        right, not approximately right. Lightning's ``estimated_stepping_batches`` is
        the authority — it knows the real dataloader length plus
        ``accumulate_grad_batches``, ``limit_train_batches``, ``max_steps`` and the
        device count, none of which a formula in ``train.py`` tracks.

        It is not always usable: it returns ``inf``/``max_steps`` (i.e. ``-1``) for an
        unsized ``IterableDataset``, and there is no trainer at all when
        ``configure_optimizers`` is called directly. Both fall back to the
        ``total_steps`` hparam, which ``train.py`` computes exactly for those cases.
        An explicit ``optimizer.total_steps`` in the config sets
        ``auto_total_steps=False`` and always wins.
        """
        fallback = self.hparams.total_steps
        if not self.hparams.auto_total_steps or self._trainer is None:
            return fallback, "hparams.total_steps"

        estimated = self.trainer.estimated_stepping_batches
        if not np.isfinite(estimated) or estimated <= 0:
            # Unsized IterableDataset with no --max_steps.
            return fallback, "hparams.total_steps (trainer estimate unavailable)"
        return int(estimated), "trainer.estimated_stepping_batches"

    def configure_optimizers(
        self,
    ):
        """Adam + a one-cycle LR schedule (warmup to ``lr``, then cosine anneal)."""
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.98)
        )
        total_steps, source = self._resolve_total_steps()
        print(f"OneCycleLR total_steps={total_steps} (from {source})")
        # Warmup is configured in steps (the CLI and slurm scripts speak steps);
        # OneCycleLR wants it as a fraction of the cycle.
        raw_pct = self.hparams.warmup_iters / total_steps
        pct_start = min(max(raw_pct, 1e-3), 0.5)
        if pct_start != raw_pct:
            warnings.warn(
                f"warmup_iters={self.hparams.warmup_iters} is "
                f"{raw_pct:.4g} of total_steps={total_steps}; "
                f"clamped pct_start to {pct_start:g} (must be in (0, 1)).",
                stacklevel=2,
            )
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
            div_factor=self.hparams.div_factor,
            final_div_factor=self.hparams.final_div_factor,
            # Adam exposes `betas`, not `momentum`: left on, OneCycleLR would cycle
            # betas[0] between 0.85 and 0.95 and silently override the (0.9, 0.98)
            # pinned above.
            cycle_momentum=False,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "one_cycle",
            },
        }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.log("lr", self.lr_scheduler.get_last_lr()[0])
