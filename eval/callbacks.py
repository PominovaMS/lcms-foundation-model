"""PyTorch Lightning callbacks for downstream linear probe evaluation."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torch import Tensor
from torchmetrics.functional import accuracy


class FineTuner(L.Callback):
    """
    Retrain-style probe: re-initializes and fully retrains the linear probe
    from scratch at the end of each SSL training epoch, then evaluates on probe_val.

    Trains until avg_loss < min_train_loss OR max n_epochs, whichever comes first.
    """

    def __init__(
        self,
        encoder_output_dim: int,
        num_classes: int,
        target_key: str,
        probe_train_loader,
        probe_val_loader,
        class_weights: Tensor | None = None,
        lr: float = 1e-2,
        n_epochs: int = 10,
        min_train_loss: float = 0.2,
        eval_every_n_epochs: int = 1,
        use_cls: bool = False,
    ) -> None:
        super().__init__()

        if class_weights is not None:
            assert class_weights.size(0) == num_classes

        self.encoder_output_dim = encoder_output_dim
        self.num_classes = num_classes
        self.target_key = target_key
        self.class_weights = class_weights
        self.lr = lr
        self.n_epochs = n_epochs
        self.min_train_loss = min_train_loss
        self.eval_every_n_epochs = eval_every_n_epochs
        self.use_cls = use_cls

        self.probe_train_loader = probe_train_loader
        self.probe_val_loader = probe_val_loader

    def _init_model_opt(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.finetuner = nn.Linear(self.encoder_output_dim, self.num_classes).to(
            pl_module.device
        )
        self.optimizer = torch.optim.Adam(self.finetuner.parameters(), lr=self.lr)

    def _encode_run(self, pl_module, run_mz, run_I):
        run_mz, run_I = run_mz.to(pl_module.device), run_I.to(pl_module.device)
        with torch.no_grad():
            cls_emb, peak_embs = pl_module.forward(run_mz, run_I)
            if self.use_cls:
                spec_embs = cls_emb  # (n_spectra, d_model)
            else:
                spec_embs = peak_embs.mean(dim=1)  # (n_spectra, d_model)
            spec_embs = spec_embs.unsqueeze(dim=0)  # (1, n_spectra, d)
            run_emb = spec_embs.mean(dim=1)  # (1, d)
        return run_emb

    def _step(self, pl_module, batch, train=True):
        runs_mz = batch["mz_array"]
        runs_I = batch["intensity_array"]
        targets = batch[self.target_key].to(pl_module.device)

        runs_emb = [
            self._encode_run(pl_module, runs_mz[i], runs_I[i])
            for i in range(len(runs_mz))
        ]
        embs = torch.cat(runs_emb, dim=0)

        preds = self.finetuner(embs)
        loss = F.cross_entropy(preds, targets, weight=self.class_weights)

        if train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        acc = accuracy(preds, targets, task="multiclass", num_classes=self.num_classes)

        if not train:
            n = 30
            sample_df = np.hstack(
                (
                    targets[:n].cpu().numpy()[:, None],
                    F.softmax(preds, dim=1)[:n].detach().cpu().numpy(),
                )
            )
            col_names = ["targets"] + [f"prob_{i}" for i in range(self.num_classes)]
            print(pd.DataFrame(sample_df, columns=col_names).to_string())

        return loss.detach(), acc.detach()

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        # Only retrain + evaluate the probe every N SSL epochs to save time
        if trainer.current_epoch % self.eval_every_n_epochs != 0:
            return
        self._init_model_opt(trainer, pl_module)

        was_training = pl_module.training
        pl_module.eval()
        self.finetuner.train()

        probe_epoch = 0
        avg_loss = float(
            "inf"
        )  # start high so the while condition enters on first iteration
        avg_acc = 0.0

        while (avg_loss > self.min_train_loss) and (
            self.n_epochs is None or probe_epoch < self.n_epochs
        ):
            total_loss, total_acc, n_batches = 0.0, 0.0, 0
            for batch in self.probe_train_loader:
                loss, acc = self._step(pl_module, batch, train=True)
                total_loss += float(loss)
                total_acc += float(acc)
                n_batches += 1

            avg_loss = total_loss / n_batches
            avg_acc = total_acc / n_batches
            print(f"Probe epoch {probe_epoch} loss: {avg_loss:.4f}  acc: {avg_acc:.4f}")
            probe_epoch += 1

        pl_module.log(
            "retrain_train_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False
        )
        pl_module.log(
            "retrain_train_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False
        )

        if was_training:
            pl_module.train()

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        # workaround for num_sanity_val_steps > 0
        if not hasattr(self, "finetuner"):
            self._init_model_opt(trainer, pl_module)

        # Skip evaluation on epochs where the probe wasn't retrained
        if trainer.current_epoch % self.eval_every_n_epochs != 0:
            return

        was_training = pl_module.training
        pl_module.eval()
        self.finetuner.eval()

        total_loss, total_acc, n_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in self.probe_val_loader:
                loss, acc = self._step(pl_module, batch, train=False)
                total_loss += float(loss)
                total_acc += float(acc)
                n_batches += 1

        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches

        pl_module.log(
            "retrain_val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False
        )
        pl_module.log(
            "retrain_val_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        if was_training:
            pl_module.train()


class OnlineFineTuner(L.Callback):
    """
    Online-style probe: initializes the linear probe once at the start of training,
    then takes one gradient step per SSL epoch and evaluates on probe_val.
    """

    def __init__(
        self,
        encoder_output_dim: int,
        num_classes: int,
        target_key: str,
        probe_train_loader,
        probe_val_loader=None,
        agg_type: str = "mean",
        class_weights: Tensor | None = None,
        probe_lr: float = 1e-2,
        use_cls: bool = False,
    ) -> None:
        super().__init__()

        if class_weights is not None:
            assert class_weights.size(0) == num_classes

        self.encoder_output_dim = encoder_output_dim
        self.num_classes = num_classes
        self.target_key = target_key
        self.agg_type = agg_type
        self.class_weights = class_weights
        self.probe_lr = probe_lr
        self.use_cls = use_cls

        self.probe_train_loader = probe_train_loader
        self.probe_val_loader = probe_val_loader

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        pl_module.online_finetuner = nn.Linear(
            self.encoder_output_dim, self.num_classes
        ).to(pl_module.device)
        self.optimizer = torch.optim.Adam(
            pl_module.online_finetuner.parameters(), lr=self.probe_lr
        )

    def _encode_run(self, pl_module, run_mz, run_I):
        run_mz, run_I = run_mz.to(pl_module.device), run_I.to(pl_module.device)
        with torch.no_grad():
            cls_emb, peak_embs = pl_module.forward(run_mz, run_I)
            if self.use_cls:
                spec_embs = cls_emb  # (n_spectra, d_model)
            else:
                spec_embs = peak_embs.mean(dim=1)  # (n_spectra, d_model)
            spec_embs = spec_embs.unsqueeze(dim=0)  # (1, n_spectra, d)
            if self.agg_type == "mean":
                run_emb = spec_embs.mean(dim=1)  # (1, d)
        return run_emb

    def _step(self, pl_module, batch, train=True):
        runs_mz = batch["mz_array"]
        runs_I = batch["intensity_array"]
        targets = batch[self.target_key].to(pl_module.device)

        runs_emb = [
            self._encode_run(pl_module, runs_mz[i], runs_I[i])
            for i in range(len(runs_mz))
        ]
        embs = torch.cat(runs_emb, dim=0)

        preds = pl_module.online_finetuner(embs)
        loss = F.cross_entropy(preds, targets, weight=self.class_weights)

        if train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        acc = accuracy(preds, targets, task="multiclass", num_classes=self.num_classes)

        if not train:
            n = 30
            sample_df = np.hstack(
                (
                    targets[:n].cpu().numpy()[:, None],
                    F.softmax(preds, dim=1)[:n].detach().cpu().numpy(),
                )
            )
            col_names = ["targets"] + [f"prob_{i}" for i in range(self.num_classes)]
            print(pd.DataFrame(sample_df, columns=col_names).to_string())

        return loss.detach(), acc.detach()

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        was_training = pl_module.training
        pl_module.eval()
        pl_module.online_finetuner.train()

        total_loss, total_acc, n_batches = 0.0, 0.0, 0
        for batch in self.probe_train_loader:
            loss, acc = self._step(pl_module, batch, train=True)
            total_loss += float(loss)
            total_acc += float(acc)
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches

        pl_module.log(
            "online_train_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False
        )
        pl_module.log(
            "online_train_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=False
        )

        if was_training:
            pl_module.train()

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        was_training = pl_module.training
        pl_module.eval()
        pl_module.online_finetuner.eval()

        total_loss, total_acc, n_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in self.probe_val_loader:
                loss, acc = self._step(pl_module, batch, train=False)
                total_loss += float(loss)
                total_acc += float(acc)
                n_batches += 1

        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches

        pl_module.log(
            "online_val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False
        )
        pl_module.log(
            "online_val_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        if was_training:
            pl_module.train()
