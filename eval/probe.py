"""Standalone (trainer-free) linear probe over run-level embeddings.

Lifted from ``eval/callbacks.py::FineTuner`` so a *pretrained* checkpoint can be
evaluated without running any SSL training or a PyTorch Lightning ``Trainer``.

The encoder is frozen (``torch.no_grad`` in :func:`encode_run`); a fresh
``nn.Linear`` probe is trained on run-level embeddings until the average train
loss drops below ``min_train_loss`` (or ``n_epochs`` is reached), then evaluated
on the probe validation set. This mirrors the retrain-style probe reported as
``retrain_val_acc`` during co-trained SSL evaluation.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy

logger = logging.getLogger(__name__)


def encode_run(model, run_mz, run_I, device):
    """Frozen run-level embedding: mean-pool peaks, then mean-pool spectra.

    Matches ``FineTuner._encode_run`` (eval/callbacks.py) so probe numbers are
    comparable to the co-trained SSL evaluation.
    """
    run_mz, run_I = run_mz.to(device), run_I.to(device)
    with torch.no_grad():
        peak_embs = model.forward(run_mz, run_I)
        spec_embs = peak_embs.mean(dim=1)
        spec_embs = spec_embs.unsqueeze(dim=0)  # (1, T, d)
        run_emb = spec_embs.mean(dim=1)  # (1, d)
    return run_emb


def _embed_batch(model, probe, batch, target_key, device):
    runs_mz = batch["mz_array"]
    runs_I = batch["intensity_array"]
    targets = batch[target_key].to(device)

    runs_emb = [
        encode_run(model, runs_mz[i], runs_I[i], device) for i in range(len(runs_mz))
    ]
    embs = torch.cat(runs_emb, dim=0)
    return probe(embs), targets


def run_retrain_probe(
    model,
    probe_train_loader,
    probe_val_loader,
    d_model: int,
    num_classes: int,
    target_key: str = "label",
    lr: float = 1e-2,
    n_epochs: int = 100,
    min_train_loss: float = 0.3,
    device=None,
) -> tuple[float, float]:
    """Train a fresh linear probe on the frozen encoder and evaluate it.

    Returns ``(val_acc, val_loss)`` averaged over the probe validation loader.
    """
    device = device or next(model.parameters()).device
    model.eval()

    probe = nn.Linear(d_model, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    # --- train the probe ---
    probe.train()
    avg_loss = float("inf")  # enter the loop on the first iteration
    avg_acc = 0.0
    probe_epoch = 0
    while (avg_loss > min_train_loss) and (probe_epoch < n_epochs):
        total_loss, total_acc, n_batches = 0.0, 0.0, 0
        for batch in probe_train_loader:
            preds, targets = _embed_batch(model, probe, batch, target_key, device)
            loss = F.cross_entropy(preds, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            acc = accuracy(
                preds, targets, task="multiclass", num_classes=num_classes
            )
            total_loss += float(loss.detach())
            total_acc += float(acc.detach())
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches
        logger.info(
            f"Probe epoch {probe_epoch} loss: {avg_loss:.4f}  acc: {avg_acc:.4f}"
        )
        probe_epoch += 1

    # --- evaluate the probe ---
    probe.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in probe_val_loader:
            preds, targets = _embed_batch(model, probe, batch, target_key, device)
            loss = F.cross_entropy(preds, targets)
            acc = accuracy(preds, targets, task="multiclass", num_classes=num_classes)
            total_loss += float(loss)
            total_acc += float(acc)
            n_batches += 1

    val_loss = total_loss / n_batches
    val_acc = total_acc / n_batches
    return val_acc, val_loss
