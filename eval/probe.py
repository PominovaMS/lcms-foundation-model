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


def encode_dataset(model, loader, target_key, device):
    """Encode every run in a loader ONCE to a fixed (N, d) embedding matrix.

    The encoder is frozen, so a run's embedding never changes — precomputing it
    once avoids re-running the transformer on every probe epoch (the previous
    behavior, which made the probe ~n_epochs times slower than necessary).
    Returns ``(embeddings, targets)`` on ``device``.
    """
    embs, targets = [], []
    for batch in loader:
        runs_mz = batch["mz_array"]
        runs_I = batch["intensity_array"]
        for i in range(len(runs_mz)):
            embs.append(encode_run(model, runs_mz[i], runs_I[i], device))
        targets.append(batch[target_key].to(device))
    return torch.cat(embs, dim=0), torch.cat(targets, dim=0)


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

    Run embeddings are computed once up front (the encoder is frozen), then the
    linear probe trains on those cached vectors. Returns ``(val_acc, val_loss)``.
    """
    device = device or next(model.parameters()).device
    model.eval()

    # --- encode all runs once (the expensive part, done a single time) ---
    X_train, y_train = encode_dataset(model, probe_train_loader, target_key, device)
    X_val, y_val = encode_dataset(model, probe_val_loader, target_key, device)

    probe = nn.Linear(d_model, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    # --- train the probe on cached embeddings (full-batch; it's just a linear map) ---
    probe.train()
    avg_loss = float("inf")  # enter the loop on the first iteration
    probe_epoch = 0
    while (avg_loss > min_train_loss) and (probe_epoch < n_epochs):
        optimizer.zero_grad()
        preds = probe(X_train)
        loss = F.cross_entropy(preds, y_train)
        loss.backward()
        optimizer.step()

        avg_loss = float(loss.detach())
        avg_acc = float(
            accuracy(preds, y_train, task="multiclass", num_classes=num_classes)
        )
        logger.info(f"Probe epoch {probe_epoch} loss: {avg_loss:.4f}  acc: {avg_acc:.4f}")
        probe_epoch += 1

    # --- evaluate the probe ---
    probe.eval()
    with torch.no_grad():
        preds = probe(X_val)
        val_loss = float(F.cross_entropy(preds, y_val))
        val_acc = float(
            accuracy(preds, y_val, task="multiclass", num_classes=num_classes)
        )
    return val_acc, val_loss
