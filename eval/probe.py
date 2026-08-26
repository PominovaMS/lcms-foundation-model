"""Standalone (trainer-free) linear probe over run-level embeddings.

Evaluates a *pretrained* checkpoint without running any SSL training or a PyTorch
Lightning ``Trainer``.

The encoder is frozen (``torch.no_grad`` in :func:`encode_run`); a fresh
``nn.Linear`` probe is trained on run-level embeddings until the average train
loss drops below ``min_train_loss`` (or ``n_epochs`` is reached), then evaluated
on the probe validation set.

Two properties matter for comparing checkpoints against each other:

*Reproducibility.* :func:`fit_probe` seeds the probe initialisation, and
:func:`run_retrain_probe` fits it ``n_repeats`` times over the same cached
embeddings, so a run reports mean ± std instead of one draw from an unseeded RNG.

*Collapse visibility.* A probe that predicts a single class can still post a
respectable micro accuracy when the classes are unbalanced. Every fit therefore
also reports ``val_acc_macro``, ``n_pred_classes``, and the ``majority_acc`` /
``random_acc`` baselines it has to beat.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy

logger = logging.getLogger(__name__)


def encode_run(model, run_mz, run_I, device, chunk_size: int = 512):
    """Frozen run-level embedding: mean-pool peaks, then mean-pool spectra.

    Encodes ``chunk_size`` spectra at a time. A whole LC-MS run is thousands of
    MS1 spectra, and pushing it through the transformer in a single forward needs
    tens of GB: the attention term alone is ``nhead * (n_peaks+1)**2 * 4B`` per
    spectrum (~1.3 MB at 200 peaks) and it does NOT shrink with ``d_model``.

    Chunking is exact, not an approximation. ``SpectrumTransformerEncoder`` puts
    spectra on the *batch* dimension and attends only over the peaks within a
    spectrum, so splitting dim 0 yields identical per-spectrum embeddings; only
    the floating-point summation order of the final mean changes. Dropout is off
    (callers set ``model.eval()``), so chunk boundaries cannot alter the result.
    """
    n = run_mz.shape[0]
    total = None
    with torch.no_grad():
        for start in range(0, n, chunk_size):
            mz = run_mz[start : start + chunk_size].to(device)
            I = run_I[start : start + chunk_size].to(device)
            spec_embs = model.forward(mz, I).mean(dim=1)  # (chunk, d)
            chunk_sum = spec_embs.sum(dim=0)  # (d,)
            total = chunk_sum if total is None else total + chunk_sum
    return (total / n).unsqueeze(0)  # (1, d)


def encode_dataset(model, loader, target_key, device, chunk_size: int = 512):
    """Encode every run in a loader ONCE to a fixed (N, d) embedding matrix.

    The encoder is frozen, so a run's embedding never changes — precomputing it
    once avoids re-running the transformer on every probe epoch (the previous
    behavior, which made the probe ~n_epochs times slower than necessary).
    Returns ``(embeddings, targets)`` on ``device``.

    Logs the run-size distribution: how many spectra a run holds is what decides
    whether ``chunk_size`` is sensible, and it is recorded nowhere else.
    """
    embs, targets, sizes = [], [], []
    for batch in loader:
        runs_mz = batch["mz_array"]
        runs_I = batch["intensity_array"]
        for i in range(len(runs_mz)):
            sizes.append(int(runs_mz[i].shape[0]))
            embs.append(
                encode_run(model, runs_mz[i], runs_I[i], device, chunk_size=chunk_size)
            )
        targets.append(batch[target_key].to(device))

    if sizes:
        sizes_t = torch.tensor(sizes, dtype=torch.float)
        logger.info(
            f"Encoded {len(sizes)} runs, {sum(sizes):,} spectra "
            f"(min {min(sizes)} / median {int(sizes_t.median())} / "
            f"max {max(sizes)} per run, chunk_size={chunk_size})"
        )
    return torch.cat(embs, dim=0), torch.cat(targets, dim=0)


def baseline_metrics(y_train, y_val, num_classes: int) -> dict[str, float]:
    """Scores the probe has to beat. Seed-independent, so computed once per run.

    ``majority_acc`` is the accuracy on val of always predicting the most frequent
    *train* class — the exact thing a collapsed probe does. When train and val
    class distributions disagree it can fall *below* ``random_acc``, which is
    precisely the failure mode worth catching.
    """
    majority_class = int(torch.bincount(y_train, minlength=num_classes).argmax())
    return {
        "majority_acc": float((y_val == majority_class).float().mean()),
        "random_acc": 1.0 / num_classes,
    }


def fit_probe(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes: int,
    *,
    seed: int = 0,
    lr: float = 1e-2,
    n_epochs: int = 100,
    min_train_loss: float = 0.3,
) -> dict[str, float]:
    """Fit one linear probe on cached embeddings and evaluate it.

    Takes plain tensors — no model, no DataLoader — so it is cheap to repeat and
    straightforward to unit-test. ``seed`` fixes the ``nn.Linear`` initialisation,
    which is the only stochastic element: the probe trains full-batch, so data
    ordering does not enter into it.

    Note this seeds the *global* torch RNG, which is fine in the eval path
    (``eval/probe_checkpoint.py`` does nothing else stochastic afterwards) but
    would need a local generator if it were ever called mid-training.
    """
    device = X_train.device
    torch.manual_seed(seed)
    probe = nn.Linear(X_train.shape[1], num_classes).to(device)
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
        logger.debug(
            f"Probe epoch {probe_epoch} loss: {avg_loss:.4f}  acc: {avg_acc:.4f}"
        )
        probe_epoch += 1

    # --- evaluate the probe ---
    probe.eval()
    with torch.no_grad():
        preds = probe(X_val)
        return {
            "val_loss": float(F.cross_entropy(preds, y_val)),
            "val_acc": float(
                accuracy(preds, y_val, task="multiclass", num_classes=num_classes)
            ),
            "val_acc_macro": float(
                accuracy(
                    preds,
                    y_val,
                    task="multiclass",
                    num_classes=num_classes,
                    average="macro",
                )
            ),
            # 1 == the probe predicts a single class for every run, i.e. collapsed.
            "n_pred_classes": float(preds.argmax(dim=1).unique().numel()),
            # Pins at n_epochs when min_train_loss was never reached -> underfit,
            # which is a different problem than collapse.
            "probe_epochs": float(probe_epoch),
            "train_loss": avg_loss,
        }


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
    seed: int = 0,
    n_repeats: int = 3,
    chunk_size: int = 512,
) -> dict[str, float]:
    """Train fresh linear probes on the frozen encoder and evaluate them.

    Run embeddings are computed once up front (the encoder is frozen), then
    ``n_repeats`` probes are fit on those cached vectors with seeds
    ``seed, seed+1, ...``. ``chunk_size`` bounds encoder memory (see
    :func:`encode_run`) and cannot change the numbers, only whether they compute.
    Returns a flat dict of ``<metric>`` (mean) and
    ``<metric>_std`` across repeats, plus the seed-independent baselines from
    :func:`baseline_metrics`.
    """
    device = device or next(model.parameters()).device
    model.eval()

    # --- encode all runs once (the expensive part, done a single time) ---
    X_train, y_train = encode_dataset(
        model, probe_train_loader, target_key, device, chunk_size=chunk_size
    )
    X_val, y_val = encode_dataset(
        model, probe_val_loader, target_key, device, chunk_size=chunk_size
    )

    runs = [
        fit_probe(
            X_train,
            y_train,
            X_val,
            y_val,
            num_classes,
            seed=seed + i,
            lr=lr,
            n_epochs=n_epochs,
            min_train_loss=min_train_loss,
        )
        for i in range(n_repeats)
    ]

    results = {}
    for key in runs[0]:
        vals = torch.tensor([r[key] for r in runs])
        results[key] = float(vals.mean())
        results[f"{key}_std"] = float(vals.std(unbiased=False))
    results.update(baseline_metrics(y_train, y_val, num_classes))

    # --- flag the two signatures of a collapsed probe ---
    if any(r["n_pred_classes"] == 1 for r in runs):
        logger.warning(
            "COLLAPSE: at least one probe repeat predicted a single class for every "
            "run. The reported accuracy reflects the class balance, not the encoder."
        )
    if results["val_acc_macro"] <= results["random_acc"]:
        logger.warning(
            f"COLLAPSE: macro accuracy {results['val_acc_macro']:.4f} does not beat "
            f"the random rate {results['random_acc']:.4f} — the representation "
            f"carries no usable organism signal."
        )
    if results["probe_epochs"] >= n_epochs:
        logger.warning(
            f"Probe never reached min_train_loss={min_train_loss} within "
            f"{n_epochs} epochs (train loss {results['train_loss']:.4f}); it is "
            f"underfit rather than collapsed."
        )
    return results
