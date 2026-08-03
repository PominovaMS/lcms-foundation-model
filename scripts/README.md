# scripts

## Data-diversity scaling experiment

Measure how the foundation model's downstream performance changes as **more
diverse PRIDE data** is added to self-supervised (SSL) pretraining. The model is
pretrained on `pride_data` (adding one PXD accession at a time, cumulatively) and
evaluated on the `abele_data` genus-classification task with a frozen-encoder
linear probe, on a **fixed** abele split so stages are comparable.

Downloading/converting the PRIDE `.raw` files is **not** part of this experiment —
it assumes each dataset is already on disk as `<root>/<ACCESSION>/mzml/*.mzML`.

### 1. Build cumulative training sets — `build_stage.py`

Symlinks the already-ingested mzML into a sequence of cumulative training sets.
No downloads; purely offline.

```bash
python scripts/build_stage.py \
    --root /mnt/data/shared/lc_ms_foundation/pride_data \
    --accessions PXD014877 PXD012345 PXD067890 \
    --out-root /mnt/data/shared/lc_ms_foundation/training_sets/sweep \
    --cumulative --val-frac 0.1 --seed 0
```

Produces:

```
training_sets/sweep/
  stage01/{train_mzml,val_mzml,accessions.txt}   # PXD014877
  stage02/{train_mzml,val_mzml,accessions.txt}   # PXD014877 + PXD012345
  stage03/{train_mzml,val_mzml,accessions.txt}   # + PXD067890
```

The `--accessions` order is the **diversity order** — put the most different
datasets later to see where added diversity helps or hurts. The train/val split is
a stable hash of `seed:filename`, so a file keeps its assignment across stages and
earlier stages are never reshuffled. Re-running is idempotent (use `--force` to
replace existing symlinks).

### 2. Pretrain the foundation model per stage

Each stage dir is consumed directly by `source/train.py`. Give every stage a
distinct `--run_name` so its logs + checkpoints land in their own directory (and
stages don't overwrite each other):

```bash
cd source && python train.py \
    --data_dir /mnt/data/shared/lc_ms_foundation/training_sets/sweep/stage03 \
    --config ../config.yaml \
    --run_name stage03
```

Checkpoints (including a stable `last.ckpt`) are written under
`train_checkpoints/foundation_model/lightning_logs/<run_name>/checkpoints/`.

### 3. Evaluate each checkpoint downstream — `eval/probe_checkpoint.py`

Loads a pretrained checkpoint, freezes the encoder, trains a fresh linear probe on
the abele genus task, and reports validation accuracy. Point `--results_csv` at the
**same file** for every stage to accumulate the scaling curve. Keep
`--n_probe_genera` / `--n_ssl_top` identical across stages so the probe split is fixed.

```bash
python eval/probe_checkpoint.py \
    --ckpt_path train_checkpoints/foundation_model/lightning_logs/stage03/checkpoints/last.ckpt \
    --data_dir /mnt/data/shared/lc_ms_foundation/abele_data/mzml \
    --meta_path /mnt/data/shared/lc_ms_foundation/abele_data/all_abele_metadata.csv \
    --config config.yaml \
    --run_name stage03 \
    --results_csv sweep.csv
```

`sweep.csv` gains one row per stage
(`run_name, ckpt_path, n_probe_classes, probe_val_acc, probe_val_loss`) — plot
`probe_val_acc` against the number of pretraining datasets to read off the curve.

## Data QC — `mass_dist.py`

Compare the **MS1 peak m/z distribution** of each new PRIDE repository against the abele
data, to check for a train/eval distribution shift. The model only predicts m/z bins over
`[bin_mz_min, bin_mz_max)` = `[300, 1500)` (peaks below → ignored `-1`; at/above → top bin),
so a repo whose mass distribution is shifted — or that carries a lot of mass outside that
window — is worth spotting before pretraining on it.

Each PXD accession is one series (its own table row, curve, and JSD/TV/KS divergence vs the
reference); abele is the reference. Uses the same preprocessing as `train.py`; no model/GPU.
Reading is parse-dominated, so `--limit-files` (default 5, strided per series) is the main
speed knob — an m/z distribution stabilises on very few files.

```bash
python scripts/mass_dist.py \
    --pride-root /mnt/data/shared/lc_ms_foundation/pride_data \
    --dataset abele /mnt/data/shared/lc_ms_foundation/abele_data/mzml \
    --reference abele -o mass_dist.png --dump-csv mass_dist.csv
```

Prints a per-series stats table (mean/median/quantiles, `%<300`, `%≥1500`, `%in-window`,
`JSDvsref`), writes a two-panel overlay PNG (full range + model-window zoom, abele bold), and
optionally dumps per-bin counts to CSV for custom replots.
