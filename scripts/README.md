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

### 2. Build a pre-shuffled dataset — `build_lance.py`

Parses the stage's mzML into a Lance database **once**, in random order, and keeps
it on disk.

```bash
python scripts/build_lance.py \
    --data_dir /mnt/data/shared/lc_ms_foundation/training_sets/sweep/stage03 \
    --out      /mnt/data/shared/lc_ms_foundation/training_sets/sweep/stage03/lance \
    --seed 42
# -> <out>/{train.lance, val.lance, dataset_info.json}
```

**Why the shuffle is on disk.** Lance scans sequentially and ingest walks files in
sorted-name order, so unshuffled training replays an identical sequence every epoch
and every batch of 1024 comes from a single ~195k-spectrum file. That shows up as a
loss sawtooth locked to position within the epoch. Shuffling the rows physically
makes each batch a cross-section of files and datasets, costs nothing at training
time (still a sequential scan — no random reads), and leaves the batch format alone.

**Why it is persistent.** Without this, every run re-parses the whole corpus into a
temporary Lance DB and discards it. Build once, then sweep hyperparameters against
byte-identical data. The same `--seed` reproduces the same on-disk order.

The order is fixed on disk, so each epoch sees the same batch composition — the
shuffle is global but not re-drawn per epoch. That is the tradeoff that makes the
data reusable.

| flag | default | notes |
| --- | --- | --- |
| `--seed` | `42` | Shuffle seed; part of the experiment, so record it if you change it. |
| `--chunk-size` | `50000` | Spectra per write chunk. Bounds peak RAM during the shuffle; lower it if the build OOMs. |
| `--force` | off | Rebuild an existing output. Without it, a complete build is a no-op. |

**Disk:** staging and shuffled output coexist during the build, so it transiently
needs ~2x the final size (~11 GB for 3.5M spectra, so budget ~22 GB). Staging is
removed on success.

**Preprocessing is baked in.** `filter_intensity(max_num_peaks=...)` and the sqrt
scaling are applied at ingest, so a dataset built under one peak cap cannot be
reused under another. `dataset_info.json` records the value and `train.py` refuses
to start on a mismatch — rebuild instead. The sidecar also records the seed, the
spectra counts, and the exact file list per split, so a run can always be traced
back to its data.

### 3. Pretrain the foundation model per stage

Point `train.py` at the prebuilt dataset. Give every stage a distinct `--run_name`
so its logs + checkpoints land in their own directory (and stages don't overwrite
each other):

```bash
cd source && python train.py \
    --lance_dir /mnt/data/shared/lc_ms_foundation/training_sets/sweep/stage03/lance \
    --config ../config.yaml \
    --run_name stage03
```

`--data_dir <stage dir>` still works and ingests mzML directly, but it is
**unshuffled** — kept only to reproduce runs made before the shuffle existed.

**Two shuffles, two jobs — both are on by default.**

| | what it does | when |
| --- | --- | --- |
| `build_lance.py --seed` | randomises the **stored** order, so batches are a cross-section of files and datasets | once, at build |
| `train.py` batch shuffling | **re-draws** batches, so each epoch sees a different partition | every epoch |

The first makes the dataset reusable and fixes batches being one-file-at-a-time; the
second stops all 50 epochs training on the identical 2097 groupings. `--no_shuffle`
turns the second off (batches stay cross-file, because the stored order is already
random) and `--seed` makes it reproducible.

Per-epoch shuffling reads each batch with one random `take` instead of a sequential
scan. Measured cost is ~40 ms/batch on top of a ~675 ms/step baseline (~6%); it is
one `take` per *step*, not per row, which is what keeps it cheap.

**Validation is never reshuffled.** `validation_step` seeds its masking with
`42 + batch_idx` so masks are identical across epochs — which only holds if batch *k*
is the same spectra every time. Shuffling val would add noise to val loss and make
epoch-to-epoch comparisons meaningless.

Checkpoints (including a stable `last.ckpt`) are written under
`train_checkpoints/foundation_model/lightning_logs/<run_name>/checkpoints/`.

### 4. Evaluate each checkpoint downstream — `eval/probe_checkpoint.py`

Loads a pretrained checkpoint, freezes the encoder, trains a fresh linear probe on
the abele genus task, and reports validation accuracy. Point `--results_csv` at the
**same file** for every stage to accumulate the scaling curve. Keep
`--n_probe_genera`, `--n_ssl_top`, `--max_files_per_species` and
`--max_files_per_genus` identical across stages so the probe split is fixed.

```bash
python eval/probe_checkpoint.py \
    --ckpt_path train_checkpoints/foundation_model/lightning_logs/stage03/checkpoints/last.ckpt \
    --data_dir /mnt/data/shared/lc_ms_foundation/abele_data/mzml \
    --meta_path /mnt/data/shared/lc_ms_foundation/abele_data/all_abele_metadata.csv \
    --config config.yaml \
    --run_name stage03 \
    --results_csv sweep.csv
```

`sweep.csv` gains one row per stage — plot `probe_val_acc` against the number of
pretraining datasets to read off the curve. Every split and probe setting is
recorded alongside the metrics, so a row is self-describing.

#### Reading the numbers

Accuracy is a **mean over `--probe_repeats` (default 3) seeded probe fits**; the
probe initialisation is the only stochastic element, so `probe_val_acc_std` is the
noise floor. **Two runs are only distinguishable if their gap exceeds it.** Use
`--probe_seed` to reproduce a row exactly.

Four columns exist to tell a *collapsed* probe from a merely weak one — a probe
that predicts one class for every run can still post a respectable micro accuracy:

| column | meaning |
| --- | --- |
| `probe_val_acc_macro` | per-class mean accuracy; unaffected by class imbalance |
| `majority_acc` | score of always predicting the train-modal class — what a collapsed probe gets |
| `random_acc` | `1 / n_probe_classes` |
| `n_pred_classes` | distinct classes predicted on val; **1 means collapsed** |

`probe_epochs` distinguishes a third case: if it pins at `--probe_n_epochs`, the
probe never reached `--probe_min_train_loss` and is *underfit*, not collapsed.

#### Probe class balance

`assign_splits` alternates *species* between probe train and val, which assumes
species carry comparable file counts. In abele they do not — 80 of the 87 probe
species have 3 files, but *Escherichia coli* has 48, and being alphabetically
first it always lands in probe_train. Left uncapped it is 25% of probe_train and
2.3% of probe_val, so a probe that collapses onto it scores *below* random.

`--max_files_per_species` (default 3, the modal count) caps each species within
each probe split, keeping evenly-strided files rather than a prefix. Capped-out
files become `split="unused"` and are read by nothing — in particular they are
**not** donated to the SSL split, which would leak probe genera into pretraining.

| setting | train | val | train-vs-val TV | `majority_acc` |
| --- | --- | --- | --- | --- |
| `--max_files_per_species 0` (no cap) | 192 | 132 | 0.232 | 0.023 |
| `--max_files_per_species 3` (default) | 141 | 120 | 0.082 | 0.175 |
| `... 3 --max_files_per_genus 6` | 87 | 87 | 0.000 | 0.069 |

The per-genus cap balances the classes exactly, at roughly half the files. Numbers
produced under different caps are not comparable.

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

## Training curves — `plot_curves.py`

Overlay loss / accuracy / lr curves from one or more runs' TensorBoard event files into a
single PNG. Pass several run dirs to compare diversity stages on the same axes; on the
loss and accuracy panels color encodes train vs val and linestyle encodes the run.

```bash
python scripts/plot_curves.py ./tb_logs/stage01 ./tb_logs/stage04 -o compare.png
```

`train_loss` and `train_acc_mz_bin` are logged per optimizer step, so the raw curves are
mostly batch-to-batch noise. By default the raw values are drawn as a faint trace with a
TensorBoard-style EMA (bias-corrected) over the top carrying the trend:

| flag | default | effect |
| --- | --- | --- |
| `--smooth` | `0.9` | EMA weight. `0` disables it and plots the raw curves alone; `0.98` for a long, noisy run. |
| `--smooth-min-points` | `50` | Only smooth series with at least this many points, which leaves the short per-epoch `retrain_*` / `online_*` probe metrics raw. |

**Val curves are never smoothed**, whatever those flags say. They are logged once per epoch
and are already a mean over the whole validation set, so there is no batch-to-batch noise
to remove and an EMA would only add lag — on a 50-epoch run that lag was distorting
`val_loss` by ~30% of its dynamic range. The `lr` panel is never smoothed either: it is
deterministic, and checking the schedule shape is what that panel is for.

One thing to keep in mind when reading a train-vs-val gap: an EMA is causal, so on a
falling curve the smoothed train line sits slightly *above* the raw values
(~`1/(1-weight)` points of lag — about 10 logged points at 0.9). Part of any apparent
train-above-val gap early in a run is that lag, not the model. The faint raw trace
underneath is what to check it against.

If the train trend line still looks spiky at a *regular* period, that is not noise the EMA
failed to remove — it is epoch structure. `source/train.py` feeds spectra in fixed file
order with no shuffling, so every epoch replays the same sequence and per-file difficulty
shows up as a sawtooth. Turning `--smooth` up hides it rather than fixing it.
