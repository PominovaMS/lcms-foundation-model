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
| `--force` | off | Rebuild unconditionally. See the reuse rule below. |

**Reuse — re-running a stage is cheap.** The build is skipped unless something that
would change the data changed: the mzML file list in either split dir, the `--seed`,
or the config's `max_num_peaks` (or a `.lance` dir has gone missing). A rebuild always
says why:

```
Reusing .../lance — inputs unchanged (train=1,874,904  val=168,085 spectra, seed 42, ...)
Rebuilding .../lance: train inputs changed (+12 / -0 files)
Rebuilding .../lance: seed changed (42 -> 99)
```

So a hyperparameter sweep over one stage pays the ingest cost once, while changing
`--accessions` or `--limit` rebuilds on its own. `run_sweep.slurm` regenerating the
symlink farm does *not* trigger a rebuild — the names are stable.

Comparison is by file **name**. Replacing a symlink target with different content
under the same name goes unnoticed; use `--force` for that.

> If a re-run is re-parsing mzML when you expected reuse, check that the output dir
> isn't inside a directory something else deletes first — `run_sweep.slurm` keeps it
> at `$SWEEP_ROOT/lance/$STAGE`, deliberately outside the `rm -rf`'d stage dir.

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
`train_checkpoints/foundation_model/lightning_logs/<run_name>/checkpoints/`. Only
best-`val_loss` and `last.ckpt` survive by default; add `--save_every_n_epochs N` to
also keep a weights-only checkpoint per N epochs under `checkpoints/epochs/`, which is
what step 4 needs to plot probe accuracy against pretraining epoch.

### 4. Evaluate each checkpoint downstream — `eval/probe_checkpoint.py`

Loads a pretrained checkpoint, freezes the encoder, trains a fresh linear probe on
the abele genus task, and reports validation accuracy. Point `--results_csv` at the
**same file** for every stage to accumulate the scaling curve. Keep
`--probe_label`, `--n_classes`, `--val_frac` and `--label_offset` identical across
stages so the probe split is fixed.

```bash
python eval/probe_checkpoint.py \
    --ckpt_path train_checkpoints/foundation_model/lightning_logs/stage03/checkpoints/last.ckpt \
    --data_dir /mnt/data/shared/lc_ms_foundation/abele_data/mzml \
    --meta_path /mnt/data/shared/lc_ms_foundation/abele_data/all_abele_metadata.csv \
    --config config.yaml \
    --run_name stage03 \
    --probe_label genus --n_classes 2 --val_frac 0.3 --label_offset 0 \
    --mzml_cache_dir /mnt/data/cadams/cache/abele_mzml \
    --results_csv sweep.csv
```

`--encode_chunk_size` (512) is how many spectra go through the encoder in one
forward pass. It is not an eval setting — chunking is exact, so it changes only
whether the numbers compute, never what they are — but a whole LC-MS run in a
single forward is tens of GB of activations and OOMs the GPU. Lower it if the
encoder still runs out of memory; the log prints the per-run spectrum counts.

`--mzml_cache_dir` writes one parquet per parsed mzML and reloads them on later
runs. The ~450-file parse dominates the runtime of a probe, so only the first
invocation pays it. A cache dir is valid for exactly one `(--data_dir,
max_num_peaks)` pair and the run aborts if either disagrees.

`sweep.csv` gains one row per stage — plot `probe_val_acc` against the number of
pretraining datasets to read off the curve. Every split and probe setting is
recorded alongside the metrics, so a row is self-describing.

#### Probe accuracy vs. pretraining epoch

`--ckpt_path` takes any checkpoint, so the series from `train.py --save_every_n_epochs`
answers "how much pretraining does the downstream task actually need?" on a single run.
Each row records `ssl_epoch` / `ssl_step`, read out of the checkpoint, so one `run_name`
and one CSV are enough:

```bash
# one array task per saved epoch; all share MZML_CACHE, so only the first parses mzML
sbatch --array=0-9 --export=ALL,RUN=<run_name>,CKPT_GLOB=1 run_probe.slurm
python scripts/plot_sweep.py --results_csv results/probe/<run_name>.csv \
    --x ssl_epoch -o epochs.png
```

Use a **fresh** CSV under `results/probe/`, not the sweep's `results/<run>.csv`: these
rows vary pretraining time rather than data, and the default `--x stage` axis would
read them as extra diversity stages.

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

**The bar is `majority_acc` (~0.70), not `random_acc` (0.50)** — the default
2-genus split is deliberately unbalanced (see below), so a probe scoring 0.65 is
worse than predicting "Pseudomonas" every time. Read `probe_val_acc_macro` and
`n_pred_classes` alongside `probe_val_acc`.

`probe_epochs` distinguishes a third case: if it pins at `--probe_n_epochs`, the
probe never reached `--probe_min_train_loss` and is *underfit*, not collapsed.

#### The probe split

`assign_splits` takes the `--n_classes` (2) most abundant `--probe_label` values
(`genus` by default; `species` reads the `organism` column), dropping
`genus == "food"` — a sample type, not an organism. On abele that is Pseudomonas
(312 files) and Staphylococcus (136). Each class's **own** files are then split
`--val_frac` (0.3) into train/val, with val indices chosen by `stride` — evenly
spaced across the sorted file list rather than a contiguous prefix, so val samples
across acquisition order. Every other file is `split="unused"` and read by nothing.

| | files | train | val |
| --- | --- | --- | --- |
| Pseudomonas | 312 | 218 | 94 |
| Staphylococcus | 136 | 95 | 41 |
| **total** | **448** | **313** | **135** |

Nothing is reserved for SSL — pretraining runs on PRIDE, so abele is purely an
evaluation set and there is no `"train"` split. Classes are not balanced or capped
either: they keep their natural 2.3:1 ratio, hence `majority_acc ≈ 0.70`. Because
the split is proportional, train and val carry the *same* class distribution, which
is what makes `majority_acc` a well-behaved collapse threshold.

`--label_offset` shifts that selection down the ranking: `--label_offset 1` skips
the largest label and probes #2 + #3 instead — Staphylococcus (136) + Bacillus
(109), 245 files at 1.25:1. That is the lever for the imbalance, and it lowers
`majority_acc` to ~0.55, so **compare `majority_acc` across rows, not just
`probe_val_acc`**: an offset row can score lower and still be the better result.
The cost is 245 files instead of 448, so `probe_val_acc` is coarser. At
`--probe_label species` the second- and third-largest species may sit in the same
genus, which makes it a within-genus task — a different question, not comparable
with the genus rows. `class_names` records exactly what was selected, so the offset
gets no CSV column of its own. The split log prints the label ranking with the
selected rows marked; read it before picking an offset.

**What this measures.** Train and val share species by construction, so this is
"same species, unseen run", not cross-species generalisation — deliberately the
lower-variance sanity check on whether the encoder carries organism information at
all. Pseudomonas is also near-monospecific, so in practice it is *P. aeruginosa* vs
assorted Staphylococcus. `assign_splits` logs each class's per-species train/val
counts; that log is the only record of the composition, so read it before drawing
conclusions. Numbers from the older species-alternating split are not comparable
with these, and the results CSV header changed with it — write to a fresh file.

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
