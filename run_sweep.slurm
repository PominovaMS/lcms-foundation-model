#!/bin/bash
# Per stage: build the training set, pretrain, then probe on abele.
# Build one mixed training set from ALL accessions, pretrain, then probe on abele.
#   Single mixed run:  sbatch run_sweep.slurm
#
# Re-running the SAME data with different hyperparameters (e.g. another LR): change
# LR and RUN_TAG below and resubmit. Step 2 reuses the existing Lance dataset, so the
# runs train on byte-identical data.
#   BUT WAIT for any in-flight job on the same DATA_TAG to finish step 2 (it prints
#   "Wrote .../lance/stageNN_<DATA_TAG>") before submitting. Earlier than that, this
#   job's `rm -rf "$STAGE_DIR"` deletes the dir that job is still ingesting from, and
#   both would write the same Lance dataset at once. Editing this file mid-run is safe —
#   Slurm spools the script at submit time.
#
# Training on a DIFFERENT input set (e.g. 5xnolim instead of lim5x50): change
# ACCESSIONS/LIMIT, then change DATA_TAG to match. That routes the stage dir and the
# Lance dataset to their own paths, so the new set is built and saved alongside the old
# one rather than over it — and neither job has to wait on the other.

#SBATCH --partition=one_day
#SBATCH --job-name=pride_sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --nodelist=asimov
#SBATCH --output=/mnt/data/cadams/slurm_output/stdout.%A_%a
#SBATCH --error=/mnt/data/cadams/slurm_output/stderr.%A_%a

set -euo pipefail

ENV=/home/cadams/miniconda3/envs/lcms-foundation
REPO=/mnt/data/cadams/lcms-foundation-model
cd "$REPO"

# Reduce CUDA memory fragmentation (the OOM error recommends this; reclaims the
# large "reserved but unallocated" pool).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PRIDE_ROOT=/mnt/data/shared/lc_ms_foundation/pride_data
SWEEP_ROOT=/mnt/data/shared/lc_ms_foundation/training_sets/sweep
ABELE_MZML=/mnt/data/shared/lc_ms_foundation/abele_data/mzml
ABELE_META=/mnt/data/shared/lc_ms_foundation/abele_data/all_abele_metadata.csv

# Cache of the parsed abele spectra for step 4 (one parquet per mzML), so each sweep run
# reloads them instead of re-parsing all ~260 files after pretraining. Shared with
# run_probe.slurm — same mzML, same max_num_peaks, so the two fill one cache for each
# other. `-` not `:-`: an explicitly empty export must mean "no cache". A cache dir is
# tied to one ABELE_MZML at one max_num_peaks and the run aborts if either disagrees, so
# change the two together.
MZML_CACHE="${MZML_CACHE-/mnt/data/cadams/cache/abele_mzml}"

# Datasets to mix into a single training set
ACCESSIONS=(PXD004452 PXD012636 PXD009737 PXD030983 PXD013868)
MAX_EPOCHS=25     # smoke test; raise for the real run
LIMIT=50        # files per accession. Empty ("") = use all.
# Keep a weights-only checkpoint every N epochs under checkpoints/epochs/, so probe
# accuracy can be plotted against pretraining epoch afterwards:
#     sbatch --array=0-9 --export=ALL,RUN=<run>,CKPT_GLOB=1 run_probe.slurm
#     python scripts/plot_sweep.py --results_csv results/probe/<run>.csv --x ssl_epoch
# Empty ("") = off, and the run keeps only best-val_loss + last.ckpt as before. Costs
# ~310 MB per file at d_model=1024/L9/ff2048, so pick N for MAX_EPOCHS/N files.
SAVE_EVERY_N_EPOCHS=1

# Names the DATA (accessions + LIMIT), and nothing else. It goes into the stage dir and
# the Lance dir, so each data variant gets its own paths and they coexist instead of
# overwriting each other:
#   5x50   -> 5 accessions, LIMIT=50
#   5xnolim   -> 5 accessions, no limit
# CHANGE THIS WHENEVER YOU CHANGE ACCESSIONS OR LIMIT. If you don't, the new data
# lands on the old dataset's path and step 2 rebuilds over it — the previous dataset
# is gone. Hyperparameters do NOT belong here; they go in RUN_TAG. Keep it echoed in
# RUN_TAG too, so the run name says which data the checkpoint was trained on.
DATA_TAG=5x50

# --- probe split + probe reproducibility -----------------------------------
# These define the EVAL, not the model. Every run you intend to compare must use
# identical values — they are passed explicitly rather than left to argparse
# defaults so a default change can never silently make new rows incomparable with
# old ones. All of them are recorded in the results CSV, so each row is
# self-describing.
PROBE_LABEL=genus         # genus | species
N_CLASSES=2               # the N most abundant labels by file count
VAL_FRAC=0.3              # held out per class; train/val share species
PROBE_SEED=0
PROBE_REPEATS=3           # accuracy is the mean over this many seeded probe fits;
                          # the reported std is the noise floor for comparing runs

# Optimizer overrides (empty = use config.yaml: lr 5e-4, warmup 2000). Set these to
# sweep LR without touching the shared config, which the abele runs also read.
# Bigger + more heterogeneous training sets tolerate (and usually want) a higher
# peak LR, but need a longer warmup so the early, high-variance batches from five
# different labs don't spike the loss before the schedule has ramped.
LR=5e-5
WARMUP_ITERS=6000

SHUFFLE_SEED=42   # seeds the on-disk shuffle of the training data (step 2).
                  # Fixed across a sweep so every stage differs only in its data.
TRAIN_SEED=42     # seeds training itself (step 3): the per-epoch batch shuffle, the
                  # peak masking, and model init. Also fixed across a sweep, for the
                  # same reason — vary it only to measure run-to-run variance.

RUN_TAG=l9d1024ff2048_bs64_lim5x100_ep50_lr5e-5
                # optional label to separate runs of the SAME stage (e.g. "lim25_ep50").
                # Keeps checkpoints, logs, and results from colliding/overwriting.
                # Include the LR here whenever you change it, or runs overwrite each other —
                # nothing else in the pipeline records it, so this string is the only
                # thing distinguishing two runs over the same data at different LRs.
                # Same goes for model geometry, batch/accumulation and precision: none of
                # them reach the run name or the results CSV on their own, so a d512/ff1024
                # fp32 run and a d512/ff2048 bf16 run are otherwise indistinguishable rows.

# Stage = array index, or 1 for a bare submit. DATA_ID names everything that depends on
# the DATA, so a new input set (5xnolim vs lim5x50) builds a SEPARATE stage dir and
# Lance dataset instead of overwriting the previous one — both stay on disk and are
# reusable. RUN adds RUN_TAG on top so distinct configs (e.g. another LR over that same
# data) don't overwrite each other's checkpoints, logs and results.
STAGE_NUM="${SLURM_ARRAY_TASK_ID:-1}"
STAGE=$(printf "stage%02d" "$STAGE_NUM")
DATA_ID="$STAGE${DATA_TAG:+_$DATA_TAG}"
STAGE_DIR="$SWEEP_ROOT/$DATA_ID"
# Deliberately OUTSIDE STAGE_DIR: step 1 does `rm -rf "$STAGE_DIR"`, which would
# otherwise delete the prebuilt dataset before it could ever be reused.
LANCE_DIR="$SWEEP_ROOT/lance/$DATA_ID"
RUN="$STAGE${RUN_TAG:+_$RUN_TAG}"
CKPT="$REPO/source/train_checkpoints/foundation_model/lightning_logs/$RUN/checkpoints/last.ckpt"

# Echo the resolved paths up front: step 2 is silent when it reuses an existing
# dataset, so this is what tells you which data this run trained on.
echo "run        : $RUN"
echo "stage dir  : $STAGE_DIR"
echo "lance dir  : $LANCE_DIR"
echo "mzml cache : ${MZML_CACHE:-(disabled — re-parsing abele mzML)}"

# 1) Build the mixed training set from ALL accessions (one combined train/val set).
#    Clear first so the dir (a regenerable symlink farm — cheap) reflects the current
#    LIMIT. Symlink names are the original mzML names and the train/val split is a
#    stable hash, so regenerating it is a no-op as far as step 2 is concerned.
rm -rf "$STAGE_DIR"
"$ENV/bin/python" scripts/build_stage.py \
    --root "$PRIDE_ROOT" --out-root "$STAGE_DIR" \
    --accessions "${ACCESSIONS[@]}" \
    ${LIMIT:+--limit "$LIMIT"}

# 2) Ingest into a persistent, PRE-SHUFFLED Lance dataset.
#    Shuffling on disk is what makes a batch a cross-section of files rather than a
#    slice of one mzML (unshuffled, the loss shows a sawtooth locked to position
#    within the epoch).
#    NO --force: the script compares the mzML file list, the seed and max_num_peaks
#    against what it recorded last time, so re-running this stage with different
#    hyperparameters skips ingestion entirely and trains on byte-identical data,
#    while a change to ACCESSIONS or LIMIT rebuilds automatically. Step 1 regenerating
#    the symlink farm does NOT force a rebuild — the file names are stable.
#    Add --force to rebuild regardless (e.g. an mzML was replaced in place).
#    SHUFFLE_SEED is part of the experiment: record it if you change it.
"$ENV/bin/python" scripts/build_lance.py \
    --data_dir "$STAGE_DIR" --out "$LANCE_DIR" --config config.yaml \
    --seed "$SHUFFLE_SEED"

# 3) Pretrain (train.py runs from source/, so checkpoints land under source/).
#    Batches are re-drawn every epoch (--seed makes that reproducible); pass
#    --no_shuffle to reproduce a pre-shuffle run.
( cd source && "$ENV/bin/python" train.py \
    --lance_dir "$LANCE_DIR" --config ../config.yaml \
    --run_name "$RUN" --max_epochs "$MAX_EPOCHS" --seed "$TRAIN_SEED" \
    ${LR:+--lr "$LR"} ${WARMUP_ITERS:+--warmup_iters "$WARMUP_ITERS"} \
    ${SAVE_EVERY_N_EPOCHS:+--save_every_n_epochs "$SAVE_EVERY_N_EPOCHS"} )

# 4) Probe the checkpoint on abele -> one row in results/<run>.csv.
#    Results stay in a PER-RUN file: this script reads SLURM_ARRAY_TASK_ID, so it
#    can run as an array, and concurrent appends to one shared CSV on networked
#    storage are not guaranteed atomic. Every row records its own split settings,
#    so comparing runs is just reading several files:
#        python scripts/plot_sweep.py --results_csv results/*.csv -o compare.png
mkdir -p "$REPO/results"
"$ENV/bin/python" eval/probe_checkpoint.py \
    --ckpt_path "$CKPT" --data_dir "$ABELE_MZML" --meta_path "$ABELE_META" \
    --config config.yaml --run_name "$RUN" \
    --probe_label "$PROBE_LABEL" --n_classes "$N_CLASSES" \
    --val_frac "$VAL_FRAC" \
    --probe_seed "$PROBE_SEED" --probe_repeats "$PROBE_REPEATS" \
    --results_csv "$REPO/results/$RUN.csv" \
    ${MZML_CACHE:+--mzml_cache_dir "$MZML_CACHE"}

# 22 columns on one line is unreadable in a slurm log; tabulate if `column` exists.
column -s, -t < "$REPO/results/$RUN.csv" || cat "$REPO/results/$RUN.csv"
