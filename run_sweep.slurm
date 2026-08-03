#!/bin/bash
# Per stage: build the training set, pretrain, then probe on abele.
# Build one mixed training set from ALL accessions, pretrain, then probe on abele.
#   Single mixed run:  sbatch run_sweep.slurm

#SBATCH --partition=seven_days
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

# Datasets to mix into a single training set
ACCESSIONS=(PXD004452 PXD012636 PXD009737 PXD030983 PXD013868)
MAX_EPOCHS=50     # smoke test; raise for the real run
LIMIT=100         # smoke test: files per accession. Set empty ("") to use all.
PROBE_ALL=      # empty = probe 15 mid-sized genera. Set to 1 to probe ALL genera (slower).

# Optimizer overrides (empty = use config.yaml: lr 5e-4, warmup 2000). Set these to
# sweep LR without touching the shared config, which the abele runs also read.
# Bigger + more heterogeneous training sets tolerate (and usually want) a higher
# peak LR, but need a longer warmup so the early, high-variance batches from five
# different labs don't spike the loss before the schedule has ramped.
LR=1e-3
WARMUP_ITERS=6000

RUN_TAG=lim5x100_ep50_lr1e-3        # optional label to separate runs of the SAME stage (e.g. "lim25_ep50").
                # Keeps checkpoints, logs, and results from colliding/overwriting.
                # Include the LR here whenever you change it, or runs overwrite each other.

# Stage = array index, or 1 for a bare submit. RUN adds the optional tag so distinct
# configs (e.g. a data-volume ramp on one accession) don't overwrite each other.
STAGE_NUM="${SLURM_ARRAY_TASK_ID:-1}"
STAGE=$(printf "stage%02d" "$STAGE_NUM")
STAGE_DIR="$SWEEP_ROOT/$STAGE"
RUN="$STAGE${RUN_TAG:+_$RUN_TAG}"
CKPT="$REPO/source/train_checkpoints/foundation_model/lightning_logs/$RUN/checkpoints/last.ckpt"

# 1) Build the mixed training set from ALL accessions (one combined train/val set).
#    Clear first so the dir (a regenerable symlink farm) reflects the current LIMIT.
rm -rf "$STAGE_DIR"
"$ENV/bin/python" scripts/build_stage.py \
    --root "$PRIDE_ROOT" --out-root "$STAGE_DIR" \
    --accessions "${ACCESSIONS[@]}" \
    ${LIMIT:+--limit "$LIMIT"}

# 2) Pretrain (train.py runs from source/, so checkpoints land under source/).
( cd source && "$ENV/bin/python" train.py \
    --data_dir "$STAGE_DIR" --config ../config.yaml \
    --run_name "$RUN" --max_epochs "$MAX_EPOCHS" \
    ${LR:+--lr "$LR"} ${WARMUP_ITERS:+--warmup_iters "$WARMUP_ITERS"} )

# 3) Probe the checkpoint on abele -> one row in results/<run>.csv.
mkdir -p "$REPO/results"
"$ENV/bin/python" eval/probe_checkpoint.py \
    --ckpt_path "$CKPT" --data_dir "$ABELE_MZML" --meta_path "$ABELE_META" \
    --config config.yaml --run_name "$RUN" \
    --results_csv "$REPO/results/$RUN.csv" \
    ${PROBE_ALL:+--probe_all}

cat "$REPO/results/$RUN.csv"
