#!/bin/bash
# Per stage: build the training set, pretrain, then probe on abele.
#   Smoke test (stage 1):  sbatch run_sweep.slurm
#   Full sweep:            sbatch --array=1-3 run_sweep.slurm

#SBATCH --partition=one_day
#SBATCH --job-name=pride_sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=asimov
#SBATCH --output=/mnt/data/cadams/slurm_output/stdout.%A_%a
#SBATCH --error=/mnt/data/cadams/slurm_output/stderr.%A_%a

set -euo pipefail

ENV=/home/cadams/miniconda3/envs/lcms-foundation
REPO=/mnt/data/cadams/lcms-foundation-model
cd "$REPO"

PRIDE_ROOT=/mnt/data/shared/lc_ms_foundation/pride_data
SWEEP_ROOT=/mnt/data/shared/lc_ms_foundation/training_sets/sweep
ABELE_MZML=/mnt/data/shared/lc_ms_foundation/abele_data/mzml
ABELE_META=/mnt/data/shared/lc_ms_foundation/abele_data/all_abele_metadata.csv

# Accessions in diversity order; stage N = the first N. Array size must match.
ACCESSIONS=(PXD014877)
MAX_EPOCHS=2    # smoke test; raise for the real run
LIMIT=4         # smoke test: files per accession. Set empty ("") to use all.
PROBE_ALL=      # empty = probe 15 mid-sized genera. Set to 1 to probe ALL genera (slower).

# Stage = array index, or 1 for a bare submit.
STAGE_NUM="${SLURM_ARRAY_TASK_ID:-1}"
STAGE=$(printf "stage%02d" "$STAGE_NUM")
STAGE_DIR="$SWEEP_ROOT/$STAGE"
CKPT="$REPO/source/train_checkpoints/foundation_model/lightning_logs/$STAGE/checkpoints/last.ckpt"

# 1) Build this stage's training set from accessions 1..STAGE_NUM.
#    Clear first so the dir (a regenerable symlink farm) reflects the current LIMIT.
rm -rf "$STAGE_DIR"
"$ENV/bin/python" scripts/build_stage.py \
    --root "$PRIDE_ROOT" --out-root "$STAGE_DIR" \
    --accessions "${ACCESSIONS[@]:0:$STAGE_NUM}" \
    ${LIMIT:+--limit "$LIMIT"}

# 2) Pretrain (train.py runs from source/, so checkpoints land under source/).
( cd source && "$ENV/bin/python" train.py \
    --data_dir "$STAGE_DIR" --config ../config.yaml \
    --run_name "$STAGE" --max_epochs "$MAX_EPOCHS" )

# 3) Probe the checkpoint on abele -> one row in results/<stage>.csv.
mkdir -p "$REPO/results"
"$ENV/bin/python" eval/probe_checkpoint.py \
    --ckpt_path "$CKPT" --data_dir "$ABELE_MZML" --meta_path "$ABELE_META" \
    --config config.yaml --run_name "$STAGE" \
    --results_csv "$REPO/results/$STAGE.csv" \
    ${PROBE_ALL:+--probe_all}

cat "$REPO/results/$STAGE.csv"
