#!/bin/bash
#SBATCH --partition=one_day
#SBATCH --job-name=ingest_pxd
#SBATCH --cpus-per-task=16
#SBATCH --output=/mnt/data/cadams/slurm_output/stdout.%j
#SBATCH --error=/mnt/data/cadams/slurm_output/stderr.%j
#SBATCH --nodelist=asimov

set -uo pipefail

ENV=/home/cadams/miniconda3/envs/lcms-foundation
export THERMORAWFILEPARSER=$ENV/bin/ThermoRawFileParser   # ingest_pxd.py reads this

cd /mnt/data/cadams/lcms-foundation-model

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Where all datasets are symlinked into train_mzml/ + val_mzml/ for training.
LINK_INTO="/mnt/data/shared/lc_ms_foundation/training_sets/current"
VAL_FRAC=0.1          # deterministic, stable per-file split
SEED=0
JOBS=8                # parallel conversions (keep == --cpus-per-task above)
PRUNE_RAW=1           # 1 = delete each .raw once its mzML is verified on disk; 0 = keep raws

# Direction through the sorted file list. Default forward; override at submit time
# with `sbatch --export=ALL,REVERSE=1 ingest_data.slurm` to walk from the end, so a
# second job can chew the same dataset from the opposite end concurrently.
REVERSE=${REVERSE:-0} # 0 = first LIMIT files, 1 = last LIMIT files

# FTP robustness for large files on slow PRIDE mirrors (ppx defaults: 10s / 10 reconnects).
# ppx resumes partial downloads (append + REST), so retries pick up where they left off.
TIMEOUT=60            # per-operation FTP socket timeout (seconds)
MAX_RECONNECTS=50     # reconnects ppx attempts mid-file before raising
DOWNLOAD_ATTEMPTS=20  # outer resume attempts if a download still fails after all reconnects

# Datasets to ingest, in order.
ACCESSIONS=(
    PXD014877
    # PXD012345
    # PXD067890
)

# Applied to every dataset: filename filter (fnmatch) and per-dataset file cap.
GLOB="*.raw"
LIMIT=500

# --------------------------------------------------------------------------- #
# Ingest each dataset (continue on failure, report a summary at the end)
# --------------------------------------------------------------------------- #
failed=()
for acc in "${ACCESSIONS[@]}"; do
    args=("$acc"
          --link-into "$LINK_INTO"
          --val-frac "$VAL_FRAC"
          --seed "$SEED"
          --jobs "$JOBS"
          --glob "$GLOB"
          --limit "$LIMIT"
          --timeout "$TIMEOUT"
          --max-reconnects "$MAX_RECONNECTS"
          --download-attempts "$DOWNLOAD_ATTEMPTS")
    [[ "$PRUNE_RAW" -eq 1 ]] && args+=(--prune-raw)
    [[ "$REVERSE" -eq 1 ]] && args+=(--reverse)

    echo "=========================================================="
    echo "Ingesting ${acc} ($([[ "$REVERSE" -eq 1 ]] && echo reverse || echo forward))"
    echo "=========================================================="
    if ! python scripts/ingest_pxd.py "${args[@]}"; then
        echo "ERROR: ingest failed for ${acc}" >&2
        failed+=("$acc")
    fi
done

echo
echo "=========================================================="
if [[ ${#failed[@]} -eq 0 ]]; then
    echo "All ${#ACCESSIONS[@]} dataset(s) ingested into ${LINK_INTO}"
else
    echo "Ingested with ${#failed[@]} failure(s): ${failed[*]}"
    exit 1
fi

