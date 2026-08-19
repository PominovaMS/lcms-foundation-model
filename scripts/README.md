# scripts

## `ingest_pxd.py` — automated ProteomeXchange ingestion

Download a ProteomeXchange dataset by accession, convert Thermo `.raw` files to
mzML, and land the result in the shared server repository ready for training.

### Prerequisites

- Run on the **Linux host** where `/mnt/data/shared/lc_ms_foundation` is mounted
  (not on a Mac — the default paths won't exist there).
- `ppx` and `ThermoRawFileParser` must be available. Both are declared in
  `environment.yml`:

  ```bash
  conda env update -f environment.yml
  # sanity check:
  python -c "import ppx"
  ThermoRawFileParser --help
  ```

  If `ThermoRawFileParser` is installed under a different command name, point the
  script at it with the `THERMORAWFILEPARSER` env var.

### Usage

```bash
# Dry run: list the files that would be fetched, download nothing.
python scripts/ingest_pxd.py PXD014877 --glob "*iRT*.raw" --limit 2 --dry-run

# Real ingest: download a subset, convert to mzML, and symlink into a training dir.
python scripts/ingest_pxd.py PXD014877 \
    --glob "*iRT*.raw" --limit 20 \
    --link-into /mnt/data/shared/lc_ms_foundation/training_sets/current \
    --val-frac 0.1 --jobs 4
```

Then train against the linked directory (no code change needed):

```bash
cd source && python train.py \
    --data_dir /mnt/data/shared/lc_ms_foundation/training_sets/current \
    --config ../config.yaml
```

### Key options

| Flag | Purpose |
|---|---|
| `--root` | Collection root (default `/mnt/data/shared/lc_ms_foundation/pride_data`) |
| `--glob` | fnmatch filter on remote file names, e.g. `"*iRT*.raw"` |
| `--limit` | Cap the number of files |
| `--all` | Opt in to downloading the **entire** dataset |
| `--jobs` | Parallel RAW→mzML conversions |
| `--link-into` | Training dir to symlink converted mzML into |
| `--val-frac` / `--seed` | Deterministic, file-level train/val split |
| `--dry-run` | List + plan only |
| `--force` | Ignore the manifest and redo download/convert/link |

**Guardrail:** a bare `ingest_pxd.py PXD014877` with none of `--glob`, `--limit`,
or `--all` refuses to run — ProteomeXchange datasets can be hundreds of GB.

### What it produces

```
/mnt/data/shared/lc_ms_foundation/pride_data/
  PXD014877/
    raw/            # downloaded .raw
    mzml/           # converted .mzML  (per-dataset source of truth)
    manifest.json   # download + conversion state (idempotent re-runs)

/mnt/data/shared/lc_ms_foundation/training_sets/current/
    train_mzml/     # symlinks into pride_data/*/mzml  (train split)
    val_mzml/       # symlinks into pride_data/*/mzml  (val split)
```

Re-running is safe: the manifest skips already-downloaded and already-converted
files, and the split is stable per file name (adding datasets never reshuffles
existing assignments).

### Notes

- Only MS1 scans are used for training; conversion keeps **all** MS levels and
  `train.py` filters to `ms_level=1` downstream. Do not filter MS level at
  conversion time.
- After editing `environment.yml`, regenerate `conda-lock.yml` with `conda-lock`.
