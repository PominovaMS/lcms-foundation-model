import os

import numpy as np
import lance
import torch
import torch.nn.functional as F
from depthcharge.data import SpectrumDataset, spectra_to_df
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def build_dataset(data_dir, preprocessing_fn, batch_size, path=None):
    """Build a SpectrumDataset by streaming one mzML into Lance at a time.

    Appending per file (depthcharge's `add_spectra`, mode="append") keeps only a
    single file's DataFrame in RAM, so peak memory scales with one file instead of
    the whole corpus — the previous `pl.concat` of every file did not scale to a
    full dataset (hundreds of ~195k-spectra files).

    `path` is where the Lance database lands. `None` puts it in a TemporaryDirectory
    owned by the returned SpectrumDataset (so it dies with the process); pass a path
    to persist it, which is what `scripts/build_lance.py` does.

    Returns
    -------
    (SpectrumDataset, list of str, list of str)
        The dataset, the mzML files ingested, and the files skipped as unreadable.
    """
    # Only feed mzML to the parser. depthcharge dispatches a parser by extension, and
    # its MGF parser is MS2-only ("ms_level 1 is currently not supported") — so a stray
    # .mgf (or any non-mzML) symlinked into the dir would crash MS1 ingestion. Match the
    # `.mzml` convention used by the stage builders and peak_stats.py.
    all_entries = sorted(os.listdir(data_dir))
    mzml_files = [f for f in all_entries if f.lower().endswith((".mzml", ".mzml.gz"))]
    skipped = [f for f in all_entries if f not in mzml_files]
    if skipped:
        print(f"Skipping {len(skipped)} non-mzML file(s) in {data_dir}: {skipped}")

    ds = None
    ingested = []
    corrupt = []
    for mzml_file in mzml_files:
        try:
            df = spectra_to_df(
                os.path.join(data_dir, mzml_file),
                metadata_df=None,
                ms_level=1,
                preprocessing_fn=preprocessing_fn,
                valid_charge=None,
                custom_fields=None,
                progress=True,
            )
        except Exception as e:  # corrupt/truncated mzML — skip so one bad file
            corrupt.append(mzml_file)  # doesn't kill a long run mid-build
            print(f"WARNING: skipping unreadable mzML {mzml_file} ({type(e).__name__}: {e})")
            continue
        if ds is None:
            ds = SpectrumDataset(df, batch_size=batch_size, path=path)
        else:
            ds.add_spectra(df)
        ingested.append(mzml_file)
        del df  # free this file before loading the next
    if corrupt:
        # Conspicuous summary so a systemic conversion problem can't hide behind
        # per-file warnings scrolled off the log.
        print(f"WARNING: skipped {len(corrupt)}/{len(mzml_files)} unreadable mzML "
              f"file(s) in {data_dir}: {corrupt}")
    if ds is None:
        raise SystemExit(f"No readable mzML files in {data_dir}")
    return ds, ingested, corrupt


def iterable_steps_per_epoch(spectra, batch_size):
    """Exact number of batches a sequential scan of ``spectra`` yields per epoch.

    NOT ``ceil(n_spectra / batch_size)``. Two things break that formula:

    * Lance's scanner never emits an Arrow batch spanning two fragments, so each
      fragment ends in a short batch. There is one fragment per mzML file (or per
      ``build_lance.py`` write chunk).
    * ``lance.torch.data._buffer_arrow_batches`` then greedily packs whole Arrow
      batches into groups of at most ``batch_size`` — it concatenates across
      fragments but never *splits* a batch, so a group can come out well under
      ``batch_size``.

    Net effect: with many small files the real step count runs above the formula
    (measured: 100 files x 800 spectra at batch 1024 gives 100 batches, not 79).
    That undercount is what made ``OneCycleLR`` raise "Tried to step N times"
    partway through a run, and because it scales with file count it hit some sweeps
    and not others.

    This replays the same greedy packing, so it is exact rather than a bound.
    """
    steps = 0
    pending = 0
    for frag in spectra.dataset.get_fragments():
        full, remainder = divmod(frag.count_rows(), batch_size)
        arrow_batches = [batch_size] * full + ([remainder] if remainder else [])
        for size in arrow_batches:
            if pending > 0 and pending + size > batch_size:
                steps += 1
                pending = 0
            pending += size
    return steps + (1 if pending else 0)


class SpectrumIndexDataset(Dataset):
    """Map-style view over a depthcharge ``SpectrumDataset``, so it can be shuffled.

    ``SpectrumDataset`` is an ``IterableDataset`` (it inherits lance's
    ``LanceDataset``), and PyTorch refuses ``shuffle=True`` on those — which is the
    only reason a plain shuffled DataLoader doesn't work here. But depthcharge does
    support batched random access: ``spectra[[i, j, k]]`` runs ``listify`` over the
    index, does ONE ``lance.take``, and returns a fully collated, padded batch.

    So this dataset just hands out indices and :func:`batch_collate` redeems them,
    letting DataLoader's ordinary ``RandomSampler`` re-draw batches every epoch. The
    indirection is what keeps it to one ``take`` per *step* rather than one per row.

    All tensorization and padding stay inside depthcharge's ``_to_tensor``, so
    batches are identical to the sequential path — including per-batch padding
    width, which the ``intensity != 0`` real-peak mask relies on.
    """

    def __init__(self, spectra):
        self.spectra = spectra

    def __len__(self):
        return self.spectra.n_spectra

    def __getitem__(self, idx):
        return idx


def batch_collate(spectra):
    """Build a ``collate_fn`` that turns a list of indices into one depthcharge batch."""

    def collate(indices):
        return spectra[list(indices)]

    return collate


class LanceMapDataset(Dataset):
    """
    Map-style dataset wrapper
    (to support shuffle=True with Lance dataset).

    Note: num_workers is supposed to be 0. If you want num_workers>0,
    multi-processing workarounds are required.
    Note: it's implemented in official lance package as SafeLanceDataset,
    but we can't use newer lance versions not to break depthcharge.
    """

    def __init__(self, lance_path, seq_len=None):
        self.lance_path = str(lance_path)
        self.seq_len = seq_len

        self._ds = (
            self._get_ds()
        )  # must be moved from __init__ to __getitem__ if num_workers>0
        self._n = lance.dataset(self.lance_path).count_rows()

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        item_dict = self._ds.take([int(idx)]).to_pydict()
        item_dict = {k: v[0] for k, v in item_dict.items()}
        item_dict["mz_array"] = np.array(item_dict["mz_array"], dtype=np.float32)
        item_dict["intensity_array"] = np.array(
            item_dict["intensity_array"], dtype=np.float32
        )

        # pad peaks sequence to seq_len (FIXME: move to collate_fn?)
        if self.seq_len is not None and len(item_dict["mz_array"]) < self.seq_len:
            pad_right = self.seq_len - len(item_dict["mz_array"])
            item_dict["mz_array"] = np.pad(item_dict["mz_array"], (0, pad_right))
            item_dict["intensity_array"] = np.pad(
                item_dict["intensity_array"], (0, pad_right)
            )

        return item_dict

    def _get_ds(self):
        self._ds = lance.dataset(self.lance_path)
        return self._ds


class RunDataset(Dataset):
    """
    Dataset to return a full LCMS run as item.

    Each run is represented as a list of MS1 spectra
    (ordered by RT time, but RT values are currently not provided).
    Run length can vary - has to be handled properly in data loader.
    Run_labels can be returned as labels for each run item, if provided.
    """

    def __init__(self, run_dfs, run_labels=None, seq_len=None):
        self.seq_len = seq_len

        self.runs = []
        self.run_labels = [] if run_labels is not None else None
        for run_df in tqdm(run_dfs):
            run_mz_arrays = run_df["mz_array"].to_numpy()
            run_intensity_arrays = run_df["intensity_array"].to_numpy()

            if self.seq_len is not None:
                run_mz_arrays = [self._pad_sequence(seq) for seq in run_mz_arrays]
                run_mz_arrays = np.stack(run_mz_arrays, axis=0).astype(np.float32)
                run_intensity_arrays = [
                    self._pad_sequence(seq) for seq in run_intensity_arrays
                ]
                run_intensity_arrays = np.stack(run_intensity_arrays, axis=0).astype(
                    np.float32
                )

            run_data = {
                "mz_array": run_mz_arrays,
                "intensity_array": run_intensity_arrays,
            }
            self.runs.append(run_data)

            if run_labels is not None:
                run_file = run_df["peak_file"].first()
                self.run_labels.append(run_labels[run_file])

    def __len__(self):
        return len(self.runs)

    def __getitem__(self, idx):
        item = self.runs[idx]
        if self.run_labels is not None:
            item["label"] = self.run_labels[idx]
        return item

    def _pad_sequence(self, sequence):
        if len(sequence) < self.seq_len:
            pad_right = self.seq_len - len(sequence)
            return np.pad(sequence, (0, pad_right))
        return sequence
