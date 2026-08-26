import os
import numpy as np
import pyarrow as pa
import lance
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from depthcharge.data import SpectrumDataset, spectra_to_df, preprocessing, CustomField
from tqdm import tqdm


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

        self._ds = self._get_ds() # must be moved from __init__ to __getitem__ if num_workers>0
        self._n = self._ds.count_rows()

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return self.__getitems__([idx])[0]

    def __getitems__(self, indices: list[int]) -> list:
        items = self._ds.take([int(i) for i in indices]).to_pylist()
        return [self._prepare_item(item) for item in items]

    def _prepare_item(self, item):
        item["mz_array"] = np.asarray(item["mz_array"], dtype=np.float32)
        item["mz_array"] = self._pad_sequence(item["mz_array"])

        item["intensity_array"] = np.asarray(item["intensity_array"], dtype=np.float32)
        item["intensity_array"] = self._pad_sequence(item["intensity_array"])
        return item

    def _get_ds(self):
        return lance.dataset(self.lance_path)

    def _pad_sequence(self, sequence):
        if self.seq_len is not None and len(sequence) < self.seq_len:
            pad_right = self.seq_len - len(sequence)
            return np.pad(sequence, (0, pad_right))
        return sequence


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


# FIXME: check that this collate function corresponds to the one used in the notebooks (and thus is correct)
def run_collate_fn(rows: list[dict]) -> dict:
    """Collate function for RunDataset.

    Keeps mz_array and intensity_array as lists of per-run tensors
    (runs can have different lengths), and stacks scalar fields.
    """
    keys = rows[0].keys()
    batch = {}
    for key in keys:
        if key in ("mz_array", "intensity_array"):
            batch[key] = [torch.tensor(r[key]) for r in rows]
        else:
            batch[key] = torch.tensor([r[key] for r in rows])
    return batch


def build_dataset(
    mzml_files, 
    data_config, 
    lance_path, 
    force_rebuild: bool = False,
) -> LanceMapDataset:
    """Build a Lance-backed dataset from a collection of mzML files.

    MS1 spectra are read from each mzML file, preprocessed, and added to a
    Lance-backed ``SpectrumDataset``. The resulting dataset is wrapped in
    ``LanceMapDataset`` to provide map-style access and support PyTorch
    DataLoader shuffling.

    Parameters
    ----------
    mzml_files : list[str]
        Paths to the mzML files to include in the dataset.
    data_config : DataConfig
        Data configuration containing preprocessing parameters, including
        ``max_num_peaks``.
    lance_path : str
        Path at which to create the Lance dataset.
    force_rebuild : bool, optional
        If True, rebuild the Lance dataset even if it already exists at
        ``lance_path``. Default is False.

    Returns
    -------
    LanceMapDataset
        Map-style wrapper around the created Lance dataset.
    """

    if os.path.exists(lance_path) and not force_rebuild:
        print(f"Reusing existing Lance dataset at: {lance_path}")
        return LanceMapDataset(
            lance_path,
            seq_len=data_config.max_num_peaks,
        )
        
    # Spectrum preprocessing transforms
    preprocessing_fn = [
        preprocessing.filter_intensity(max_num_peaks=data_config.max_num_peaks),
        preprocessing.scale_intensity(scaling="root", max_intensity=1.0),
    ]
    
    # Custom field for extracting Retention time
    rt_field = CustomField(
        # The new column name:
        name="ret_time",
        # The function to extract the retention time:
        accessor=lambda x: x["scanList"]["scan"][0]["scan start time"],
        # The expected data type:
        dtype=pa.float64(),
    )

    # Batch size for SpectrumDataset class. Doesn't impact training batch size
    lance_batch_size = 256

    lance_dataset = None
    for mzml_file in mzml_files:
        df = spectra_to_df(
            mzml_file,
            metadata_df=None,
            ms_level=1,
            preprocessing_fn=preprocessing_fn,
            valid_charge=None,
            custom_fields=rt_field,
            progress=True,
        )
        if lance_dataset is None:
            lance_dataset = SpectrumDataset(df, path=lance_path, batch_size=lance_batch_size)
        else:
            lance_dataset.add_spectra(df)
        del df
    print("Created Lance dataset at:", lance_path)
        
    # Add LanceMapDataset wrapper for shuffle support
    dataset = LanceMapDataset(lance_path, seq_len=data_config.max_num_peaks)
    return dataset
