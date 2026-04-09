import numpy as np
import lance
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
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
