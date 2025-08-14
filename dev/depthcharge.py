import polars as pl
import depthcharge as dc
from torch.utils.data import DataLoader
from depthcharge.transformers import SpectrumTransformerEncoder
import torch

mzml_file = "/Users/adams/Projects/MoSTERT-rescoring/chick-data/mzml/b1906_293T_proteinID_01A_QE3_122212.mzML"

# Create a DataFrame containing the parsed MS1 spectra
df_ms1 = dc.data.spectra_to_df(
    mzml_file,
    metadata_df=None,
    ms_level=1,
    preprocessing_fn=None,
    valid_charge=None,
    custom_fields=None,
    progress=True,
)

# Convert the polars DataFrame to a depthcharge SpectrumDataset
ms1_dataset = dc.data.SpectrumDataset(df_ms1, batch_size=4)

ms1_dataset.n_spectra
ms1_dataset.peak_files
ms1_dataset.path
ms1_dataset[0]

loader = DataLoader(ms1_dataset, batch_size=None)

for scan in loader:
    print(scan["mz_array"].shape, scan["intensity_array"].shape)
    print(scan["scan_id"])
    print(scan["precursor_charge"])
    break

encoder = dc.encoders.PeakEncoder(
    d_model=128,  # size of embedding vectors
    min_mz_wavelength=0.001,
    max_mz_wavelength=10000,
    min_intensity_wavelength=1e-06,
    max_intensity_wavelength=1,
    learnable_wavelengths=False,
)


for scan in loader:
    mz = scan["mz_array"].squeeze(0)
    intensities = scan["intensity_array"].squeeze(0)
    peaks = torch.stack([mz, intensities], dim=-1)
    encoded = encoder(peaks)
    print("Encoded shape:", encoded.shape)
    break
