import polars as pl
import depthcharge as dc
from torch.utils.data import DataLoader
from depthcharge.transformers import SpectrumTransformerEncoder
import torch

mzml_file = "/Users/adams/Projects/MoSTERT-rescoring/chick-data/mzml/b1906_293T_proteinID_01A_QE3_122212.mzML"

df_ms1 = dc.data.spectra_to_df(
    mzml_file,
    metadata_df=None,
    ms_level=1,
    preprocessing_fn=None,
    valid_charge=None,
    custom_fields=None,
    progress=True,
)

len(df_ms1)
type(df_ms1)

ms1_dataset = dc.data.SpectrumDataset(df_ms1, batch_size=2)

ms1_dataset.n_spectra
ms1_dataset.peak_files
ms1_dataset.path
ms1_dataset[0]

loader = DataLoader(ms1_dataset, batch_size=4)

encoder = dc.encoders.PeakEncoder(
    d_model=128,
    min_mz_wavelength=0.001,
    max_mz_wavelength=10000,
    min_intensity_wavelength=1e-06,
    max_intensity_wavelength=1,
    learnable_wavelengths=False,
)

for spectra in loader:
    print(spectra)
    break

# Getting a TypeError: SpectrumDataset._to_tensor() got an unexpected keyword argument 'hf_converter'
encoded = encoder(df_ms1)
