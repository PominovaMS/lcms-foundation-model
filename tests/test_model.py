import pytest
import torch

from source.model import MS1Encoder


@pytest.fixture
def batch():
    """A mass spectrum."""
    spectra = torch.tensor(
        [
            [[100.1, 0.1], [200.2, 0.2], [300.3, 0.3]],
            [[400.4, 0.4], [500, 0.5], [0, 0]],
        ]
    )

    batch_dict = {
        "mz_array": spectra[:, :, 0],
        "intensity_array": spectra[:, :, 1],
    }

    return batch_dict


def test_peaks_mask(batch):
    """Test random peaks mask is created."""
    model = MS1Encoder(d_model=8, nhead=1, dim_feedforward=12, n_layers=1)
    mask = model.get_peaks_mask(batch["intensity_array"])
    assert mask.shape == (2, 3)


def test_forward(batch):
    """Test peaks encoding."""
    model = MS1Encoder(d_model=8, nhead=1, dim_feedforward=12, n_layers=1)
    peak_embs = model.forward(
        mzs=batch["mz_array"], intensities=batch["intensity_array"]
    )
    assert peak_embs.shape == (2, 3, model.d_model)


def test_training_step(batch):
    """Test traning step is performed (basic)."""
    model = MS1Encoder(d_model=8, nhead=1, dim_feedforward=12, n_layers=1)
    loss = model.training_step(batch, 0)
    assert torch.isreal(loss)
