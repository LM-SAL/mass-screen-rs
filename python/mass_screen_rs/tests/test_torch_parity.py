"""
The kernel reproduces the PyTorch reference mask.
"""

import numpy as np
import pytest

from mass_screen_rs import mass_screen
from mass_screen_rs.tests.make_torch_reference import CASES, DATA, make_case, torch_mask


@pytest.mark.parametrize("name", list(CASES))
def test_matches_recorded_torch_mask(name):
    with np.load(DATA) as data:
        response, spectra = data[f"{name}/response"], data[f"{name}/spectra"]
        kept, (threshold, fill_value) = data[f"{name}/kept"], data[f"{name}/params"]

    keep = mass_screen(response, spectra, threshold=threshold, fill_value=fill_value)

    assert keep.shape == kept.shape
    np.testing.assert_array_equal(keep == 1, kept)
    np.testing.assert_array_equal(np.isnan(keep), ~kept)


def test_matches_torch_live():
    pytest.importorskip("torch")
    response, spectra = make_case(30, 3000, 70, 0.05, seed=3)

    keep = mass_screen(response, spectra, threshold=2.24, fill_value=0.25)

    np.testing.assert_array_equal(keep, torch_mask(response, spectra, 2.24, 0.25))
