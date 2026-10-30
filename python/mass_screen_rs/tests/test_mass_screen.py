"""
Tests for mass_screen.
"""

import numpy as np
import pytest

from mass_screen_rs import mass_screen


def reference(response, spectra, threshold, fill_value):
    """
    Evaluate the mask rule in numpy, one cell at a time.
    """
    response = np.asarray(response, dtype=np.float32)
    spectra = np.asarray(spectra, dtype=np.float32)
    spectra = np.where(spectra <= 0, np.float32(fill_value), spectra)
    col_max = response.max(axis=0)
    scale = np.divide(np.float32(threshold), col_max, out=np.zeros_like(col_max), where=col_max > 0)
    rhs = scale * response  # [n, m], same float32 arithmetic as the kernel
    kept = np.all(spectra[:, :, None] >= rhs[:, None, :], axis=0) & (col_max > 0)
    return np.where(kept, np.float32(1), np.float32(np.nan))


def random_case(rng, n, m, b, density=0.05):
    response = np.where(rng.random((n, m)) < density, rng.random((n, m)), 0).astype(np.float32)
    spectra = (rng.random((n, b)) * 4 - 1).astype(np.float32)  # about a quarter <= 0
    return response, spectra


def test_worked_example():
    response = np.array([[1, 0.5, 0.1], [0.2, 1, 0.3]], dtype=np.float32)
    spectra = np.array([[3, 1], [4, 5]], dtype=np.float32)
    keep = mass_screen(response, spectra, threshold=2.24, fill_value=1)
    assert keep.dtype == np.float32
    np.testing.assert_array_equal(keep, [[1, 1, 1], [np.nan, np.nan, 1]])


def test_zero_response_component_is_masked():
    response = np.array([[1, 0, 0.1], [0.5, 0, 0.3]], dtype=np.float32)
    spectra = np.array([[3, 4], [4, 5]], dtype=np.float32)
    keep = mass_screen(response, spectra, threshold=2.24, fill_value=1)
    np.testing.assert_array_equal(keep[:, 1], [np.nan, np.nan])
    np.testing.assert_array_equal(keep[:, [0, 2]], 1)


def test_fill_value_replaces_non_positive_spectra():
    response = np.array([[1.0], [1.0]], dtype=np.float32)
    spectra = np.array([[-5.0, 0.0], [3.0, 3.0]], dtype=np.float32)
    kept = mass_screen(response, spectra, threshold=2.0, fill_value=2.5)
    masked = mass_screen(response, spectra, threshold=2.0, fill_value=0.25)
    np.testing.assert_array_equal(kept, 1)
    np.testing.assert_array_equal(masked, np.nan)


@pytest.mark.parametrize(
    ("n", "m", "b"), [(5, 7, 3), (40, 300, 50), (12, 8192 * 2 + 5, 64 * 3 + 9)]
)
def test_matches_reference(n, m, b):
    rng = np.random.default_rng(n * m + b)
    response, spectra = random_case(rng, n, m, b)
    keep = mass_screen(response, spectra, threshold=2.24, fill_value=0.25)
    assert keep.shape == (b, m)
    np.testing.assert_array_equal(keep, reference(response, spectra, 2.24, 0.25))


def test_accepts_float64_and_non_contiguous_input():
    rng = np.random.default_rng(0)
    response, spectra = random_case(rng, 10, 20, 6)
    keep = mass_screen(
        response.astype(np.float64), spectra.T.copy().T, threshold=2.24, fill_value=1
    )
    np.testing.assert_array_equal(keep, reference(response, spectra, 2.24, 1))


@pytest.mark.parametrize(("m", "b"), [(0, 3), (4, 0)])
def test_empty(m, b):
    keep = mass_screen(
        np.ones((2, m), np.float32), np.ones((2, b), np.float32), threshold=2.24, fill_value=1
    )
    assert keep.shape == (b, m)


@pytest.mark.parametrize(
    ("response", "spectra", "kwargs", "match"),
    [
        (np.ones((2, 3)), np.ones((3, 2)), {"threshold": 1, "fill_value": 1}, "rows"),
        (np.ones((2, 3)), np.ones((2, 2)), {"threshold": 1, "fill_value": -1}, "fill_value"),
        (np.ones((2, 3)), np.ones((2, 2)), {"threshold": np.nan, "fill_value": 1}, "threshold"),
        (np.ones(3), np.ones((2, 2)), {"threshold": 1, "fill_value": 1}, "2-D"),
    ],
)
def test_rejects_bad_input(response, spectra, kwargs, match):
    with pytest.raises(ValueError, match=match):
        mass_screen(response, spectra, **kwargs)
