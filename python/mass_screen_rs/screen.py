"""
Python API for the mass-screen-rs Rust backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._mass_screen_rs import mass_screen as _mass_screen

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = ["mass_screen"]


def mass_screen(
    response: ArrayLike, spectra: ArrayLike, *, threshold: float, fill_value: float
) -> NDArray[np.float32]:
    """
    Mask the VDEM components a set of spectra cannot constrain.

    Component ``m`` is kept for image pixel ``b`` when
    ``min_n(spectra[n, b] / response[n, m]) * max_n(response[n, m]) >= threshold``,
    after replacing ``spectra <= 0`` by ``fill_value``. A component whose response is
    nowhere positive is always masked.

    Parameters
    ----------
    response : array_like
        Response matrix of shape ``[n, m]``: spectral pixel × VDEM component.
    spectra : array_like
        Spectra of shape ``[n, b]``: spectral pixel × image pixel.
    threshold : float
        Smallest signal, in the units of ``spectra``, a component must be able to produce.
    fill_value : float
        Replaces the ``spectra`` values that are ``<= 0``; must be ``>= 0``.

    Returns
    -------
    numpy.ndarray
        float32 array of shape ``[b, m]``: 1.0 where a component is kept, NaN where masked.

    Notes
    -----
    The kernel runs on all cores through rayon; ``RAYON_NUM_THREADS`` caps it.
    """
    response = np.ascontiguousarray(response, dtype=np.float32)
    spectra = np.ascontiguousarray(spectra, dtype=np.float32)
    if (response.ndim, spectra.ndim) != (2, 2):
        msg = f"response and spectra must be 2-D, got shapes {response.shape} and {spectra.shape}"
        raise ValueError(msg)
    return _mass_screen(response, spectra, float(threshold), float(fill_value))
