//! Python entry point: validation, the output array, and the GIL release around the kernel.
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::{pyfunction, Bound, PyResult, Python};

use crate::screen::mass_screen_core;

/// Mask the VDEM components a set of spectra cannot constrain.
///
/// `response` is `[n, m]` and `spectra` is `[n, b]`, both C-contiguous float32. Returns a float32
/// array of shape `[b, m]` holding 1.0 where a component is kept and NaN where it is masked:
/// component `m` is kept for pixel `b` when `min_n(spectra[n, b] / response[n, m]) *
/// max_n(response[n, m]) >= threshold`, after replacing `spectra <= 0` by `fill_value`. A
/// component whose response is nowhere positive is always masked.
#[pyfunction]
#[pyo3(signature = (response, spectra, threshold, fill_value))]
pub(crate) fn mass_screen<'py>(
    py: Python<'py>,
    response: PyReadonlyArray2<'py, f32>,
    spectra: PyReadonlyArray2<'py, f32>,
    threshold: f32,
    fill_value: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    if !threshold.is_finite() {
        return Err(PyValueError::new_err("threshold must be finite"));
    }
    if !fill_value.is_finite() || fill_value < 0.0 {
        return Err(PyValueError::new_err("fill_value must be finite and >= 0"));
    }
    let [n, m] = [response.shape()[0], response.shape()[1]];
    let [n_ys, b] = [spectra.shape()[0], spectra.shape()[1]];
    if n_ys != n {
        return Err(PyValueError::new_err(format!(
            "response has {n} rows but spectra has {n_ys}; both are indexed by spectral pixel"
        )));
    }
    let rg = response
        .as_slice()
        .map_err(|_| PyValueError::new_err("response must be a C-contiguous float32 array"))?;
    let ys = spectra
        .as_slice()
        .map_err(|_| PyValueError::new_err("spectra must be a C-contiguous float32 array"))?;
    // SAFETY: the array is uninitialised until mass_screen_core has written every element,
    // and nothing reads it before that.
    let out = unsafe { PyArray2::<f32>::new(py, [b, m], false) };
    let keep = unsafe { out.as_slice_mut() }
        .map_err(|_| PyValueError::new_err("could not allocate the mask"))?;
    py.detach(|| mass_screen_core(rg, ys, keep, n, m, b, threshold, fill_value));
    Ok(out)
}
