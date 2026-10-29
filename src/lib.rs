//! Data-driven response mask for the MUSE inversions.
//!
//! For a response matrix `rg` of shape `[n, m]` (spectral pixel × VDEM component) and spectra `ys`
//! of shape `[n, b]` (spectral pixel × image pixel), the mask is
//!
//! ```text
//! keep[b, m] = 1    if  min_n(ys[n, b] / rg[n, m]) * max_n(rg[n, m]) >= threshold
//!            = NaN  otherwise,
//! ```
//!
//! with `ys <= 0` replaced by `fill_value` first. Rewritten as `ys[n, b] >= rhs[n, m]` for every
//! `n`, with `rhs = threshold / max_n(rg) * rg`, only the entries with `rhs > 0` can mask a cell,
//! and a MUSE response is a few percent non-zero. The kernel keeps those entries per row, orders the
//! rows so the ones that mask most run first, and updates the flags of 64 image pixels per entry
//! with one vector compare.
mod api;
mod screen;

#[cfg(test)]
mod tests;

use pyo3::prelude::{pymodule, Bound, PyModule, PyResult, Python};
use pyo3::types::PyModuleMethods;
use pyo3::wrap_pyfunction;

#[pymodule(gil_used = false)]
#[pyo3(name = "_mass_screen_rs")]
fn mass_screen_rs(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(api::mass_screen, module)?)?;
    Ok(())
}
