//! Data-driven response mask for spectral inversions.
mod api;
mod screen;
mod workers;

#[cfg(test)]
mod tests;

use pyo3::prelude::{pymodule, Bound, PyModule, PyResult, Python};
use pyo3::types::PyModuleMethods;
use pyo3::wrap_pyfunction;

#[pymodule(gil_used = false)]
#[pyo3(name = "_mass_screen_rs")]
fn mass_screen_rs(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<workers::WorkerPool>()?;
    module.add_function(wrap_pyfunction!(api::mass_screen, module)?)?;
    Ok(())
}
