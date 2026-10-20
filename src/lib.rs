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
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::{pyfunction, pymodule, Bound, PyModule, PyResult, Python};
use pyo3::types::PyModuleMethods;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use std::slice;

/// Image pixels per tile: a column's flags are one 64-byte line, one f32 compare covers them.
const BT: usize = 64;
/// Columns per block: the fail block is MBLK * BT = 512 KiB, which stays in L2 across all rows.
const MBLK: usize = 8192;

fn column_max(rg: &[f32], m: usize) -> Vec<f32> {
    let merge = |mut a: Vec<f32>, b: &[f32]| {
        for (x, &y) in a.iter_mut().zip(b) {
            if y > *x {
                *x = y;
            }
        }
        a
    };
    // A partial maximum per chunk of rows, not per row: each partial is an m-float vector.
    let rows_per_chunk = (rg.len() / m)
        .div_ceil(4 * rayon::current_num_threads())
        .max(1);
    rg.par_chunks(m * rows_per_chunk)
        .map(|chunk| chunk.chunks(m).fold(vec![f32::NEG_INFINITY; m], merge))
        .reduce(|| vec![f32::NEG_INFINITY; m], |a, b| merge(a, &b))
}

/// One response row as (column, rhs) pairs for the entries with rhs > 0, plus block offsets.
struct CsrRow {
    idx: Vec<u32>,
    val: Vec<f32>,
    /// `blk[q]..blk[q + 1]` is the slice of `idx`/`val` inside column block `q`.
    blk: Vec<u32>,
    /// Sum of `val`: rows with large rhs mask cells earliest, so they run first.
    score: f64,
    max_val: f32,
}

fn csr_row(rg_row: &[f32], scale: &[f32], nblocks: usize) -> CsrRow {
    let mut idx = Vec::new();
    let mut val = Vec::new();
    let mut score = 0.0f64;
    let mut max_val = 0.0f32;
    for (j, (&r, &s)) in rg_row.iter().zip(scale).enumerate() {
        let v = s * r;
        if v > 0.0 {
            idx.push(j as u32);
            val.push(v);
            score += f64::from(v);
            max_val = max_val.max(v);
        }
    }
    let blk = (0..=nblocks)
        .map(|q| idx.partition_point(|&j| (j as usize) < q * MBLK) as u32)
        .collect();
    CsrRow {
        idx,
        val,
        blk,
        score,
        max_val,
    }
}

struct SyncPtr(*mut f32);
unsafe impl Sync for SyncPtr {}
unsafe impl Send for SyncPtr {}
impl SyncPtr {
    // A method keeps the closure capturing the wrapper, not the bare pointer.
    unsafe fn add(&self, off: usize) -> *mut f32 {
        self.0.add(off)
    }
}

/// Writes the mask into `keep` (`[b, m]`, fully overwritten). `rg` is `[n, m]` and `ys` is
/// `[n, b]`, all C order. Columns with no positive response are masked. Needs `fill_value >= 0`,
/// so that every screened value is non-negative and the zero entries of `rg` can be skipped.
#[allow(clippy::too_many_arguments)]
pub fn mass_screen_core(
    rg: &[f32],
    ys: &[f32],
    keep: &mut [f32],
    n: usize,
    m: usize,
    b: usize,
    threshold: f32,
    fill_value: f32,
) {
    assert_eq!(rg.len(), n * m);
    assert_eq!(ys.len(), n * b);
    assert_eq!(keep.len(), b * m);
    assert!(fill_value >= 0.0);
    if b == 0 || m == 0 {
        return;
    }
    let col_max = column_max(rg, m);
    let dead_cols: Vec<u32> = (0..m)
        .filter(|&j| col_max[j].is_nan() || col_max[j] <= 0.0)
        .map(|j| j as u32)
        .collect();
    let scale: Vec<f32> = col_max
        .iter()
        .map(|&mx| if mx > 0.0 { threshold / mx } else { 0.0 })
        .collect();
    let nblocks = m.div_ceil(MBLK);
    let ntiles = b.div_ceil(BT);
    let rows: Vec<CsrRow> = rg
        .par_chunks(m)
        .map(|r| csr_row(r, &scale, nblocks))
        .collect();
    let mut order: Vec<usize> = (0..n).filter(|&i| !rows[i].idx.is_empty()).collect();
    order.sort_by(|&i, &j| rows[j].score.total_cmp(&rows[i].score));
    let keep_ptr = SyncPtr(keep.as_mut_ptr());

    // One flag block per worker, reused across its tasks: a fresh zeroed block per task costs
    // a page fault per 4 KiB and showed as a 50 % slowdown on 32 threads.
    (0..ntiles * nblocks).into_par_iter().for_each_init(
        || vec![0u8; MBLK * BT],
        |fail, task| {
            let (t, q) = (task / nblocks, task % nblocks);
            let b0 = t * BT;
            let bt = BT.min(b - b0);
            let m0 = q * MBLK;
            let mb = MBLK.min(m - m0);
            // fail[j * BT + k] != 0 once column m0 + j is masked for image pixel b0 + k.
            let fail = &mut fail[..mb * BT];
            fail.fill(0);
            let dead_lo = dead_cols.partition_point(|&j| (j as usize) < m0);
            let dead_hi = dead_cols.partition_point(|&j| (j as usize) < m0 + mb);
            for &j in &dead_cols[dead_lo..dead_hi] {
                fail[(j as usize - m0) * BT..][..BT].fill(1);
            }
            let mut ybuf = [f32::INFINITY; BT]; // lanes past bt never fail
            let mut since_recount = 0usize;
            for &i in &order {
                let row = &rows[i];
                let (lo, hi) = (row.blk[q] as usize, row.blk[q + 1] as usize);
                if lo == hi {
                    continue;
                }
                let mut ymin = f32::INFINITY;
                for (yb, &y) in ybuf[..bt].iter_mut().zip(&ys[i * b + b0..i * b + b0 + bt]) {
                    *yb = if y <= 0.0 { fill_value } else { y };
                    ymin = ymin.min(*yb);
                }
                if ymin >= row.max_val {
                    continue; // no pixel of this tile is below any rhs of this row
                }
                for (&j, &v) in row.idx[lo..hi].iter().zip(&row.val[lo..hi]) {
                    let f = &mut fail[(j as usize - m0) * BT..][..BT];
                    for (f, &y) in f.iter_mut().zip(&ybuf) {
                        *f |= u8::from(y < v);
                    }
                }
                // Stop once every column of the block has failed for every pixel of the tile. The
                // check costs one pass over fail, so it runs once per ~4 passes' worth of updates.
                since_recount += hi - lo;
                if since_recount >= 4 * mb {
                    since_recount = 0;
                    let mut all_failed = [1u8; BT];
                    for col in fail.chunks_exact(BT) {
                        for (a, &f) in all_failed.iter_mut().zip(col) {
                            *a &= f;
                        }
                    }
                    if all_failed[..bt].iter().all(|&a| a == 1) {
                        break;
                    }
                }
            }
            for k in 0..bt {
                // SAFETY: tasks write disjoint rectangles [b0 + k, m0..m0 + mb] of keep.
                let out = unsafe { slice::from_raw_parts_mut(keep_ptr.add((b0 + k) * m + m0), mb) };
                for (j, o) in out.iter_mut().enumerate() {
                    *o = if fail[j * BT + k] != 0 { f32::NAN } else { 1.0 };
                }
            }
        },
    );
}

/// Mask the VDEM components a set of spectra cannot constrain.
///
/// `response` is `[n, m]` and `spectra` is `[n, b]`, both C-contiguous float32. Returns a float32
/// array of shape `[b, m]` holding 1.0 where a component is kept and NaN where it is masked:
/// component `m` is kept for pixel `b` when `min_n(spectra[n, b] / response[n, m]) *
/// max_n(response[n, m]) >= threshold`, after replacing `spectra <= 0` by `fill_value`. A
/// component whose response is nowhere positive is always masked.
#[pyfunction]
#[pyo3(signature = (response, spectra, threshold, fill_value))]
fn mass_screen<'py>(
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

#[pymodule]
#[pyo3(name = "_mass_screen_rs")]
fn mass_screen_rs(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(mass_screen, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(rg: &[f32], ys: &[f32], n: usize, m: usize, b: usize, fill: f32) -> Vec<f32> {
        let mut keep = vec![0.0; b * m];
        mass_screen_core(rg, ys, &mut keep, n, m, b, 2.24, fill);
        keep
    }

    fn same(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()))
    }

    #[test]
    fn muse_example() {
        // The case muse/inversions/tests/test_mass_screen.py checks.
        let rg = [1.0, 0.5, 0.1, 0.2, 1.0, 0.3];
        let ys = [3.0, 1.0, 4.0, 5.0];
        let keep = run(&rg, &ys, 2, 3, 2, 1.0);
        assert!(same(&keep, &[1.0, 1.0, 1.0, f32::NAN, f32::NAN, 1.0]));
    }

    #[test]
    fn zero_column_is_masked_and_fill_value_applies() {
        // Column 1 has no response; ys[0, 1] <= 0 is replaced by the fill value 1.0.
        let rg = [1.0, 0.0, 0.1, 0.5, 0.0, 0.3];
        let ys = [3.0, -1.0, 4.0, 5.0];
        let keep = run(&rg, &ys, 2, 3, 2, 1.0);
        assert!(same(&keep, &[1.0, f32::NAN, 1.0, f32::NAN, f32::NAN, 1.0]));
    }

    #[test]
    fn many_tiles_and_blocks_match_the_dense_rule() {
        // Exercises partial tiles (b % BT != 0) and several column blocks.
        let (n, m, b) = (7, 2 * MBLK + 5, 3 * BT + 9);
        let mut seed = 12345u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 40) as f32 / (1u64 << 24) as f32
        };
        let rg: Vec<f32> = (0..n * m)
            .map(|_| if next() < 0.05 { next() } else { 0.0 })
            .collect();
        let ys: Vec<f32> = (0..n * b).map(|_| next() * 4.0 - 1.0).collect();
        let keep = run(&rg, &ys, n, m, b, 0.25);
        let col_max = column_max(&rg, m);
        for bi in 0..b {
            for j in 0..m {
                let kept = col_max[j] > 0.0
                    && (0..n).all(|i| {
                        let y = if ys[i * b + bi] <= 0.0 {
                            0.25
                        } else {
                            ys[i * b + bi]
                        };
                        y >= 2.24 / col_max[j] * rg[i * m + j]
                    });
                assert_eq!(keep[bi * m + j] == 1.0, kept, "cell ({bi}, {j})");
            }
        }
    }
}
