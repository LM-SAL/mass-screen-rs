//! The mask kernel: compressed positive entries per row, flags laid out per column for 64 pixels.
use rayon::prelude::*;
use std::slice;

/// Image pixels per tile: a column's flags are one 64-byte line, one f32 compare covers them.
pub(crate) const BT: usize = 64;
/// Columns per block: the fail block is MBLK * BT = 512 KiB, which stays in L2 across all rows.
pub(crate) const MBLK: usize = 8192;

pub(crate) fn column_max(rg: &[f32], m: usize) -> Vec<f32> {
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

/// Whether this CPU can run the AVX2 build of the task loop.
fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Writes the mask into `keep` (`[b, m]`, fully overwritten). `rg` is `[n, m]` and `ys` is
/// `[n, b]`, all C order. Columns with no positive response are masked. Needs `fill_value >= 0`,
/// so that every screened value is non-negative and the zero entries of `rg` can be skipped.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mass_screen_core(
    rg: &[f32],
    ys: &[f32],
    keep: &mut [f32],
    n: usize,
    m: usize,
    b: usize,
    threshold: f32,
    fill_value: f32,
) {
    mass_screen_with(rg, ys, keep, n, m, b, threshold, fill_value, true);
}

/// `mass_screen_core`, running the AVX2 build of the task loop when `allow_avx2` is set and the
/// CPU has AVX2, and the baseline build otherwise.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mass_screen_with(
    rg: &[f32],
    ys: &[f32],
    keep: &mut [f32],
    n: usize,
    m: usize,
    b: usize,
    threshold: f32,
    fill_value: f32,
    allow_avx2: bool,
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
    let tasks = Tasks {
        rows: &rows,
        order: &order,
        dead_cols: &dead_cols,
        ys,
        keep: SyncPtr(keep.as_mut_ptr()),
        m,
        b,
        nblocks,
        fill_value,
    };
    let avx2 = allow_avx2 && has_avx2();

    // One flag block per worker, reused across its tasks: a fresh zeroed block per task costs
    // a page fault per 4 KiB and showed as a 50 % slowdown on 32 threads.
    (0..ntiles * nblocks).into_par_iter().for_each_init(
        || vec![0u8; MBLK * BT],
        |fail, task| {
            if avx2 {
                // SAFETY: avx2 is only set on an x86-64 CPU that has AVX2.
                #[cfg(target_arch = "x86_64")]
                return unsafe { tasks.run_avx2(fail, task) };
            }
            tasks.run(fail, task);
        },
    );
}

/// What every task reads, and the mask, which tasks write in disjoint rectangles.
struct Tasks<'a> {
    rows: &'a [CsrRow],
    order: &'a [usize],
    dead_cols: &'a [u32],
    ys: &'a [f32],
    keep: SyncPtr,
    m: usize,
    b: usize,
    nblocks: usize,
    fill_value: f32,
}

impl Tasks<'_> {
    /// Masks tile `task / nblocks` of `BT` image pixels against column block `task % nblocks`,
    /// with `fail` as scratch. Always inlined, so `run_avx2` gets its own build of it.
    #[inline(always)]
    fn run(&self, fail: &mut [u8], task: usize) {
        let Tasks {
            rows,
            order,
            dead_cols,
            ys,
            keep: SyncPtr(keep),
            m,
            b,
            nblocks,
            fill_value,
        } = *self;
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
        for &i in order {
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
                for col in fail.as_chunks::<BT>().0 {
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
            let out = unsafe { slice::from_raw_parts_mut(keep.add((b0 + k) * m + m0), mb) };
            for (j, o) in out.iter_mut().enumerate() {
                *o = if fail[j * BT + k] != 0 { f32::NAN } else { 1.0 };
            }
        }
    }

    /// `run` built for AVX2, whose 256-bit vectors update twice as many flags per instruction as
    /// the 128-bit SSE2 of baseline x86-64.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    fn run_avx2(&self, fail: &mut [u8], task: usize) {
        self.run(fail, task);
    }
}
