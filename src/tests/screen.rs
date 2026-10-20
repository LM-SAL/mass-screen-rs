use crate::screen::{column_max, mass_screen_core, BT, MBLK};

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
