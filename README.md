# mass-screen-rs

The MUSE data-driven response mask ("mass screen") with a Rust core and a Python API.

```bash
pip install mass-screen-rs
```

```python
from mass_screen_rs import mass_screen

keep = mass_screen(response, spectra, threshold=2.24, fill_value=0.25)
```

`response` is the response matrix `[n, m]` (spectral pixel × VDEM component), `spectra` is
`[n, b]` (spectral pixel × image pixel). `keep` is a float32 array `[b, m]` holding 1.0 where a
component is kept and NaN where it is masked:

```text
keep[b, m] = 1    if  min_n(spectra[n, b] / response[n, m]) * max_n(response[n, m]) >= threshold
           = NaN  otherwise,
```

after replacing `spectra <= 0` by `fill_value` (which must be `>= 0`). A component whose
response is nowhere positive is always masked, as in the PyTorch reference implementation in
`muse.inversions`; the earlier C++ kernel kept those and masked every component whose largest
response was below `1e-9`.

## Performance

The kernel keeps only the positive entries of each response row (a MUSE response is ~3 % non-zero),
runs the rows that mask most first, and updates the flags of 64 image pixels per entry with one
vector compare. On a 16-core Ryzen 9 9950X:

| Inputs (n × m, b)                          | C++/OpenMP kernel | mass-screen-rs | torch (GPU) |
|--------------------------------------------|------------------:|---------------:|------------:|
| real run, 3072 × 109585, 3216 pixels       |            27.9 s |         0.19 s |        59 s |
| synthetic, 3072 × 246225, 4444 pixels      |           188.9 s |         0.85 s |             |
| synthetic, 3072 × 7035, 561 pixels         |            0.48 s |        0.006 s |             |

The masks agree with the torch reference on every cell. The published wheels target the x86-64
baseline (no `-march=native`) and are about 20 % slower than a native build. The kernel runs on all
cores through rayon; `RAYON_NUM_THREADS` caps it, and on SMT machines the physical core count is
slightly faster than the default.

## Development

```bash
pip install maturin
maturin develop --release   # build into the active environment
cargo test                  # Rust unit tests
pytest --pyargs mass_screen_rs
```

Wheels for Linux, macOS (arm64) and Windows are built by the GitHub workflow and published to PyPI
on a `v*` tag.
