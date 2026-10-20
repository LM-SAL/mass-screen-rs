# mass-screen-rs

The MUSE data-driven response mask ("mass screen") with a Rust core and a Python API.

`mass_screen(response, spectra, threshold=..., fill_value=...)` takes the response matrix
`[n, m]` (spectral pixel × VDEM component) and the spectra `[n, b]` (spectral pixel × image
pixel) and returns a float32 array `[b, m]` with 1.0 where a component can produce at least
`threshold` of signal for that pixel and NaN where it cannot, after replacing spectra `<= 0` by
`fill_value`. It computes the same mask as the PyTorch reference in `muse.inversions` and
replaces the C++ kernel muse used to build at install time. The tests check it against a recorded
PyTorch mask.

## Install

```bash
pip install mass-screen-rs
```

Wheels are published for Linux (x86-64), macOS (arm64) and Windows, Python 3.12 to 3.14.

## Development

```bash
pip install maturin pytest
maturin develop --release        # build into the active environment
cargo test                       # Rust unit tests
pytest --pyargs mass_screen_rs   # Python tests, including parity with the recorded PyTorch mask
prek run --all-files             # lint hooks (ruff, docformatter, cargo fmt, clippy, cargo test)
```

With `torch` installed, the parity test also checks a fresh random case against PyTorch directly,
and `python -m mass_screen_rs.tests.make_torch_reference` re-records the reference.

Releases: tag `vX.Y.Z` on `main`; the workflow builds the wheels and the sdist and uploads them to
PyPI.
