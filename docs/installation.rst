Installation
============

Requirements
------------

- Python >= 3.12
- NumPy >= 2.0
- Rust >= 1.99 and `maturin <https://github.com/PyO3/maturin>`_ (source builds only)

From PyPI
---------

Wheels are published for Linux x86-64, macOS arm64 and Windows x86-64:

.. code-block:: bash

   pip install mass-screen-rs

From source
-----------

.. code-block:: bash

   git clone https://github.com/LM-SAL/mass-screen-rs
   cd mass-screen-rs
   pip install maturin
   maturin develop --release

The wheels target baseline x86-64 so they run everywhere. A source build with

.. code-block:: bash

   RUSTFLAGS="-C target-cpu=native" maturin develop --release

is about 30 % faster on the machine it is built on; see "Performance" in the design notes.

Development
-----------

.. code-block:: bash

   pip install maturin
   maturin develop --release --extras tests  # build into the active environment
   cargo test                                # Rust unit tests
   pytest --pyargs mass_screen_rs            # Python tests, including parity with the recorded PyTorch mask
   prek run --all-files                      # lint hooks (ruff, docformatter, cargo fmt, clippy, cargo test)

With ``torch`` installed, the parity test also checks a fresh random case against PyTorch
directly, and ``python -m mass_screen_rs.tests.make_torch_reference`` re-records the reference.

Releases: bump ``version`` in ``Cargo.toml`` (the only place it is set) and run
``towncrier build --version X.Y.Z``, which moves the fragments in ``changelog/`` into
``CHANGELOG.rst``; merge, then tag ``vX.Y.Z`` on ``main``; the workflow builds the wheels and the
sdist and uploads them to PyPI.
