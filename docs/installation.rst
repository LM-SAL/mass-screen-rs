Installation
============

Requirements
------------

- Python >= 3.12
- NumPy >= 2.0
- Rust >= 1.99 and `maturin <https://github.com/PyO3/maturin>`_ (source builds only)

From PyPI
---------

Wheels are published for Linux x86-64 and arm64, macOS arm64 and Windows x86-64. One wheel per
platform serves CPython 3.12 and later; free-threaded CPython 3.14 and 3.15 have their own wheels
and run without the GIL:

.. code-block:: bash

   pip install mass-screen-rs

From source
-----------

.. code-block:: bash

   git clone https://github.com/LM-SAL/mass-screen-rs
   cd mass-screen-rs
   pip install maturin
   maturin develop --release

On x86-64 the wheel selects AVX2 when the CPU supports it. To build for the
current CPU, use the following command. See the :ref:`performance measurements <screen-performance>` for timings.

.. code-block:: bash

   RUSTFLAGS="-C target-cpu=native" maturin develop --release

Development
-----------

.. code-block:: bash

   pip install maturin
   maturin develop --release --extras tests
   cargo test
   pytest --pyargs mass_screen_rs
   prek run --all-files

With ``torch`` installed, the parity test also checks a fresh random case against PyTorch
directly, and ``python -m mass_screen_rs.tests.make_torch_reference`` re-records the reference.

Releases: bump ``version`` in ``Cargo.toml`` (the only place it is set) and run
``towncrier build --version X.Y.Z``, which moves the fragments in ``changelog/`` into
``CHANGELOG.rst``; merge, then tag ``vX.Y.Z`` on ``main``; the workflow builds the wheels and the
sdist and uploads them to PyPI.
