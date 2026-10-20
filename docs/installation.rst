Installation
============

Requirements
------------

- Python >= 3.13
- NumPy >= 2.0
- Rust >= 1.85 (build only)

From PyPI
---------

.. code-block:: bash

   pip install mass-screen-rs

Wheels are published for Linux (x86-64), macOS (arm64) and Windows, Python 3.13 to 3.15.

From source
-----------

Requires Rust >= 1.85 and `maturin <https://github.com/PyO3/maturin>`_.

.. code-block:: bash

   git clone https://github.com/LM-SAL/mass-screen-rs
   cd mass-screen-rs
   pip install maturin
   maturin develop --release

The wheel targets baseline x86-64 so it runs everywhere. A source build with

.. code-block:: bash

   RUSTFLAGS="-C target-cpu=native" maturin develop --release

is about 30 % faster on the machine it is built on; see "Performance" in the design notes.
