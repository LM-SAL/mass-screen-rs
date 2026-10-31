0.2.1 (2026-10-31)
==================

New Features
------------

- Wheels are now published for Linux arm64 and for free-threaded CPython 3.14 and 3.15, where the module runs without the GIL, and one wheel per platform serves CPython 3.12 and later. On x86-64 CPUs with AVX2, the mask runs a build of its kernel compiled for AVX2. (`#16 <https://github.com/LM-SAL/mass-screen-rs/pull/16>`__)


Bug Fixes
---------

- Building mass-screen-rs from source now needs Rust 1.99 or later; macOS 27 refuses to load an extension stripped by Rust 1.97 or older ("mis-aligned LINKEDIT string pool"). (`#16 <https://github.com/LM-SAL/mass-screen-rs/pull/16>`__)


Documentation
-------------

- The user guide example now runs as written and is tested with the docs. (`#15 <https://github.com/LM-SAL/mass-screen-rs/pull/15>`__)


Release notes for 0.2.0 and earlier are on the `GitHub releases page <https://github.com/LM-SAL/mass-screen-rs/releases>`__.
