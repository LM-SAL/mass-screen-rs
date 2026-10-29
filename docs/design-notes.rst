Design notes
============

Design decisions and known limitations, one section per decision: what the code does, why, and
what it costs. Each section names the code and the tests that pin the behavior, so a change to
either is a change to the decision.

The PyTorch Reference Defines The Mask
--------------------------------------

- **Owner:** Nabil Freij.
- **Code:** ``src/screen.rs::mass_screen_core`` (``dead_cols`` and ``scale``).
- **Tests:**
  ``python/mass_screen_rs/tests/test_torch_parity.py`` (a mask recorded from
  ``muse.inversions.inversions._mass_screen_eager``, and a live comparison when torch is
  installed), ``src/tests/screen.rs::zero_column_is_masked_and_fill_value_applies``.

Current Behavior
~~~~~~~~~~~~~~~~

The kernel computes ``keep[b, m] = all_n(spectra[n, b] >= rhs[n, m])`` with
``rhs = threshold / max_n(response) * response``, which is the reference rule
``min_n(spectra / response) * max_n(response) >= threshold`` rearranged. A component whose
response is nowhere positive is masked for every pixel.

Rationale
~~~~~~~~~

The old C++ mask in muse diverged from the reference in two places: it kept every component with
no response at all, and it masked every component whose largest response was below ``1e-9`` (an
epsilon guarding a division). On a real MUSE run those two cases covered 13.7 % of the mask
cells. The reference is the documented algorithm, so the kernel follows it; the rule is
scale-invariant, so no epsilon is needed.

Cost
~~~~

The two formulas round differently, so a cell exactly on the threshold could in principle differ
from the reference. None did on a real run of 352 million cells, nor on the recorded cases.

Sparse, Pixel-Major Kernel
--------------------------

- **Owner:** Nabil Freij.
- **Code:** ``src/screen.rs`` (``csr_row``, the task loop in ``mass_screen_core``).
- **Tests:**
  ``src/tests/screen.rs::many_tiles_and_blocks_match_the_dense_rule``,
  ``python/mass_screen_rs/tests/test_mass_screen.py::test_matches_reference``.

Current Behavior
~~~~~~~~~~~~~~~~

1. Only the entries with ``rhs > 0`` are kept, per row, as (column, value) pairs. With
   ``fill_value >= 0`` every screened spectrum value is non-negative, so a zero or negative entry
   can never mask a cell. A MUSE response is about 3 % non-zero.
2. Rows run in order of decreasing sum of ``rhs``, so the rows that mask most run first, and a
   row is skipped for a tile of pixels when the tile's smallest spectrum value is at least the
   row's largest ``rhs``.
3. The mask flags are laid out ``[column][64 pixels]``: one response entry updates the flags of 64
   image pixels with one vector compare, whatever the column order of the response. Work is split
   into tasks of 64 pixels × 8192 columns so a task's flag block (512 KiB) stays in L2.
4. A task stops early once every column of its block has failed for every pixel of its tile; the
   check runs once per four blocks' worth of updates.

Rationale
~~~~~~~~~

The old C++ mask scaled the whole response once and then, for every image pixel, walked every
entry of every row, recounting the surviving columns after each row. That streams the scaled
response (1.3 GB on a real run) once per pixel and does 97 % of its compares against zeros.

Performance
~~~~~~~~~~~

On a 16-core Ryzen 9 9950X, 32 threads:

===================================  =====================  ===============
Inputs (n × m, b)                    old C++/OpenMP mask    mass-screen-rs
===================================  =====================  ===============
real run, 3072 × 109585, 3216 px     27.9 s                 0.17 s
synthetic, 3072 × 246225, 4444 px    188.9 s                0.53 s
synthetic, 3072 × 7035, 561 px       0.48 s                 0.006 s
===================================  =====================  ===============

The x86-64 wheels run a build of the task loop compiled for AVX2 on CPUs that have it (see "AVX2
At Runtime"). Before that build was added, they were about 30 % slower than a
``target-cpu=native`` build; how much of that gap it closes has not been measured.

Cost
~~~~

The compressed rows and per-task flag blocks are extra memory: about 8 bytes per positive
response entry, plus 512 KiB per worker thread.

AVX2 At Runtime
---------------

- **Owner:** Nabil Freij.
- **Code:** ``src/screen.rs`` (``Tasks::run``, ``Tasks::run_avx2``, ``has_avx2``).
- **Tests:** ``src/tests/screen.rs::avx2_build_matches_the_baseline_build``.

Current Behavior
~~~~~~~~~~~~~~~~

On x86-64 the task loop is compiled twice, for baseline x86-64 and for AVX2. Each call checks once
whether the CPU has AVX2 and, if it does, runs the AVX2 build. The two builds give bit-identical
masks: the loop only compares, takes minima and sets flags, so the vector width changes no result.
Other architectures have one build.

Rationale
~~~~~~~~~

The wheels have to run on any x86-64 CPU, so they cannot be compiled for ``target-cpu=native``.
The flag updates are the hot loop, and AVX2 doubles their vector width over the SSE2 of baseline
x86-64.

Cost
~~~~

The task loop is in the extension twice, and ``mass_screen_with`` takes a flag the tests use to
force the baseline build. Features beyond AVX2 that a ``target-cpu=native`` build may use, such as
AVX-512, are not used.
