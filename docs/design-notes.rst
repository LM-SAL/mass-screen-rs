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

The reference is the documented algorithm, so the kernel follows it, edge cases included: a
component with no positive response is masked, and no epsilon guards a small response, because
the rule is scale-invariant and needs none.

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

A dense pass, walking every entry of every row for every image pixel, streams the scaled
response (1.3 GB on a real run) once per pixel and does 97 % of its compares against zeros.
Keeping only the positive entries removes the zero compares, and updating 64 pixels per entry
reads the response once per tile of pixels rather than once per pixel.

Performance
~~~~~~~~~~~

On a 16-core Ryzen 9 9950X, 32 threads, masks identical in every cell:

===================================  ===============  ==============================
Inputs (n × m, b)                    mass-screen-rs   same algorithm in C++/OpenMP
===================================  ===============  ==============================
real run, 3072 × 109585, 3216 px     0.17 s           0.19 s
synthetic, 3072 × 246225, 4444 px    0.53 s           0.53 s
synthetic, 3072 × 7035, 561 px       0.006 s          0.018 s
===================================  ===============  ==============================

The algorithm sets the speed, not the language: the same algorithm in C++/OpenMP is no faster.
The published wheels target baseline x86-64 and are about 30 % slower than a
``target-cpu=native`` build.

Cost
~~~~

The compressed rows and per-task flag blocks are extra memory: about 8 bytes per positive
response entry, plus 512 KiB per worker thread.
