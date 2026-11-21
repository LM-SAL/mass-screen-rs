Design notes
============

The PyTorch Reference Defines The Mask
--------------------------------------

- **Owner:** Nabil Freij.
- **Code:** ``src/screen.rs::mass_screen_core`` (``dead_cols`` and ``scale``).
- **Tests:**
  ``python/mass_screen_rs/tests/test_torch_parity.py`` (a mask recorded from
  the PyTorch reference, and a live comparison when torch is
  installed), ``src/tests/screen.rs::zero_column_is_masked_and_fill_value_applies``.

Current Behavior
~~~~~~~~~~~~~~~~

The kernel computes ``keep[b, m] = all_n(spectra[n, b] >= rhs[n, m])`` with
``rhs = threshold / max_n(response) * response``, which is the reference rule
``min_n(spectra / response) * max_n(response) >= threshold`` rearranged. A component whose
response is nowhere positive is masked for every pixel.

Rationale
~~~~~~~~~

The old C++ mask kept columns with no response and masked columns with a positive
maximum below ``1e-9``. Those cases covered 13.7 % of mask cells on a real run.
Scaling a positive response column leaves the rule unchanged, so its magnitude
alone cannot disqualify it.

Cost
~~~~

Rounding can affect cells at the threshold. The formulas agreed on all
352 million cells of a real run and on the recorded test cases.

Sparse, Pixel-Major Kernel
--------------------------

- **Owner:** Nabil Freij.
- **Code:** ``src/screen.rs`` (``csr_row``, the task loop in ``mass_screen_core``).
- **Tests:**
  ``src/tests/screen.rs::many_tiles_and_blocks_match_the_dense_rule``,
  ``python/mass_screen_rs/tests/test_mass_screen.py::test_matches_reference``.

Current Behavior
~~~~~~~~~~~~~~~~

1. Each row stores the columns and values for ``rhs > 0``. Only these entries
   can mask a cell because the fill value is non-negative. A typical response
   is about 3 % non-zero.
2. Rows with larger sums of ``rhs`` run first to eliminate cells early. If the
   smallest spectrum value in a tile is at least the largest ``rhs`` in the
   row, the task can skip that row.
3. Each task handles 64 pixels by 8192 columns. A column's 64 adjacent flags
   let the compiler vectorize comparisons across pixels. The block uses 512 KiB.
4. After four blocks' worth of updates, the task checks whether every cell
   has failed and can stop if so.

Rationale
~~~~~~~~~

The old C++ mask scaled the whole response once and then, for every image pixel, walked every
entry of every row, recounting the surviving columns after each row. That streams the scaled
response (1.3 GB on a real run) once per pixel and does 97 % of its compares against zeros.

.. _screen-performance:

Performance
~~~~~~~~~~~

Times on a 16-core Ryzen 9 9950X using 32 threads.

===================================  =====================  ===============
Inputs (n × m, b)                    old C++/OpenMP mask    mass-screen-rs
===================================  =====================  ===============
real run, 3072 × 109585, 3216 px     27.9 s                 0.17 s
synthetic, 3072 × 246225, 4444 px    188.9 s                0.53 s
synthetic, 3072 × 7035, 561 px       0.48 s                 0.006 s
===================================  =====================  ===============

Before runtime AVX2 selection, the x86-64 wheels were about 30 % slower than a
``target-cpu=native`` build. The remaining gap has not been measured.

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

Worker pools
------------

``set_workers`` stores a Rayon pool in the current Python context. A call with the
same worker count reuses it. A different count creates a pool for that call.
``test_call_and_scoped_workers_reach_the_kernel`` checks which pool the binding
receives. ``the_kernel_inherits_the_selected_pool`` checks the thread count used for the mask's column maxima.

The worker helpers record the process ID on first use. A forked child inherits
the pool handles without their worker threads.
``test_forked_child_rejects_parent_workers`` covers implicit and explicit counts.
