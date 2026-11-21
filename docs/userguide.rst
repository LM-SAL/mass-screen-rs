User Guide
==========

Masking a response matrix
-------------------------

Each row of the response matrix and spectra represents a spectral pixel.
Their columns represent VDEM components and image pixels, respectively.
:func:`~mass_screen_rs.mass_screen` returns one float32 row per image pixel,
with ``1.0`` for each component kept and ``NaN`` for each component masked.

.. code-block:: python

   >>> import numpy as np
   >>> from mass_screen_rs import mass_screen

   >>> rng = np.random.default_rng(0)
   >>> response = rng.random((30, 70), dtype=np.float32)        # [n, m], float32
   >>> spectra = 20 * rng.random((30, 1000), dtype=np.float32)  # [n, b], float32
   >>> keep = mass_screen(response, spectra, threshold=2.24, fill_value=0.25)
   >>> print(f"{np.mean(keep == 1) * 100:.1f} % of the (pixel, component) cells are kept")
   14.4 % of the (pixel, component) cells are kept

The rule
--------

Component ``m`` is kept for pixel ``b`` when

.. code-block:: text

   min_n(spectra[n, b] / response[n, m]) * max_n(response[n, m]) >= threshold

after replacing non-positive samples with ``fill_value``. Scaled to fit the
spectrum at every spectral pixel, the component must still produce at least
``threshold`` at its peak. Both settings must be finite and use the units of the spectra. The fill value must be non-negative.
A component with no positive response is always masked.

C-contiguous float32 inputs avoid a copy. Other threads can run during the
call, so keep both arrays unchanged until it returns.

.. _stage-worker-counts:

Choosing a worker count
-----------------------

Use :func:`~mass_screen_rs.set_workers` to share a Rayon thread pool across calls.
Pass a positive ``workers`` count to :func:`~mass_screen_rs.mass_screen` to override it for one call.

>>> from mass_screen_rs import set_workers
>>> with set_workers(4):
...     keep = mass_screen(response, spectra, threshold=2.24, fill_value=0.25)
...     single_worker_keep = mass_screen(response, spectra, threshold=2.24, fill_value=0.25, workers=1)
>>> np.testing.assert_array_equal(keep, single_worker_keep)

Leaving the block restores the previous setting. With no count or active context,
a call uses Rayon's default pool. To size that pool, set ``RAYON_NUM_THREADS``
before starting Python.

Each package manages its own pool. Budget threads across concurrent calls. If a
host application accepts worker counts from users, enforce its thread budget
before passing the count to the library.

For multiprocessing, use ``spawn`` or ``forkserver`` and set the worker count in
each child. After the parent uses workers, a forked child raises ``RuntimeError``
when it tries to use them.
