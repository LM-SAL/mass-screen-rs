User Guide
==========

Masking a response matrix
-------------------------

:func:`~mass_screen_rs.mass_screen` takes the response matrix ``response`` of shape ``[n, m]``
(spectral pixel × VDEM component) and the spectra ``spectra`` of shape ``[n, b]`` (spectral pixel
× image pixel), and returns a float32 array of shape ``[b, m]`` holding ``1.0`` where a component
is kept for a pixel and ``NaN`` where it is masked.

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

after replacing every ``spectra <= 0`` by ``fill_value``: the component is kept when, scaled so
that it does not exceed the spectrum in any spectral pixel, it can still produce ``threshold`` of
signal where its response peaks. ``threshold`` and ``fill_value`` are in the units of ``spectra``,
and ``fill_value`` must be ``>= 0``.

A component whose response is nowhere positive is always masked. This follows the PyTorch
reference implementation; the tests check the kernel against a mask
recorded from it (see :doc:`design-notes`).

Inputs are converted to C-contiguous float32 arrays, so float64 or non-contiguous arrays work at
the cost of a copy. Passing float32 C-contiguous arrays avoids it.

The call releases the GIL. Other threads can run while it reads the inputs, so keep
``response`` and ``spectra`` unchanged until it returns.

.. _stage-worker-counts:

Stage worker counts
-------------------

Pass a positive integer as ``workers=`` to :func:`~mass_screen_rs.mass_screen` for one call.
For a stage with repeated calls, :func:`~mass_screen_rs.set_workers` lets them reuse one Rayon pool.
An explicit count overrides the stage setting for that call. For example, these calls
use four workers and one worker respectively.

>>> from mass_screen_rs import set_workers
>>> with set_workers(4):
...     keep = mass_screen(response, spectra, threshold=2.24, fill_value=0.25)
...     single_worker_keep = mass_screen(response, spectra, threshold=2.24, fill_value=0.25, workers=1)
>>> np.testing.assert_array_equal(keep, single_worker_keep)

Leaving the block restores the previous setting. Each package manages its own pool.
If stages run concurrently, budget workers across the active pools. For a stage running
in another process, enter the context in that process.

With ``workers=None`` (the default), a call uses the active context or falls back to
Rayon's default pool. Set ``RAYON_NUM_THREADS`` before starting Python to configure
that fallback, for example ``RAYON_NUM_THREADS=4 python pipeline.py``. When unset,
Rayon chooses the default size from the available CPUs.
