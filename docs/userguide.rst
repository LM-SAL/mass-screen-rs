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

Threads
-------

The kernel runs on all cores through `rayon <https://github.com/rayon-rs/rayon>`_;
``RAYON_NUM_THREADS`` caps the thread count and must be set before the first call.
The call releases the GIL, so other threads keep running; leave ``response`` and ``spectra``
unchanged until it returns, as writing to an array that another thread is reading gives
inconsistent results, as with NumPy.
