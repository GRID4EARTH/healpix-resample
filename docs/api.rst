API Reference
=============

Resamplers
----------

.. autosummary::
   :toctree: generated

   healpix_resample.NearestResampler
   healpix_resample.BilinearResampler
   healpix_resample.PSFResampler
   healpix_resample.CellPointResampler
   healpix_resample.GroupByResampler
   healpix_resample.ConservativeResampler

Base class
----------

:class:`~healpix_resample.knn.KNeighborsResampler` is the base class inherited by all resamplers above.
Use it directly only if you need to implement a custom weighting scheme via :meth:`~healpix_resample.knn.KNeighborsResampler.comp_matrix`.

.. autosummary::
   :toctree: generated

   healpix_resample.knn.KNeighborsResampler

Output
------

All resamplers return a :class:`~healpix_resample.base.ResampleResults` dataclass.

.. autosummary::
   :toctree: generated

   healpix_resample.base.ResampleResults

Internals
---------

These functions are not part of the public API but are documented here for contributors.
:func:`~healpix_resample.knn.healpix_weighted_nearest` is called internally by all resamplers to build the sparse operators.
:func:`~healpix_resample.psf.conjugate_gradient` and :func:`~healpix_resample.psf.least_squares_cg` are used internally by :class:`~healpix_resample.PSFResampler`.

.. autosummary::
   :toctree: generated

   healpix_resample.knn.healpix_weighted_nearest
   healpix_resample.psf.conjugate_gradient
   healpix_resample.psf.least_squares_cg