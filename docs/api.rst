API Reference
=============

Resamplers
----------

.. autosummary::
   :toctree: generated

   healpix_resample.NearestResampler
   healpix_resample.BilinearResampler
   healpix_resample.BicubicResampler
   healpix_resample.PSFResampler
   healpix_resample.CellPointResampler
   healpix_resample.GroupByResampler
   healpix_resample.ConservativeResampler

Large-scale processing
----------------------

:func:`~healpix_resample.subset_for_parent_cell` restricts a resampler's
input samples *and* output cells to one coarse HEALPix parent cell, so a
global dataset can be processed one parent cell at a time without ever
constructing a resampler over the full input. See the
:doc:`user-guide/regrid_to_healpix_parent_cell_subsetting` page for the
workflow.

.. autosummary::
   :toctree: generated

   healpix_resample.subset_for_parent_cell

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
:func:`~healpix_resample.base.estimate_pixel_area` is used by :class:`~healpix_resample.PSFResampler` to auto-estimate per-sample pixel area when ``area`` is not supplied.

.. autosummary::
   :toctree: generated

   healpix_resample.knn.healpix_weighted_nearest
   healpix_resample.psf.conjugate_gradient
   healpix_resample.psf.least_squares_cg
   healpix_resample.base.estimate_pixel_area