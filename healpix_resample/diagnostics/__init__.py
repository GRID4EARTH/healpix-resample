"""Diagnostic tools that are not part of the resampling API itself.

These modules exist to answer "what does the operator actually deliver"
questions about the package -- e.g. :mod:`kernel_geometry`, which measures
the true spatial response of :class:`~healpix_resample.psf.PSFResampler`'s
kernel once it has been truncated to its ``Npt`` nearest cells on the real
HEALPix lattice, as opposed to the idealized, untruncated Gaussian its scale
parameter would suggest.
"""
