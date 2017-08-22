.. http://www.sphinx-doc.org/en/stable/domains.html#the-python-domain


Python API reference
====================

..
   Doesn't work: no output, no error message
         .. automodule:: galario.double
            :members:
            :undoc-members:


Computing synthetic visibilities
--------------------------------
The :func:`sampleImage() <galario.double.sampleImage>` and :func:`sampleProfile() <galario.double.sampleProfile>` functions
allow to compute synthetic visibilities.

.. autofunction:: galario.double.sampleImage
.. autofunction:: galario.double.sampleProfile

.. note::
    The translation by (dRA, dDec) and the rotation by PA of the model image are optimized for speed: they are not obtained
    with extra interpolations of the model image, but rather applied in the Fourier plane.

    The offset is achieved by applying a complex phase to the sampled visibilities.
    To rotation is achieved by internally rotating the (u, v) locations by -PA.

    **Suggestion** We recommend starting with `uvcheck` set to True to ensure the results obtained are correct.
    Once a combination of matrix size and `dxy` for the given data has been found, `uvcheck`
    can be safely set to False.

Computing directly the chi square
---------------------------------
The :func:`chi2Image() <galario.double.chi2Image>` and :func:`chi2Profile() <galario.double.chi2Profile>` functions
allow to compute directly the chi square.

.. autofunction:: galario.double.chi2Image
.. autofunction:: galario.double.chi2Profile

Useful functions
----------------
.. autofunction:: galario.double.check_uvplane
.. autofunction:: galario.double.sweep
