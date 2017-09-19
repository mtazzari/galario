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
To compute the synthetic visibilities of a model use the :func:`sampleImage() <galario.double.sampleImage>` and
:func:`sampleProfile() <galario.double.sampleProfile>` functions.

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
To compute the :math:`\chi^2` of a model use the :func:`chi2Image() <galario.double.chi2Image>` and
:func:`chi2Profile() <galario.double.chi2Profile>` functions.

.. autofunction:: galario.double.chi2Image
.. autofunction:: galario.double.chi2Profile

GPU related
-----------
.. py:data:: galario.HAVE_CUDA

    Global variable (bool).
    It is `True` if the GPU libraries (`galario.double_cuda` and `galario.single_cuda`) are available, `False` otherwise.
    On a machine without a CUDA-enabled GPU it is always `False`.

.. autofunction:: galario.double.ngpus
.. autofunction:: galario.double.use_gpu

Other useful functions
----------------------
.. autofunction:: galario.double.threads
.. autofunction:: galario.double.get_image_size
.. autofunction:: galario.double.check_image_size
.. autofunction:: galario.double.sweep
.. autofunction:: galario.double.apply_phase_vis

