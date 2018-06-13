===========
Basic Usage
===========

.. |u_j| replace:: :math:`u_j`
.. |v_j| replace:: :math:`v_j`
.. |w_j| replace:: :math:`w_j`

Let's say you have an observational dataset of `M` visibility points located at :math:`(u_j, v_j)`, with :math:`j=1...M` and |u_j|, |v_j| expressed in units of the observing wavelength. :math:`V_{obs\ j}` (Jy) is the :math:`j`-th complex visibility with associated theoretical weight |w_j|.
With |galario| you can:

**1) Compute visibilities from a model image**

    If you want to compute the visibilities of a model :code:`image` (Jy/px) with pixel size `dxy` (rad) in the same :math:`(u_j, v_j)` locations of the observations, you can easily do it with the GPU accelerated |galario|:

    .. code-block:: python

        from galario.double_cuda import sampleImage

        vis = sampleImage(image, dxy, u, v)

    where `vis` is a complex array of length :math:`N` containing the real (`vis.real`) and imaginary (`vis.imag`) part of the synthetic visibilities.

**2) Compute visibilities from an axisymmetric brightness profile**

    If you want to compare the observations with a model characterized by an **axisymmetric brightness profile**, |galario| offers dedicated functions that exploit the symmetry of the model to accelerate the image creation.

    If :math:`I(R)` (Jy/sr) is the radial brightness profile, the command is as simple as:

    .. code-block:: python

        from galario.double_cuda import sampleProfile

        vis = sampleProfile(I, Rmin, dR, nxy, dxy, u, v)

    where `Rmin` and `dR` are expressed in radians and are the innermost radius and the cell size of the grid on which :math:`I(R)` is computed. An analogous function
    `chi2Profile` allows one to compute directly the chi square.

**3) Compute the** :math:`\chi^2` **of a model (image or brightness profile)**

    If you are doing a **fit** and the only number you are interested in is the :math:`\chi^2` for the likelihood computation, you can use directly one of these:

    .. code-block:: python

        from galario.double_cuda import chi2Image

        chi2 = chi2Image(image, dxy, u, v, V_obs.real, V_obs.imag, w)
        chi2 = chi2Profile(I, Rmin, dR, nxy, dxy, u, v, V_obs.real, V_obs.imag, w)


**4) Do all the above operations + translate and rotate the model image**

    To translate the model image in Right Ascension and Declination direction by (dRA, dDec) offsets (rad),
    or to rotate the image by a Position Angle PA (rad) (defined East of North), you can specify them as optional parameters.

    This works for all the `sampleImage`, `sampleProfile`, `chi2Image` and `chi2Profile` functions:

    .. code-block:: python

        from galario.double_cuda import sampleImage

        vis = sampleImage(image, dxy, u, v, dRA=dRA, dDec=dDec, PA=PA)

.. note::
    If you work on a machine **without** a CUDA-enabled GPU, don't worry: you can use the CPU version
    of |galario| by just removing the subscript `"_cuda"` from the imports above and benefit from the openMP parallelization.
    All the function names and interfaces are the same for GPU and CPU version!
