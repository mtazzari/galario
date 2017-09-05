.. galario documentation master file, created by
   sphinx-quickstart on Wed May 24 13:32:58 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. default-role:: code

.. |galario| replace:: **galario**

=======
galario
=======

**GPU Accelerated Library for Analysing Radio Interferometry Observations**
---------------------------------------------------------------------------

|galario| exploits the computing power of modern graphic cards (GPUs) to accelerate the comparison of model
predictions to the observations of radio interferometers. Namely, it speeds up the computation of the synthetic visibilities
given a model image (or an axisymmetric brightness profile) and their comparison to the observations.

It is licensed under LGPLv3 and along with the GPU accelerated version based on the
`CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_ offers a CPU counterpart accelerated with
`openMP <http://www.openmp.org>`_.
Modern radio interferometers like
`ALMA <http://www.almaobservatory.org/en/home/>`_,
`VLA <https://science.nrao.edu/facilities/vla>`_,
`NOEMA <http://www.iram-institute.org/EN/noema-project.php?ContentID=9&rub=9&srub=0&ssrub=0&sssrub=0>`_
and the future ones like
`SKA <http://skatelescope.org>`_ are pushing the computational efforts needed to model the observations to the extreme.
The unprecedented sensitivity and resolution achieved by these observatories require comparing model predictions
with huge amount of data points that sample a wide range of spatial frequencies.
In this context, |galario| provides a fast library useful for comparing a model to observations directly in the Fourier plane.

We presented |galario| in `Tazzari, Beaujean and Testi (2017) <LINK>`_, where you can find more details about the
relevant equations and the algorithm implementation.
Here we do not aim to summarize the vast literature about Radio Interferometry, but we refer the interested reader to
`this <http://aspbooks.org/a/volumes/table_of_contents/180>`_ thorough reference.

|galario| is actively developed on `GitHub <https://github.com/mtazzari/galario/>`_.

.. and has been employed in :doc:`these published studies <studies>`.

Instructions on how to build and install |galario| can be found :doc:`here <install>`.


Basic Usage
-----------
.. |u_j| replace:: :math:`u_j`
.. |v_j| replace:: :math:`v_j`
.. |w_j| replace:: :math:`w_j`

Let's say you have an observational dataset of `N` visibility points located at :math:`(u_j, v_j)`, with :math:`j=1...N` and |u_j|, |v_j|
expressed in units of the observing wavelength. :math:`V_{obs\ j}` is the :math:`j`-th complex visibility with associated theoretical weight |w_j|.
If you want to compute the visibilities of a model `image` in the same :math:`(u_j, v_j)` locations of the observations,
you can easily do it with the GPU accelerated |galario|:

.. code-block:: python

    from galario import pc, au
    from galario.double_cuda import sampleImage

    dist = 240. * pc  # distance to the source [cm]
    dxy = 10. * au    # spatial size of the pixel in the model image [cm]

    vis = sampleImage(image, dxy, dist, u, v)

where `vis` is a complex array of length :math:`N` containing the real (`vis.real`) and imaginary (`vis.imag`) part of the synthetic visibilities.

If you are doing a **fit** and the only number you are interested in is the **chi square** needed for the likelihood computation,
you can use directly:

.. code-block:: python

    from galario.double_cuda import chi2Image

    chi2 = chi2Image(image, dxy, dist, u, v, V_obs.real, V_obs.image, w)

If you want to compare the observations with a model characterized by an **axisymmetric brightness profile**, |galario| offers
dedicated functions that exploit the symmetry of the model to accelerate the image creation.
If :math:`I(R)` is the radial brightness profile, the command is as simple as:

.. code-block:: python

    from galario.double_cuda import sampleProfile

    vis = sampleProfile(I, Rmin, dR, nxy, dxy, dist, u, v)
.. add an example with inc, PA, dRA, dDec?

where `Rmin` and `dR` are the innermost radius and the cell size of the grid on which :math:`I(R)` is computed. An analogous function
`chi2Profile` allows one to compute directly the chi square.

.. note::
    If you work on a machine **without** a CUDA-enabled GPU, don't worry: you can use the CPU version
    of |galario| by just removing the subscript `"_cuda"` from the imports above and benefit from the openMP parallelization.
    All the function names and interfaces are the same for GPU and CPU version!

More details on how to get started with |galario| are given in the :doc:`quickstart <quickstart>`.

.. Be sure to checkout also the :doc:`cookbook <cookbook>` with many worked examples!

License and Attribution
-----------------------
If you use |galario| for your research, please cite `Tazzari, Beaujean and Testi (2017) <LINK>`_ .
The BibTeX entry for the paper is::

    @ARTICLE{...}

|galario| is free software licensed under the LGPLv3 License. For more details see the :doc:`LICENSE <license>`.

© Copyright 2017 Marco Tazzari, Frederik Beaujean, Leonardo Testi.

Contents
--------
.. toctree::
    :maxdepth: 2

    install
    quickstart
    py-api
    C-api
    license
..    cookbook
..    studies

Indices
-------
* :ref:`genindex`
* :ref:`search`

