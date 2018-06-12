.. default-role:: code
.. role:: bash(code)
   :language: bash

.. _cookbook:

==================
|galario| Cookbook
==================

.. _cookbook_GPU_CPU_version:

Using the GPU and CPU version
-----------------------------

The CPU version of |galario| is always compiled, even on a system without a CUDA-enabled GPU. In this case you can import
the double and single precision CPU versions of the library with:

.. code-block:: python

    from galario import double
    from galario import single

If built on a machine with a CUDA-enabled GPU, |galario| is compiled also for the GPU. You can still import
the CPU version as above, and the GPU version as follows:

.. code-block:: python

    from galario import double_cuda
    from galario import single_cuda

To check programmatically whether the GPU version is available, you can read the global variable :data:`galario.HAVE_CUDA`.

The following snippet imports the GPU version of galario if it is available, otherwise it imports the CPU version:

.. code-block:: python

    if galario.HAVE_CUDA:
        from galario import double_cuda as g_double
        from galario import single_cuda as g_single
    else:
        from galario import double as g_double
        from galario import single as g_single

This snippet simplifies the development of portable code. Since the functions in `double`, `double_cuda`, `single` and `single_cuda`
have the same interfaces, by adding this snippet to the imports of the module you can develop and test your code on a machine even
without a GPU, then move to a machine with a GPU and run it on the GPU without any change.


Selecting the GPU
-----------------
|galario| can be used on machines with one or more CUDA-capable GPUs. The number of GPUs available on the machine can be
obtained with the :func:`ngpus() <galario.double.ngpus>` function:

.. code-block:: python

    double_cuda.ngpus()   # or single_cuda.ngpus()

which returns an integer number.

It is possible to tell |galario| to use a particular GPU for the computation the :func:`use_gpu() <galario.double.use_gpu>` function:

.. code-block:: python

    double_cuda.use_gpu(ID)

where `ID` is an integer number representing the GPU ID. By default, |galario| uses the GPU with `ID=0`. This means that on machines
with only one CUDA-capable GPU it is not necessary to call `double_cuda.use_GPU(0)` as this is the default behaviour.

.. note::

    The `ID` to be used in :func:`use_gpu() <galario.double.use_gpu>` might differ from the device ID reported by the `nvidia-smi` command.
    See the documentation of :func:`use_gpu() <galario.double.use_gpu>` for more details.


Parallelization: setting the threads
------------------------------------


On the CPU
~~~~~~~~~~
The CPU version of |galario| uses OpenMP to parallelize its operations by distributing the workload to different threads.

Once you imported the CPU version of |galario| (either double or single precision), you can set the number of threads with
the :func:`threads() <galario.double.threads>` function:

.. code-block:: python

    double.threads(N_OMP)

where `N_OMP` is the number of threads to be used. Calling `double.threads(1)` ensures a serial execution of |galario|.

By default, if `double.threads(N_OMP)` is not called by the user, |galario| does not set the number of OpenMP threads to be used.
This means that, at runtime, OpenMP is free to set the number of threads dynamically. Some OpenMP implementations default to one thread, others to as many threads as there are physical cores.

It is possible to retrieve the number of threads used by galario by calling `double.threads()` without argument.

    .. note::

        Setting `N_OMP` larger than the number of **physical** cores, one can use  the HyperThreading technology,
        which can give anything from a moderate boost to a significant performance penalty. This depends on the image size, the memory latency and bandwidth, and other parameters. Experiment around what works best.

        In most cases, a considerable speedup can be obtained by setting `N_OMP` equal to the number of **physical** cores, which for matrix sizes up to 4k x 4k
        yields an almost linear scaling in our benchmark results.

        **Suggestion**: if |galario| is to be used in a code that already uses **MPI** to parallelize the tasks over multiple processes,
        setting `double.threads(1)` might turn out to give a better overall performance.

On the GPU
~~~~~~~~~~
It is possible to change the number of threads per block used to launch 1D and 2D kernels on the GPU with:

.. code-block:: python

    double_cuda.threads(N)

where `N` is the square root of the number of threads for block to be used. By default, `N` is set to 16, which implies
256 threads per block. Due to the physical structure of the current NVIDIA cards, `N` must be equal to 8, 16 or 32.

This is an advanced feature, for most cases the default value should be sufficient. More details are given in the
documentation of :func:`threads() <galario.double.threads>`.


.. _cookbook_meshgrid:

Computing the meshgrid for the image creation
---------------------------------------------
To obtain an image it is often useful to compute a coordinate meshgrid on which a brightness function can be evaluated.

In this recipe we will adopt definitions as described in the :ref:`image specifications <technical_requirements_image_specs>`.

Conceptually, to compute a brightness image it is necessary to perform a pixel by pixel `for` loop over the `x` (R.A.) and `y` (Dec.) axes
and evaluate the brightness in every pixel. However, `for` loops are particularly slow in Python and a faster solution is offered
by coordinate meshgrids that contain the :math:`[x_i, x_j]` pixel centers that can be passed in input to a brightness function.
Further details about the definition of meshgrid can be found in the documentation of the `numpy.meshgrid` function.

|galario| makes the computation of meshgrids easy with the :func:`get_coords_meshgrid() <galario.double.get_coords_meshgrid>` function that
provides meshgrids given the matrix size, the pixel angular size, and other optional parameters such as
R.A. and Dec. offsets, inclination, and matrix origin, e.g.:

.. code-block:: python

   from galario.double import get_coords_meshgrid, arcsec

   nrow, ncol = nxy, nxy   # number of rows and columns (here for a square matrix)
   dxy = 1e-3*arcsec       # pixel size (rad)
   inc = 30.*deg           # inclination (rad)
   Dx = -1.*arcsec         # R.A. offset (negative, therefore to the West)
   Dy = 0.5*arcsec         # Dec. offset (positive, therefore to the North)

   x, y, x_m, y_m, R_m = get_coords_meshgrid(nrow, ncol, dxy=dxy, inc=inc, Dx=Dx, Dy=Dy, origin='lower')


The returned `x` and `y` arrays contain the R.A., Dec. coordinate axes, `x_m` and `y_m` the :math:`[x_i, x_j]` meshgrid, and `R_m` the radial
meshgrid, which is often the only needed quantity for axisymmetric brightness functions.

The returned arrays are in radians, the same units of `dxy`. To obtain just the pixel mapping set `dxy=1.`.

For an axisymmetric brightness `f(R)`, once the meshgrid is computed, the image and its visibilities can be computed as easily as :

.. code-block:: python

   from galario.double import sampleImage, chi2Image

   image = f(R_m)

   vis = sampleImage(image, ...)  or # chi2 = chi2Image(image, ...)

