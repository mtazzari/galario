.. http://www.sphinx-doc.org/en/stable/domains.html#the-c-domain

.. default-domain:: c

.. default-role:: code

.. |galario| replace:: **galario**


====================
Using galario from C
====================

The core of galario is accessible from C, all functions are in the header `galario.h`. Of course it can be used from most other languages as well, in particular C++.

A small example
---------------

Here is a small test program that performs the FFT in 2D on an image with random values

.. literalinclude:: ../src/galario_test.c
                    :language: c

After successfully installing galario to `/path/to/galario`, a simple
test program running galario on the CPU with `openMP` and double
precision can be built with::

  gcc -I/path/to/galario/include -L/path/to/galario/lib -lgalario -DDOUBLE_PRECISION galario_test.c -o galario_test

To use single precision, simply do not define the preprocessor symbol `-DDOUBLE_PRECISION` and link in the appropriate library with `-lgalario_single`. A mismatch between the library and the preprocessor symbol causes undefined behavior but usually a segmentation fault causes the program to abort at runtime.

If galario was installed with `cuda` support, link in `-lgalario_cuda` or `-lgalario_single_cuda`.

Example walk-through
--------------------

A small walk through the example's `main` function line by line::

  galario_init();
  ...
  galario_cleanup();

Before any computation is done inside galario, the library has to be initialized. Similarly, any data created during initialization should be cleaned at the end of `main`.

To create the input image, define an array::

   dreal realdata[nx*ny];

The data type `dreal` can refer to either `float` or `double`, depending on the preprocessor symbol `DOUBLE_PRECISION`. galario assumes the input is a real image but the output of the FFT is complex. galario provides a helper function to allocate an array of the proper size and to copy over the input image::

  dcomplex* res = galario_copy_input(nx, ny, realdata);

The actual FFT is done in-place, and the result is stored in `res`. The data layout is described in the `FFTW manual <http://fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data>`_::

  galario_fft2d(nx, ny, res);

To deallocate `res`, we use::

  galario_free(res);

In general, any array created by galario and handed back to the user must be deallocated using `galario_free`.

===============
C API reference
===============

Sample and chi2 functions
-------------------------

These are four main functions that should serve the standard use of galario.

.. function::
   void galario_sample_profile(int nr, dreal* const ints,
   dreal Rmin, dreal dR, dreal dxy, int nxy, dreal dist, dreal inc,
   dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal *u,
   const dreal *v, dcomplex *fint);


   Compute visibilities for a model defined by a radial profile.

   Most parameters are described in detail in the python function :py:func:`sampleProfile`. Here we focus on the parameters that are not present in the python function.

   :param nr: The number of points of the radial profile.
   :param ints: The intensities of the radial profile. Size: `nr`
   :param Rmin: Inner edge of radial grid.
   :param dR: Radial-grid cell size.
   :param dxy: Image cell size.
   :param nxy: Number of image pixels in x- and y-direction.
   :param dist: Distance to source.
   :param inc: Inclination.
   :param dRa: Rectascension offset.
   :param dDec: Dec. offset.
   :param duv: The size of a pixel, uniform in u- and v-direction. Units: same as `u, v`.
   :param PA: Position angle.
   :param nd: Number of visibility points.
   :param u: u coordinate of visibility points.
   :param v: v coordinate of visibility points.
   :param fint: Interpolated visibilities at the points `u, v`.


.. function::
   void galario_sample_image(int nx, int ny, const dreal* image,
   dreal dRA, dreal dDec, dreal duv, dreal PA,
   int nd, const dreal* u, const dreal* v, dcomplex* fint);

   Compute visibilities for a model defined by a real image.

   :param nx: The number of points of `image` in x-direction
   :param ny: The number of points of `image` in y-direction
   :param image: A rectangular image of size `nx*ny`.

   For the other parameters, see :func:`galario_sample_profile() <.galario_sample_profile>`.


.. function::
   void galario_chi2_profile(int nr, dreal* const ints,
   dreal Rmin, dreal dR, dreal dxy, int nxy, dreal dist, dreal inc,
   dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal *u,
   const dreal *v, const dreal* fobs_re, const dreal* fobs_im,
   const dreal* weights, dreal* chi2);

   Compute the chi2 between observed visibilities and the
   interpolation based on a radial profile and return in `chi2`.

   For details, see :py:func:`chi2Profile`.

   All parameters are identical to :func:`galario_sample_profile`
   except that the model visibilities are not returned but directly
   compared to weighted observations.

   :param fobs_re: The real part of the visibilities. Size: `nd`.
   :param fobs_im: The imaginary part of the visibilities. Size: `nd`.
   :param weights: The weights of the observations. Size: `nd`.

.. function::
   void galario_chi2_image(int nx, int ny, const dreal* image,
   dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u,
   const dreal* v, const dreal* fobs_re, const dreal* fobs_im,
   const dreal* weights, dreal* chi2);

   Compute the chi2 between observed visibilities and the
   interpolation based on the input `image` and return in `chi2`.

   For details, see :py:func:`chi2Image`, :func:`galario_sample_image`
   and :func:`galario_chi2_profile`.

Management functions
--------------------

.. function::
   void galario_init();

   Initialize memory and environment for galario.

   Call this function before any computation is performed.

.. function::
   void galario_cleanup();

   Free memory and clean up environment created by
   :func:`galario_init`. Call after all computations in galario.

.. function::
   int galario_threads(int num = 0);

   Set the number of `openMP` threads that galario uses in parallel
   regions to `num`. The default of 0 doesn't change the number of
   threads. Return the current number of threads.

   By default, use the settings in the `openMP` runtime that can be
   affected for example by setting the `OMP_NUM_THREADS` variable.

   For the cuda version, this sets the number of threads per block in cuda kernels.

   For details, see the python function :py:func:`threads`.

.. function::
   int galario_ngpus();

   Get the number of available GPUs.

.. function::
    void galario_use_gpu(int device_id);

    Set the GPU to be used for the computations.

    For details, see the python function :py:func:`use_gpu`.

Individual operations
---------------------

The following functions provide low-level access to individual operations performed by the `sample` and `chi2` functions. A standard user will likely have little use for them. Refer to the python API documentation of the wrappers for details on the individual functions.

.. function::
   void galario_sweep(int nr, dreal* ints, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal dist, dreal inc, dcomplex* image);

.. function::
   void galario_uv_rotate(dreal PA, dreal dRA, dreal dDec, dreal* dRArot, dreal* dDecrot, int nd, const dreal* u, const dreal* v, dreal* urot, dreal* vrot);

.. function::
   dcomplex* galario_copy_input(int nx, int ny, const dreal* image);

.. function::
   void galario_free(void*);

.. function::
   void galario_fft2d(int nx, int ny, dcomplex* image);

.. function::
   void galario_fftshift(int nx, int ny, dcomplex* image);

.. function::
   void galario_fftshift_axis0(int nx, int ny, dcomplex* matrix);

.. function::
    void galario_interpolate(int nrow, int ncol, const dcomplex *image, int nd, const dreal *u, const dreal *v,
    const dreal duv, dcomplex *fint);

.. function::
   void galario_apply_phase_sampled(dreal dRA, dreal dDec, int nd, const dreal* u, const dreal* v, dcomplex* fint);

.. function::
   void galario_reduce_chi2(int nd, const dreal* fobs_re, const dreal* fobs_im, const dcomplex* fint, const dreal* weights, dreal* chi2);
