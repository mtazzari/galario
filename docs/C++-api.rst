.. http://www.sphinx-doc.org/en/stable/domains.html#the-c-domain

.. default-domain:: c

=================
C++ API reference
=================

The core of |galario| is implemented in C++ with a functional API. All functions
are in `namespace galario`.

Sample and chi2 functions
-------------------------

lhere are four main functions that should serve the standard use of |galario|.

.. function::
   void sample_profile(int nr, const dreal* intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u, const dreal* v, dcomplex* vis_int)

   Compute visibilities for a model defined by a radial profile.

   The parameters are described in more detail in the python function :py:func:`sampleProfile`.

   :param nr: The number of points of the radial profile.
   :param intensity: The intensities of the radial profile. Size: `nr`
   :param Rmin: Inner edge of radial grid.
   :param dR: Radial-grid cell size.
   :param dxy: Image cell size.
   :param nxy: Number of image pixels in x- and y-direction.
   :param inc: Inclination.
   :param dRa: Rectascension offset.
   :param dDec: Dec. offset.
   :param duv: The size of a pixel, uniform in u- and v-direction. Units: same as `u, v`.
   :param PA: Position angle.
   :param nd: Number of visibility points.
   :param u: u coordinate of visibility points. Size: `nd`
   :param v: v coordinate of visibility points. Size: `nd`
   :param vis_int: Interpolated visibilities at the points `u, v`. Size: `nd`


.. function::
   void sample_image(int nx, int ny, const dreal* image, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u, const dreal* v, dcomplex* vis_int)

   Compute visibilities for a model defined by a real image.

   :param nx: The number of points of `image` in x-direction
   :param ny: The number of points of `image` in y-direction
   :param image: A rectangular image of size `nx*ny`.

   For the other parameters, see :func:`sample_profile()`.


.. function::
   dreal chi2_profile(int nr, const dreal* intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u, const dreal* v, const dreal* vis_obs_re, const dreal* vis_obs_im, const dreal* weights)

   Compute the :math:`\chi^2` between observed visibilities and the
   interpolation based on a radial profile and return in `chi2`.

   For details, see the python function :py:func:`chi2Profile`.

   All parameters are identical to :func:`sample_profile`
   except that the model visibilities are not returned but directly
   compared to weighted observations.

   :param vis_obs_re: The real part of the visibilities. Size: `nd`.
   :param vis_obs_im: The imaginary part of the visibilities. Size: `nd`.
   :param weights: The weights of the observations. Size: `nd`.

.. function::
   dreal chi2_image(int nx, int ny, const dreal* image, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u, const dreal* v, const dreal* vis_obs_re, const dreal* vis_obs_im, const dreal* weights)

   Compute the :math:`\chi^2` between observed visibilities and the
   interpolation based on the input `image` and return in `chi2`.

   For details, see :func:`chi2_profile` and the python function :py:func:`chi2Image`.

.. DANGER::

   The values of `duv` and and `u`, `v` have to be consistent; i.e., :math:`\max
   |u| \leq \frac{n}{2} + 1` and :math:`v \leq \frac{n}{2}`, where :math:`n` is
   the number of rows and columns of the real input image. For performance
   reasons, the interpolate function does not check this. Inconsistent values
   may lead to segfaults.

Management functions
--------------------

.. function::
   void init()

   Initialize memory and environment for galario.

   Call this function before any computation is performed.

.. function::
   void cleanup()

   Free memory and clean up environment created by
   :func:`init`. Call after all computations in galario.

.. function::
   int threads(int num = 0)

   Set the number of `openMP` threads that galario uses in parallel
   regions to `num`. The default of 0 doesn't change the number of
   threads. Return the current number of threads.

   By default, use the settings in the `openMP` runtime that can be
   affected for example by setting the `OMP_NUM_THREADS` variable.

   For the cuda version, this sets the number of threads per block in cuda kernels.

   For more details, see the python function :py:func:`threads`.

.. function::
   int ngpus()

   Get the number of available GPUs.

.. function::
    void use_gpu(int device_id)

    Set the GPU to be used for the computations.

    For details, see the python function :py:func:`use_gpu`.

Individual operations
---------------------

The following functions provide low-level access to individual operations performed by the `sample` and `chi2` functions. A standard user will likely have little use for them. Refer to the python API documentation of the wrappers for details on the individual functions.

.. function::
   void sweep(int nr, dreal* intensity, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal inc, dcomplex* image)

.. function::
   void uv_rotate(dreal PA, dreal dRA, dreal dDec, dreal* dRArot, dreal* dDecrot, int nd, const dreal* u, const dreal* v, dreal* urot, dreal* vrot)

.. function::
   dcomplex* copy_input(int nx, int ny, const dreal* image)

.. function::
   void galario_free(void*)

Free the memory buffer allocated and returned by :func:`copy_input`.

.. function::
   void fft2d(int nx, int ny, dcomplex* image)

.. function::
   void fftshift(int nx, int ny, dcomplex* image)

.. function::
   void fftshift_axis0(int nx, int ny, dcomplex* matrix)

.. function::
   void interpolate(int nrow, int ncol, const dcomplex *image, int nd, const dreal *u, const dreal *v, const dreal duv, dcomplex *vis_int)

.. function::
   void apply_phase_sampled(dreal dRA, dreal dDec, int nd, const dreal* u, const dreal* v, dcomplex* vis_int)

.. function::
   void reduce_chi2(int nd, const dreal* vis_obs_re, const dreal* vis_obs_im, const dcomplex* vis_int, const dreal* weights, dreal* chi2)
