###############################################################################
# This file is part of GALARIO:                                               #
# Gpu Accelerated Library for Analysing Radio Interferometer Observations     #
#                                                                             #
# Copyright (C) 2017-2020, Marco Tazzari, Frederik Beaujean, Leonardo Testi.  #
#                                                                             #
# This program is free software: you can redistribute it and/or modify        #
# it under the terms of the Lesser GNU General Public License as published by #
# the Free Software Foundation, either version 3 of the License, or           #
# (at your option) any later version.                                         #
#                                                                             #
# This program is distributed in the hope that it will be useful,             #
# but WITHOUT ANY WARRANTY; without even the implied warranty of              #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                        #
#                                                                             #
# For more details see the LICENSE file.                                      #
# For documentation see https://mtazzari.github.io/galario/                   #
###############################################################################

cimport numpy as np
from cpython cimport PyObject, Py_INCREF

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
import time
import numpy as np
np.import_array()

from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Voronoi, ConvexHull

include "galario_config.pxi"

cimport galario_defs as cpp

__all__ = ['arcsec', 'deg', 'cgs_to_Jy', 'pc', 'au',
           '_init', '_cleanup', 'set_v_origin',
           'ngpus', 'use_gpu', 'threads',
           'check_obs', 'check_image_size', 'get_image_size',
           'sampleImage', 'sampleUnstructuredImage', 'sampleProfile', 'chi2Image', 'chi2Profile',
           'get_coords_meshgrid',
           'sweep', 'uv_rotate', 'interpolate', 'apply_phase_vis', 'reduce_chi2',
           '_fft2d', '_fftshift', '_fftshift_axis0']


# CONSTANTS
arcsec = 4.84813681109536e-06       # radians
deg = 0.017453292519943295          # radians
cgs_to_Jy = 1e23                    # 1 Jy = 1.0e-23 erg/(s cm^2 Hz)
pc = 3.0856775815e18                # cm (IAU 2015 Resolution B2)
au = 1.49597870700e13               # cm (IAU 2012 Resolution B1)


cdef class ArrayWrapper:
    """Wrap an array allocated in C that has to be deleted by `free`.

    See https://gist.github.com/GaelVaroquaux/1249305#file-cython_wrapper-pyx for a discussion
    """
    cdef void* data_ptr
    cdef int nx, ny

    cdef set_data(self, int nx, int ny, void* data_ptr):
        """ Set the data of the array
        This cannot be done in the constructor as it must receive C-level
        arguments.

        Parameters:
        -----------
        nx: int
            Number of image rows
        data_ptr: void*
            Pointer to the data
        """
        self.data_ptr = data_ptr
        self.nx = nx
        self.ny = ny

    cdef as_ndarray(self, int nx, int ny, void* data_ptr):
        """Create an `ndarray` that doesn't own the memory, we do."""
        cdef np.npy_intp shape[2]
        cdef np.ndarray ndarray

        self.set_data(nx, ny, data_ptr)

        shape[:] = (self.nx, int(self.ny/2)+1)

        # Create a 2D array, of length `nx*ny/2+1`
        ndarray = np.PyArray_SimpleNewFromData(2, shape, complex_typenum, self.data_ptr)
        ndarray.base = <PyObject*> self

        # without this, data would be cleaned up right away
        Py_INCREF(self)
        return ndarray

    def __dealloc__(self):
        """ Frees the array. This is called by Python when all the
        references to the object are gone. """
        cpp.galario_free(self.data_ptr)


def _init():
    """ Initializes FFTW threads """
    cpp.init()


def _cleanup():
    """ Cleans up FFTW threads """
    cpp.cleanup()


def set_v_origin(origin):
    """ Sets the Dec (v) axis orientation given the matrix origin """
    if origin == 'upper':
        # origin [0, 0] in the upper left pixel
        return 1.
    elif origin == 'lower':
        # origin [0, 0] in the lower left pixel
        return -1.
    else:
        raise AssertionError("Expect origin='upper' or 'lower', got {}".format(origin))


# ############################################################################ #
#                                                                              #
#                            GPU HELPER FUNCTIONS                              #
#                                                                              #
# ############################################################################ #

def ngpus():
    """
    Return how many GPUs are available on the machine.

    Returns
    -------
    ngpus : int
        Number of GPUs available on the machine.

    """
    return cpp.ngpus()


def use_gpu(int device_id):
    """
    Select the GPU to be used for the computation.

    Typical call signature::

        use_gpu(device_id)

    Parameters
    ----------
    device_id : int
        ID of the GPU to be used for the computation.

    Notes
    -----
    If more than one GPU is present, `device_id` might not coincide with the `ID`
    reported by the `nvidia-smi` command, which reflects the PCI order.
    If the system administrator does not provide instructions on how to set the `device_id`,
    we recommend to start from `device_id=0` and simultaneously check which GPU is used with
    `watch -n0.1 nvidia-smi`.

    """
    cpp.use_gpu(device_id)

def threads(int num=0):
    """
    Set and get the number of threads to be used in parallel sections of the code.

    To set, pass `num>0`. To get the current setting, call without any argument.

    Typical call signatures::

        default_nthreads = threads() # in the first call, a default is preset
        threads(num=16) # now change the default

    Parameters
    ----------
    num : int, optional

        On the *GPU*, `num` is the square root of the number of threads per block to be used.
        1D kernels are launched with linear blocks of size `num*num`.
        2D Kernels are launched with square blocks of size `num*num`.

        On the *CPU*, this sets the number of openMP threads. The default is
        `omp_get_max_threads()` which can be set through the `OMP_NUM_THREADS`
        environment variable. If compiled without openMP support, `num` is
        ignored and this function always returns 1.

    Notes
    -----
    The CUDA documentation suggests starting with `num*num`>=64 and multiples of 32,
    e.g. 128, 256. GPU cards with compute capability between 2 and 6.2 have
    maximum number of threads per block of 1024, which is achieved for `num=32`.

    Check the maximum number of threads per block of your GPU by running
    the `deviceQuery` command.

    On the CPU, it may useful to experiment with more threads than available
    cores to see if hyperthreading provides any benefit.

    """
    return cpp.threads(num)


# ############################################################################ #
#                                                                              #
#                                    CHECKS                                    #
#                                                                              #
# ############################################################################ #

def check_obs(vis_obs_re, vis_obs_im, vis_obs_w, vis=None, u=None, v=None):
    """ Checks whether the observed visibilities are consistent. """
    nd = len(vis_obs_re)
    assert len(vis_obs_im) == nd, "Wrong array length: vis_obs_im."
    assert len(vis_obs_w) == nd, "Wrong array length: vis_obs_w."
    if vis:
        assert len(vis) == nd, "Wrong array length: vis."
    if u:
        assert len(u) == nd, "Wrong array length: u"
    if v:
        assert len(v) == nd, "Wrong array length: v"

    return True


def check_image_size(u, v, nxy, dxy, duv, PB=0, verbose=False):
    """
    Check whether the setup of the (u, v) plane satisfies Nyquist criteria for (u, v) plane sampling.

    Typical call signature::

        check_image_size(u, v, nxy, dxy, duv, PB=0, verbose=False)

    Parameters
    ----------
    u : array_like, float
        u coordinate of the visibility points where the FT has to be sampled.
        **units**: wavelength
    v : array_like, float
        v coordinate of the visibility points where the FT has to be sampled.
        The length of v must be equal to the length of u.
        **units**: wavelength
    nxy : int
        Size of the image along x and y direction.
        **units**: pixel
    dxy : float
        Size of the image cell, assumed equal in both x and y direction.
        **units**: rad
    duv : float
        Size of the cell in the (u, v) plane, assumed uniform and equal on both u and v directions.
        **units**: wavelength
    verbose : bool, optional
        If True, prints information on the criteria to be fulfilled by `nxy` and `dxy`.

    """
    assert len(u) == len(v), "Wrong array length: u, v must have same length."

    uvdist = np.hypot(u, v)

    MRS = 0.6 / np.min(uvdist)
    max_uv = np.max(uvdist) * 2.
    # the factor of 2 comes from the fact that the FFT sample frequencies from -0.5 to 0.5 times max_uv

    FOV = nxy*dxy
    FOV_to_MRS = FOV/MRS
    UVFOV_to_MAXUV = 1/(max_uv*dxy)

    FOV_to_MRS_str = "Nxy * dxy / MRS = {} must be > 1 at the very least".format(FOV_to_MRS)
    UVFOV_to_MAXUV_str = "Nxy * duv / (2*max(u,v)) = {} must be > 2 for Nyquist sampling".format(UVFOV_to_MAXUV)

    if PB != 0:
        FOV_to_PB = FOV/PB
        FOV_to_PB_str = "Nxy * dxy / PB = {} must be > 1".format(FOV_to_PB)

    if verbose:
        print(FOV_to_MRS_str)
        print(UVFOV_to_MAXUV_str)
        if PB != 0:
            print(FOV_to_PB_str)

    assert FOV_to_MRS > 1, FOV_to_MRS_str
    assert UVFOV_to_MAXUV > 2, UVFOV_to_MAXUV_str

    if PB != 0:
        assert FOV_to_PB > 1, FOV_to_PB_str

    # to avoid segfaults in the interpolation, ensure that indices are ok
    assert np.max(np.abs(u) / duv <= nxy//2 + 1)
    assert np.max(np.abs(v) / duv <= nxy//2)

    return True


def get_image_size(u, v, PB=0, f_min=5., f_max=2.5, verbose=False):
    """
    Compute the recommended image size given the (u, v) locations.

    Typical call signature::

        nxy, dxy = get_image_size(u, v, PB=0, f_min=5., f_max=2.5, verbose=False)

    Parameters
    ----------
    u : array_like, float
        u coordinate of the visibility points where the FT has to be sampled.
        **units**: wavelength
    v : array_like, float
        v coordinate of the visibility points where the FT has to be sampled.
        The length of v must be equal to the length of u.
        **units**: wavelength
    PB : float, optional
        Primary beam of the antenna, e.g. 1.22*wavelength/Diameter for an idealized
        antenna with uniform illumination.
        **units**: rad
    f_min : float
        Size of the field of view covered by the (u, v) plane grid w.r.t. the field
        of view covered by the image. Recommended to be larger than 3 for better results.
        **units**: pure number
    f_max : float
        Nyquist rate: numerical factor that ensures the Nyquist criterion is satisfied when sampling
        the synthetic visibilities at the specified (u, v) locations. Must be larger than 2.
        The maximum (u, v)-distance covered is `f_max` times the maximum (u, v)-distance
        of the observed visibilities.
        **units**: pure number
    verbose : bool, optional
        If True, prints information on the criteria to be fulfilled by `nxy` and `dxy`.

    Returns
    -------
    nxy : int
        Size of the image along x and y direction.
        **units**: pixel
    dxy : float
        Returned only if not provided in input.
        Size of the image cell, assumed equal and uniform in both x and y direction.
        **units**: cm

    """
    uvdist = np.hypot(u, v)

    MRS = 0.6 / np.min(uvdist)
    duv = 1 / MRS / f_min
    max_uv = np.max(uvdist) * 2. * f_max

    # nxy to ensure that FOV > MRS * f_min
    nxy = int(2 ** np.ceil(np.log2(max_uv/duv)))
    nxy_MRS = nxy

    dxy = 1/(nxy*duv)

    if PB != 0:
        # impose that the field of view (FOV) is larger than the PB.
        while dxy*nxy/PB < 1. :
            nxy *= 2  # multiply by 2 to keep nxy a power of 2

    if verbose:
        print("dxy:{:e}arcsec\tnxy_MRS:{}".format(dxy/arcsec, nxy_MRS))
        print("nxy_MRS: matrix size to have FOV > f_min * MRS, where f_min:{} and MRS:{:e}arcsec".format(f_min, MRS/arcsec))

        if PB != 0:
            print("nxy_FOV:{}".format(nxy))
            print("nxy_FOV: matrix size to have FOV > PB")

    return nxy, dxy



# ############################################################################ #
#                                                                              #
#                               SCIENTIFIC APIs                                #
#                                                                              #
# ############################################################################ #

def sampleImage(dreal[:,::1] image, dxy, dreal[::1] u, dreal[::1] v,
                dRA=0., dDec=0., PA=0., check=False, origin='upper'):
    """
    Compute the synthetic visibilities of a model image at the specified (u, v) locations.

    The 2D surface brightness in `image` is Fourier transformed and sampled in the
    (u, v) locations given in the `u` and `v` arrays.

    Typical call signature::

        vis = sampleImage(image, dxy, u, v, dRA=0, dDec=0, PA=0, check=False, origin='upper')

    Parameters
    ----------
    image : 2D array_like, float
        Square matrix of shape (nxy, nxy) containing the 2D surface brightness of the model.
        Assume the x-axis (R.A.) increases from right (West) to left (East)
        and the y-axis (Dec.) increases from bottom (South) to top (North).
        `nxy` must be even.
        **units**: Jy/pixel
    dxy : float
        Size of the image cell, assumed equal in both x and y direction.
        **units**: rad
    u : array_like, float
        u coordinate of the visibility points where the FT has to be sampled.
        **units**: wavelength
    v : array_like, float
        v coordinate of the visibility points where the FT has to be sampled.
        The length of v must be equal to the length of u.
        **units**: wavelength
    dRA : float, optional
        R.A. offset w.r.t. the phase center by which the image is translated.
        If dRA > 0 translate the image towards the left (East). Default is 0.
        **units**: rad
    dDec : float, optional
        Dec. offset w.r.t. the phase center by which the image is translated.
        If dDec > 0 translate the image towards the top (North). Default is 0.
        **units**: rad
    PA : float, optional
        Position Angle, defined East of North. Default is 0.
        **units**: rad
    check : bool, optional
        If True, check whether `image` and `dxy` satisfy Nyquist criterion for
        computing the synthetic visibilities in the (u, v) locations provided.
        Additionally check that the (u, v) points fall in the image to avoid
        segmentation violations. Default is False since the check might take
        time. For executions where speed is important, set to False.
    origin : ['upper' | 'lower'], optional
        Set the [0,0] pixel index of the matrix in the upper left or lower left corner of the axes.
        It follows the same convention as in matplotlib `matshow` and `imshow` commands.
        Declination axis and the matrix y axis are parallel for `origin='lower'`, anti-parallel for `origin='upper'`.
        The central pixel corresponding to the (RA, Dec) = (0, 0) is always [Nxy/2, Nxy/2].
        For more details see the Technical Requirements page in the online docs.

    Returns
    -------
    vis : array_like, complex
        Synthetic visibilities sampled in the (u, v) locations given in `u` and `v`.
        **units**: Jy

    """
    nxy = image.shape[0]

    duv = 1 / (dxy*nxy)

    if check:
        check_image_size(u, v, nxy, dxy, duv)

    vis = np.zeros(len(u), dtype=complex_dtype)
    v_origin = set_v_origin(origin)
    cpp._sample_image(nxy, nxy, <void*>&image[0,0], v_origin, dRA, dDec, duv, PA, len(u), <void*>&u[0], <void*>&v[0], <void*>np.PyArray_DATA(vis))

    return vis


def sampleUnstructuredImage(dreal[::1] x, dreal[::1] y, dreal[::1] image, 
                int nxy, dreal dxy, dreal[::1] u, dreal[::1] v,
                dRA=0., dDec=0., PA=0., check=False, origin='upper'):
    """
    Compute the synthetic visibilities of a model image at the specified (u, v) locations.

    The 2D surface brightness in `image` is Fourier transformed and sampled in the
    (u, v) locations given in the `u` and `v` arrays.

    Typical call signature::

        vis = sampleImage(image, dxy, u, v, dRA=0, dDec=0, PA=0, check=False, origin='upper')

    Parameters
    ----------
    x : 1D array_like, float
        List of x coordinates at which intensities are known.
        **units**: rad
    y : 1D array_like, float
        List of y coordinates at which intensities are known.
        **units**: rad
    image : 1D array_like, float
        Array containing the surface brightness of the model.
        Assume the x-axis (R.A.) increases from right (West) to left (East)
        and the y-axis (Dec.) increases from bottom (South) to top (North).
        `nxy` must be even.
        **units**: Jy/st
    nxy : int
        Number of pixels to use for the interpolated gridded image.
    dxy : float
        Size of the image cell in the interpolated image, assumed equal in both x and y direction.
        **units**: rad
    u : array_like, float
        u coordinate of the visibility points where the FT has to be sampled.
        **units**: wavelength
    v : array_like, float
        v coordinate of the visibility points where the FT has to be sampled.
        The length of v must be equal to the length of u.
        **units**: wavelength
    dRA : float, optional
        R.A. offset w.r.t. the phase center by which the image is translated.
        If dRA > 0 translate the image towards the left (East). Default is 0.
        **units**: rad
    dDec : float, optional
        Dec. offset w.r.t. the phase center by which the image is translated.
        If dDec > 0 translate the image towards the top (North). Default is 0.
        **units**: rad
    PA : float, optional
        Position Angle, defined East of North. Default is 0.
        **units**: rad
    check : bool, optional
        If True, check whether `image` and `dxy` satisfy Nyquist criterion for
        computing the synthetic visibilities in the (u, v) locations provided.
        Additionally check that the (u, v) points fall in the image to avoid
        segmentation violations. Default is False since the check might take
        time. For executions where speed is important, set to False.
    origin : ['upper' | 'lower'], optional
        Set the [0,0] pixel index of the matrix in the upper left or lower left corner of the axes.
        It follows the same convention as in matplotlib `matshow` and `imshow` commands.
        Declination axis and the matrix y axis are parallel for `origin='lower'`, anti-parallel for `origin='upper'`.
        The central pixel corresponding to the (RA, Dec) = (0, 0) is always [Nxy/2, Nxy/2].
        For more details see the Technical Requirements page in the online docs.

    Returns
    -------
    vis : array_like, complex
        Synthetic visibilities sampled in the (u, v) locations given in `u` and `v`.
        **units**: Jy

    """

    # Now pick back up with what is typically done for regular grids.
    duv = 1 / (dxy*nxy)

    if check:
        check_image_size(u, v, nxy, dxy, duv)

    vis = np.zeros(len(u), dtype=complex_dtype)
    v_origin = set_v_origin(origin)
    cpp._sample_unstructured_image(<void*>&x[0], <void*>&y[0], nxy, nxy, dxy, len(x), <void*>&image[0], v_origin, dRA, dDec, duv, PA, len(u), <void*>&u[0], <void*>&v[0], <void*>np.PyArray_DATA(vis))

    return vis





def sampleProfile(dreal[::1] intensity, Rmin, dR, nxy, dxy, dreal[::1] u, dreal[::1] v,
                  dRA=0., dDec=0., PA=0., inc=0., check=False):
    """
    Compute the synthetic visibilities of a model with an axisymmetric brightness profile.

    The brightness profile `intensity` is used to build a 2D image of the model, which is
    then Fourier transformed and sampled in the (u, v) locations given in the `u` and `v` arrays.

    The image is created as in :func:`.sweep` assuming that the x-axis (R.A.) increases
    from right (West) to left (East) and the y-axis (Dec.) increases from bottom (South) to top (North).

    Typical call signature::

        vis = sampleProfile(intensity, Rmin, dR, nxy, dxy, u, v,
                            dRA=0, dDec=0, PA=0, inc=0, check=False)

    Parameters
    ----------
    intensity : (M,) array_like, float
        Array containing the radial brightness profile of the model.
        The profile is assumed to be sampled on a linear radial grid starting
        at `Rmin` with spacing `dR`.
        **units**: Jy/sr
    Rmin : float
        Inner edge of the radial grid, i.e. the radius where the brightness is intensity[0].
        **units**: rad
    dR : float
        Size of the cell of the radial grid, assumed linear.
        **units**: rad
    nxy : int
        Side of the square model image, which is internally computed.
        **units**: pixel
    dxy : float
        Size of the image cell, assumed equal in both x and y direction.
        **units**: rad
    u : array_like, float
        u coordinate of the visibility points where the FT has to be sampled.
        **units**: wavelength
    v : array_like, float
        v coordinate of the visibility points where the FT has to be sampled.
        The length of v must be equal to the length of u.
        **units**: wavelength
    dRA : float, optional
        R.A. offset w.r.t. the phase center by which the image is translated.
        If dRA > 0 translate the image towards the left (East). Default is 0.
        **units**: rad
    dDec : float, optional
        Dec. offset w.r.t. the phase center by which the image is translated.
        If dDec > 0 translate the image towards the top (North). Default is 0.
        **units**: rad
    PA : float, optional
        Position Angle, defined East of North. Default is 0.
        **units**: rad
    inc : float, optional
        Inclination of the image plane along a North-South (top-bottom) axis.
        If inc=0. the image is face-on; if inc=90. the image is edge-on.
        **units**: rad
    check : bool, optional
        If True, check whether `image` and `dxy` satisfy Nyquist criterion for
        computing the synthetic visibilities in the (u, v) locations provided.
        Additionally check that the (u, v) points fall in the image to avoid
        segmentation violations. Default is False since the check might take
        time. For executions where speed is important, set to False.

    Returns
    -------
    vis : array_like, complex
        Synthetic visibilities sampled in the (u, v) locations given in `u` and `v`.
        **units**: Jy

    See also
    --------
    :func:`.sweep`

    """
    assert Rmin < dxy, "For the interpolation of the image center, expect Rmin < dxy, but got Rmin={}, dxy={}".format(Rmin, dxy)
    duv = 1 / (dxy*nxy)

    if check:
        check_image_size(u, v, nxy, dxy, duv)

    vis = np.zeros(len(u), dtype=complex_dtype)
    cpp._sample_profile(len(intensity), <void*>&intensity[0], Rmin, dR, dxy, nxy, inc, dRA, dDec, duv, PA, len(u), <void*>&u[0], <void*>&v[0], <void*>np.PyArray_DATA(vis))

    return vis


def chi2Image(dreal[:,::1] image, dxy, dreal[::1] u, dreal[::1] v,
              dreal[::1] vis_obs_re, dreal[::1] vis_obs_im, dreal[::1] vis_obs_w,
              dRA=0., dDec=0., PA=0., check=False, origin='upper'):
    """
    Compute the chi square of a model image given the observed visibilities.

    The chi square is computed from the observed and synthetic visibilities as:

    .. math::

        \chi^2 = \sum_{j=1}^N w_j * [(Re V_{obs\ j}-Re V_{mod\ j})^2 + (Im V_{obs\ j}-Im V_{mod\ j})^2]

    where :math:`V_{mod}` are the synthetic visibilities, which are computed internally
    as in :func:`.sampleImage`.

    Typical call signature::

        chi2 = chi2Image(image, dxy, u, v, vis_obs_re, vis_obs_im, vis_obs_w,
                         dRA=0, dDec=0, PA=0, check=False, origin='upper')

    Parameters
    ----------
    image : 2D array_like, float
        Square matrix of shape (nxy, nxy) containing the 2D surface brightness of the model.
        Assume the x-axis (R.A.) increases from right (West) to left (East)
        and the y-axis (Dec.) increases from bottom (South) to top (North).
        `nxy` must be even.
        **units**: Jy/pixel
    dxy : float
        Size of the image cell, assumed equal in both x and y direction.
        **units**: rad
    u : array_like, float
        u coordinate of the visibility points where the FT has to be sampled.
        **units**: wavelength
    v : array_like, float
        v coordinate of the visibility points where the FT has to be sampled.
        The length of `v` must be equal to the length of `u`.
        **units**: wavelength
    vis_obs_re : array_like, float
        Real part of the observed visibilities.
        **units**: Jy
    vis_obs_im: array_like, float
        Imaginary part of the observed visibilities.
        The length of `vis_obs_im` must be equal to the length of `vis_obs_re`.
        **units**: Jy
    vis_obs_w: array_like, float
        Weight associated to the observed visibilities.
        The length of `vis_obs_w` must be equal to the length of `vis_obs_re`.
        **units**:
    dRA : float, optional
        R.A. offset w.r.t. the phase center by which the image is translated.
        If dRA > 0 translate the image towards the left (East). Default is 0.
        **units**: rad
    dDec : float, optional
        Dec. offset w.r.t. the phase center by which the image is translated.
        If dDec > 0 translate the image towards the top (North). Default is 0.
        **units**: rad
    PA : float, optional
        Position Angle, defined East of North. Default is 0.
        **units**: rad
    check : bool, optional
        If True, check whether `image` and `dxy` satisfy Nyquist criterion for
        computing the synthetic visibilities in the (u, v) locations provided.
        Additionally check that the (u, v) points fall in the image to avoid
        segmentation violations. Default is False since the check might take
        time. For executions where speed is important, set to False.
    origin : ['upper' | 'lower'], optional
        Set the [0,0] pixel index of the matrix in the upper left or lower left corner of the axes.
        It follows the same convention as in matplotlib `matshow` and `imshow` commands.
        Declination axis and the matrix y axis are parallel for `origin='lower'`, anti-parallel for `origin='upper'`.
        The central pixel corresponding to the (RA, Dec) = (0, 0) is always [Nxy/2, Nxy/2].
        For more details see the Technical Requirements page in the online docs.

    Returns
    -------
    chi2: float
        The chi square, not normalized.

    See also
    --------
    :func:`.sampleImage`

    """
    check_obs(vis_obs_re, vis_obs_im, vis_obs_w, u=u, v=v)
    nxy = image.shape[0]

    duv = 1 / (dxy*nxy)

    if check:
        check_image_size(u, v, nxy, dxy, duv)

    v_origin = set_v_origin(origin)

    return cpp._chi2_image(image.shape[0], image.shape[1], <void*>&image[0,0], v_origin, dRA, dDec, duv, PA, len(u), <void*> &u[0],  <void*> &v[0],  <void*>&vis_obs_re[0], <void*>&vis_obs_im[0], <void*>&vis_obs_w[0])


def chi2UnstructuredImage(dreal[::1] x, dreal[::1] y, dreal[::1] image, 
              int nxy, dreal dxy, dreal[::1] u, dreal[::1] v,
              dreal[::1] vis_obs_re, dreal[::1] vis_obs_im, dreal[::1] vis_obs_w,
              dRA=0., dDec=0., PA=0., check=False, origin='upper'):
    """
    Compute the chi square of a model unstructured image given the observed visibilities.

    The chi square is computed from the observed and synthetic visibilities as:

    .. math::

        \chi^2 = \sum_{j=1}^N w_j * [(Re V_{obs\ j}-Re V_{mod\ j})^2 + (Im V_{obs\ j}-Im V_{mod\ j})^2]

    where :math:`V_{mod}` are the synthetic visibilities, which are computed internally
    as in :func:`.sampleUnstructuredImage`.

    Typical call signature::

        chi2 = chi2UnstructuredImage(x, y, image, nxy, dxy, u, v, vis_obs_re, vis_obs_im, vis_obs_w,
                         dRA=0, dDec=0, PA=0, check=False, origin='upper')

    Parameters
    ----------
    x : 1D array_like, float
        List of x coordinates at which intensities are known.
        **units**: rad
    y : 1D array_like, float
        List of y coordinates at which intensities are known.
        **units**: rad
    image : 1D array_like, float
        Array containing the surface brightness of the model.
        Assume the x-axis (R.A.) increases from right (West) to left (East)
        and the y-axis (Dec.) increases from bottom (South) to top (North).
        `nxy` must be even.
        **units**: Jy/st
    nxy : int
        Number of pixels to use for the interpolated gridded image.
    dxy : float
        Size of the image cell in the interpolated image, assumed equal in both x and y direction.
        **units**: rad
    u : array_like, float
        u coordinate of the visibility points where the FT has to be sampled.
        **units**: wavelength
    v : array_like, float
        v coordinate of the visibility points where the FT has to be sampled.
        The length of `v` must be equal to the length of `u`.
        **units**: wavelength
    vis_obs_re : array_like, float
        Real part of the observed visibilities.
        **units**: Jy
    vis_obs_im: array_like, float
        Imaginary part of the observed visibilities.
        The length of `vis_obs_im` must be equal to the length of `vis_obs_re`.
        **units**: Jy
    vis_obs_w: array_like, float
        Weight associated to the observed visibilities.
        The length of `vis_obs_w` must be equal to the length of `vis_obs_re`.
        **units**:
    dRA : float, optional
        R.A. offset w.r.t. the phase center by which the image is translated.
        If dRA > 0 translate the image towards the left (East). Default is 0.
        **units**: rad
    dDec : float, optional
        Dec. offset w.r.t. the phase center by which the image is translated.
        If dDec > 0 translate the image towards the top (North). Default is 0.
        **units**: rad
    PA : float, optional
        Position Angle, defined East of North. Default is 0.
        **units**: rad
    check : bool, optional
        If True, check whether `image` and `dxy` satisfy Nyquist criterion for
        computing the synthetic visibilities in the (u, v) locations provided.
        Additionally check that the (u, v) points fall in the image to avoid
        segmentation violations. Default is False since the check might take
        time. For executions where speed is important, set to False.
    origin : ['upper' | 'lower'], optional
        Set the [0,0] pixel index of the matrix in the upper left or lower left corner of the axes.
        It follows the same convention as in matplotlib `matshow` and `imshow` commands.
        Declination axis and the matrix y axis are parallel for `origin='lower'`, anti-parallel for `origin='upper'`.
        The central pixel corresponding to the (RA, Dec) = (0, 0) is always [Nxy/2, Nxy/2].
        For more details see the Technical Requirements page in the online docs.

    Returns
    -------
    chi2: float
        The chi square, not normalized.

    See also
    --------
    :func:`.sampleImage`

    """
    check_obs(vis_obs_re, vis_obs_im, vis_obs_w, u=u, v=v)

    duv = 1 / (dxy*nxy)

    if check:
        check_image_size(u, v, nxy, dxy, duv)

    v_origin = set_v_origin(origin)

    return cpp._chi2_unstructured_image(<void*>&x[0], <void*>&y[0], nxy, nxy, dxy, len(x), <void*>&image[0], v_origin, dRA, dDec, duv, PA, len(u), <void*>&u[0], <void*>&v[0], <void*>&vis_obs_re[0], <void*>&vis_obs_im[0], <void*>&vis_obs_w[0])


def chi2Profile(dreal[::1] intensity, Rmin, dR, nxy, dxy, dreal[::1] u, dreal[::1] v,
                dreal[::1] vis_obs_re, dreal[::1] vis_obs_im, dreal[::1] vis_obs_w,
                dRA=0., dDec=0., PA=0., inc=0., check=False):
    """
    Compute the chi square of a model with an axisymmetric brightness profile
    given the observed visibilities.

    The image is created from the intensity profile as in :func:`.sweep`.
    The chi square is computed from the observed and synthetic visibilities as:

    .. math::

        \chi^2 = \sum_{j=1}^N w_j * [(Re V_{obs\ j}-Re V_{mod\ j})^2 + (Im V_{obs\ j}-Im V_{mod\ j})^2]

    where :math:`V_{mod}` are the synthetic visibilities, which are computed internally
    as in :func:`.sampleProfile`.

    Typical call signature::

        chi2 = chi2Profile(intensity, Rmin, dR, nxy, dxy, u, v, vis_obs_re, vis_obs_im, vis_obs_w,
                           dRA=0, dDec=0, PA=0, inc=0, check=False)

    Parameters
    ----------
    intensity : array_like, float
        Array containing the radial brightness profile of the model.
        The profile is assumed to be sampled on a linear radial grid starting
        at `Rmin` with spacing `dR`.
        **units**: Jy/sr
    Rmin : float
        Inner edge of the radial grid, i.e. the radius where the brightness is `intensity[0]`.
        **units**: rad
    dR : float
        Size of the cell of the radial grid, assumed linear.
        **units**: rad
    nxy : int
        Side of the square model image, which is internally computed.
        **units**: pixel
    dxy : float
        Size of the image cell, assumed equal in both x and y direction.
        **units**: rad
    u : array_like, float
        u coordinate of the visibility points where the FT has to be sampled.
        **units**: wavelength
    v : array_like, float
        v coordinate of the visibility points where the FT has to be sampled.
        The length of `v` must be equal to the length of `u`.
        **units**: wavelength
    vis_obs_re : array_like, float
        Real part of the observed visibilities.
        **units**: Jy
    vis_obs_im: array_like, float
        Imaginary part of the observed visibilities.
        The length of `vis_obs_im` must be equal to the length of `vis_obs_re`.
        **units**: Jy
    vis_obs_w: array_like, float
        Weight associated to the observed visibilities.
        The length of `vis_obs_w` must be equal to the length of `vis_obs_re`.
        **units**:
    dRA : float, optional
        R.A. offset w.r.t. the phase center by which the image is translated.
        If dRA > 0 translate the image towards the left (East). Default is 0.
        **units**: rad
    dDec : float, optional
        Dec. offset w.r.t. the phase center by which the image is translated.
        If dDec > 0 translate the image towards the top (North). Default is 0.
        **units**: rad
    PA : float, optional
        Position Angle, defined East of North. Default is 0.
        **units**: rad
    inc : float, optional
        Inclination of the image plane along a North-South (top-bottom) axis.
        If inc=0. the image is face-on; if inc=pi/2 the image is edge-on.
        **units**: rad
    check : bool, optional
        If True, check whether `image` and `dxy` satisfy Nyquist criterion for
        computing the synthetic visibilities in the (u, v) locations provided.
        Additionally check that the (u, v) points fall in the image to avoid
        segmentation violations. Default is False since the check might take
        time. For executions where speed is important, set to False.

    Returns
    -------
    chi2: float
        The chi square, not normalized.

    See also
    --------
    :func:`.sampleProfile`, :func:`.sweep`

    """
    check_obs(vis_obs_re, vis_obs_im, vis_obs_w, u=u, v=v)
    assert Rmin < dxy, "For the interpolation of the image center, expect Rmin < dxy, but got Rmin={}, dxy={}".format(Rmin, dxy)

    duv = 1 / (dxy*nxy)

    if check:
        check_image_size(u, v, nxy, dxy, duv)

    return cpp._chi2_profile(len(intensity), <void*> &intensity[0], Rmin, dR, dxy, nxy, inc, dRA, dDec, duv, PA, len(u), <void*> &u[0],  <void*> &v[0],  <void*>&vis_obs_re[0], <void*>&vis_obs_im[0], <void*>&vis_obs_w[0])


def get_coords_meshgrid(nrow, ncol, dxy=1., inc=0., Dx=0., Dy=0., origin='upper'):
    """
    Compute the (R.A, Dec.) coordinate mesh grid to create the image.
    (x, y) axes are the (R.A, Dec.) axes: x increases leftwards, y increases upwards.
    All coordinates are computed in linear pixels units. To convert to angular units,
    just multiply the output by the angular pixel size.
    To put (0, 0) coordinates offset by (Dx, Dy) w.r.t. the image center, specify Dx, Dy.

    Typical call signature::

            x, y, x_m, y_m, R_m = get_coords_meshgrid(nrow, ncol, dxy=dxy, inc=inc, Dx=Dx, Dy=Dy, origin='lower')

    Parameters
    ----------
    nrow : int
        Number of rows of the image.
    ncol : int
        Number of columns of the image.
    dxy : float, optional
        Size of the image cell, assumed equal in both x and y direction.
        By default is 1, thus implying the output arrays are expressed in number of pixels.
        **units**: rad
    inc : float, optional
        Inclination along the North-South axis, default is zero.
        **units**: rad
    Dx : float, optional
        Offset of the source along the x-axis (R.A.), default is zero.
        If positive, moves the origin to the East (left).
        **units**: rad
    Dy :  float, optional
        Offset of the source along the y-axis (Dec.), default is zero.
        If positive, moves the origin to the North (top).
        **units**: rad
    origin : ['upper' | 'lower'], optional
        Set the [0,0] pixel index of the matrix in the upper left or lower left corner of the axes.
        It follows the same convention as in matplotlib `matshow` and `imshow` commands.
        Declination axis and the matrix y axis are parallel for `origin='lower'`, anti-parallel for `origin='upper'`.
        The central pixel corresponding to the (RA, Dec) = (0, 0) is always [Nxy/2, Nxy/2].
        For more details see the Technical Requirements page in the online docs.

    Returns
    -------
    x, y: array_like, float
        Pixel coordinates along the (R.A., Dec.) directions.
        **units**: same as dxy. If dxy=1: number of pixels.
    x_m, y_m: array_like, float
        Pixel coordinate meshgrid along the (R.A., Dec.) directions.
        **units**: same as dxy. If dxy=1: number of pixels.
    R_m: array_like, float
        Radial coordinate meshgrid.
        **units**: same as dxy. If dxy=1: number of pixels.

    """
    v_origin = set_v_origin(origin)

    # create the mesh grid
    x = (np.linspace(0.5, -0.5 + 1./float(ncol), ncol, dtype=real_dtype)) * ncol * dxy
    y = (np.linspace(0.5, -0.5 + 1./float(nrow), nrow, dtype=real_dtype)) * nrow * dxy * v_origin

    # shrink the x axis by the inclination, since PA is the angle East of North of the
    # the plane of the disk (orthogonal to the angular momentum axis)
    # PA=0 is a disk with vertical orbital node (aligned along North-South)
    x_m, y_m = np.meshgrid((x - Dx) / np.cos(inc), (y - Dy))

    R_m = np.hypot(x_m, y_m)

    return x, y, x_m, y_m, R_m


def sweep(dreal[::1] intensity, Rmin, dR, nxy, dxy, inc=0.):
    """
    Create a 2D model image from an axisymmetric brightness profile.

    The image is created assuming that the x-axis (R.A.) increases
    from right (West) to left (East) and the y-axis (Dec.) increases
    from bottom (South) to top (North).

    Typical call signature::

        image = sweep(intensity, Rmin, dR, nxy, dxy, inc=0)

    Parameters
    ----------
    intensity : 2D array_like, float
        Array containing the radial brightness profile of the model.
        The brightness profile is assumed to be sampled on a linear radial grid
        starting at `Rmin` and with spacing `dR`.
        **units**: Jy/sr
    Rmin : float
        Inner edge of the radial grid, i.e. the radius where the brightness is intensity[0].
        **units**: rad
    dR : float
        Size of the cell of the radial grid, assumed linear.
        **units**: rad
    nxy : int
        Side of the square model image.
        **units**: pixel
    dxy : float
        Size of the image cell, assumed equal in both x and y direction.
        **units**: rad
    inc : float, optional
        Inclination of the image plane along a North-South (top-bottom) axis.
        If inc=0. the image is face-on; if inc=pi/2 the image is edge-on.
        **units**: rad

    Returns
    -------
    image : (nxy, nxy) array_like, float
        Image of the surface brightness.
        **units**: Jy/pixel

    """
    assert Rmin < dxy, "For the interpolation algorithm, Rmin must be smaller than dxy. " \
                       "Currently Rmin={}\t dxy={}".format(Rmin, dxy)

    image = np.empty((nxy, nxy//2+1), dtype=complex_dtype, order='C')

    cpp._sweep(len(intensity), <void*>&intensity[0], Rmin, dR, nxy, dxy, inc, <void*>np.PyArray_DATA(image))

    # return a C-Continuous array so that it can be used in sampleImage()
    return np.ascontiguousarray(image.view(dtype=real_dtype)[:, :-2])


def uv_rotate(PA, dRA, dDec, dreal[::1] u, dreal[::1] v):
    """
    Obtain the (u, v) locations and the dRA, dDec angular offsets in a frame
    rotated by a position angle PA.

    Typical call signature::

        dRArot, dDecrot, urot, vrot = uv_rotate(PA, dRA, dDec, u, v)

    Parameters
    ----------
    PA : float
        Position Angle, defined East of North.
        **units**: rad
    dRA : float, optional
        R.A. offset.
        **units**: rad
    dDec : float, optional
        Dec. offset.
        **units**: rad
    u : array_like, float
        u coordinate of the visibility points.
        **units**: wavelength
    v : array_like, float
        v coordinate of the visibility points.
        The length of `v` must be equal to the length of `u`.
        **units**: wavelength

    Returns
    -------
    dRArot : float
        Rotated R.A. offset.
    dDecrot : float
        Rotated Dec. offset.
    urot : array_like, float
        Rotated u coordinates of the visibility points.
    vrot : array_like, float
        Rotated v coordinates of the visibility points.

    """
    nd = len(u)
    assert nd == len(v)

    cdef dreal dRArot
    cdef dreal dDecrot
    urot = np.copy(u, order='C')
    vrot = np.copy(v, order='C')

    cpp._uv_rotate(PA, dRA, dDec, &dRArot, &dDecrot, nd,
                       <void*> &u[0], <void*> &v[0],
                       <void*>np.PyArray_DATA(urot), <void*>np.PyArray_DATA(vrot))

    return dRArot, dDecrot, urot, vrot


def interpolate(dcomplex[:,::1] r2cFT, duv, dreal[::1] u, dreal[::1] v, origin='upper'):
    """
    Interpolate the R2C Fourier transform of a model image in (u, v) locations.

    Typical call signature::

        vis = interpolate(r2cFT, duv, u, v, origin='upper')

    Parameters
    ----------
    r2cFT : 2D array_like, float
        Output of the R2C Fourier transform.
    duv : float
        Size of the cell in the (u, v) plane.
        **units**: wavelength
    u : array_like, float
        u coordinate of the visibility points where `r2cFT` has to be sampled.
        **units**: wavelength
    v : array_like, float
        v coordinate of the visibility points where `r2cFT` has to be sampled.
        The length of `v` must be equal to the length of `u`.
        **units**: wavelength
    origin : ['upper' | 'lower'], optional
        Set the [0,0] pixel index of the matrix in the upper left or lower left corner of the axes.
        It follows the same convention as in matplotlib `matshow` and `imshow` commands.
        Declination axis and the matrix y axis are parallel for `origin='lower'`, anti-parallel for `origin='upper'`.
        The central pixel corresponding to the (RA, Dec) = (0, 0) is always [Nxy/2, Nxy/2].
        For more details see the Technical Requirements page in the online docs.

    Returns
    -------
    vis : array_like, complex
        Samples of the image in the given (u, v) locations.
        **units**: Jy

    """
    vis = np.empty(len(u), dtype=complex_dtype, order='C')
    v_origin = set_v_origin(origin)

    cpp._interpolate(r2cFT.shape[0], r2cFT.shape[1], <void*>&r2cFT[0,0], v_origin, len(u), <void*>&u[0], <void*>&v[0], duv, <void*>np.PyArray_DATA(vis))

    return vis


def apply_phase_vis(dRA, dDec, dreal[::1] u, dreal[::1] v, dcomplex[::1] vis):
    """
    Apply phase to sampled visibility points as to translate the image in the real
    space by an offset dRA along Right Ascension (R.A.) and dDec along Declination.
    R.A. increases towards left (East), thus dRA>0 translates the image towards East.

    Typical call signature::

        vis_shifted = apply_phase_vis(dRA, dDec, u, v, vis)

    Parameters
    ----------
    dRA : float
        Right Ascension offset.
        **units**: rad
    dDec : float
        Declination offset.
        **units**: rad
    u : array_like, float
        u-coordinates of visibility points.
        **units**: observing wavelength
    v : array_like, float
        v-coordinates of visibility points.
        **units**: observing wavelength
    vis : array_like, complex
        complex visibilities, of form Real(Vis) + i*Imag(Vis).
        **units**: Jy

    Returns
    -------
    vis_out : array_like, complex
        shifted complex visibilities
        **units**: arbitrary, same as vis

    """
    vis_out = np.copy(vis, order='C')
    cpp._apply_phase_sampled(dRA, dDec, len(vis), <void*> &u[0], <void*> &v[0], <void*>np.PyArray_DATA(vis_out))

    return vis_out


def reduce_chi2(dreal[::1] vis_obs_re, dreal[::1] vis_obs_im, dreal[::1] vis_obs_w, dcomplex[::1] vis):
    """
    Compute the chi square of observed and model visibilities.

    Typical call signature::

        chi2 = reduce_chi2(vis_obs_re, vis_obs_im, vis_obs_w, vis)

    Parameters
    ----------
    vis_obs_re : array_like, float
        Real part of the observed visibilities.
        **units**: Jy
    vis_obs_im: array_like, float
        Imaginary part of the observed visibilities.
        The length of `vis_obs_im` must be equal to the length of `vis_obs_re`.
        **units**: Jy
    vis_obs_w: array_like, float
        Weight associated to the observed visibilities.
        The length of `vis_obs_w` must be equal to the length of `vis_obs_re`.
    vis : array_like, complex
        Complex model visibilities.
        The length of `vis` must be equal to the length of `vis_obs_re`.
        **units**: Jy

    Returns
    -------
    chi2 : float
        The chi square, not normalized.

    """
    check_obs(vis_obs_re, vis_obs_im, vis_obs_w, vis)

    return cpp._reduce_chi2(len(vis), <void*>&vis_obs_re[0], <void*>&vis_obs_im[0], <void*>&vis[0], <void*>&vis_obs_w[0])


def _fft2d(dreal[:,::1] image):
    """ Wrapper for the 2D Real to Complex FFT """
    # require contiguous arrays with stride=1 in buffer[::1]
    nx, ny = image.shape[0], image.shape[1]
    cdef void* res = cpp._copy_input(nx, ny, <void*>&image[0,0])

    cpp._fft2d(nx, ny, res)

    # Use a custom delete function to free the array http://gael-varo1quaux.info/programming/cython-example-of-exposing-c-computed-arrays-in-python-without-image-copies.html
    return ArrayWrapper().as_ndarray(nx, ny, res)


def _fftshift(dreal[:,::1] matrix):
    """
    Swap the four quarters of a matrix, such that the upper-left quarter is swapped
    with the lower-right one and the lower-left one with the upper-right one.

    This equivalent to numpy.fft.fftshift(matrix).

    Parameters
    ----------
    matrix : 2D array_like, float
        A matrix.

    Returns
    -------
    The swapped matrix.

    """
    nx, ny = matrix.shape[0], matrix.shape[1]
    assert nx % 2 == 0 and ny % 2 == 0, "Expect even matrix size but got {}".format(matrix.shape)

    cdef void* res = cpp._copy_input(nx, ny, <void*>&matrix[0,0])

    cpp._fftshift(nx, ny, res)

    return ArrayWrapper().as_ndarray(nx, ny, res)


def _fftshift_axis0(dcomplex[:,::1] matrix):
    """
    Swap the upper and lower halves of a matrix. The swap is done in-place.

    This equivalent to numpy.fft.fftshift(matrix, axes=0).

    Parameters
    ----------
    matrix : 2D array_like, float
        A matrix.

    """
    assert matrix.shape[0] % 2 == 0, "Axis 0 of `matrix` has to be even "
    cpp._fftshift_axis0(matrix.shape[0], matrix.shape[1], <void*>&matrix[0,0])
