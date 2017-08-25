cimport numpy as np
import numpy as np
from cpython cimport PyObject, Py_INCREF

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

include "galario_config.pxi"

__all__ = ['arcsec', 'deg', 'cgs_to_Jy', 'pc', 'au',
           '_init', '_cleanup',
           'ngpus', 'use_gpu', 'threads_per_block', 'threads',
           'check_image', 'check_obs', 'check_uvplane', 'get_image_size', 'get_uvcell_size',
           'sampleImage', 'sampleProfile', 'chi2Image', 'chi2Profile',
           'sweep', 'uv_rotate', 'interpolate', 'apply_phase_vis', 'reduce_chi2',
           '_fft2d', '_fftshift', '_fftshift_axis0']


# CONSTANTS
arcsec = 4.84813681109536e-06       # radians
deg = 0.017453292519943295          # radians
cgs_to_Jy = 1e23                    # 1 Jy = 1.0e-23 erg/(s cm^2 Hz)
pc = 3.0856775815e18                # cm (IAU 2015 Resolution B2)
au = 1.49597870700e13               # cm (IAU 2012 Resolution B1)


IF DOUBLE_PRECISION:
    ctypedef double dreal
    real_dtype = np.float64

    ctypedef double complex dcomplex
    complex_dtype = np.complex128
    complex_typenum = np.NPY_COMPLEX128

ELSE:
    ctypedef float dreal
    real_dtype = np.float32

    ctypedef float complex dcomplex
    complex_dtype = np.complex64
    complex_typenum = np.NPY_COMPLEX64

cdef extern from "galario_py.h":
    # Main user functions
    void _galario_sample_profile(int nr, void* intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal dist, dreal inc, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis)
    void _galario_sample_image(int nx, int ny, void* image, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis)
    void _galario_chi2_profile(int nr, void* intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal dist, dreal inc, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis_obs_re, void* vis_obs_im, void* vis_obs_w, dreal* chi2)
    void _galario_chi2_image(int nx, int ny, void* image, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis_obs_re, void* vis_obs_im, void* vis_obs_w, dreal* chi2)
    void _galario_sweep(int nr, void* intensity, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal dist, dreal inc, void* image)
    void _galario_uv_rotate(dreal PA, dreal dRA, dreal dDec, void* dRArot, void* dDecrot, int nd, void* u, void* v, void* urot, void* vrot)

    # Interface for the experts
    void* _galario_copy_input(int nx, int ny, void* realimage);
    void* _galario_fft2d(int nx, int ny, void* image)
    void _galario_fftshift(int nx, int ny, void* image)
    void _galario_fftshift_axis0(int nx, int ny, void* image);
    void _galario_interpolate(int nx, int ncol, void* image, int nd, void* u, void* v, dreal duv, void* vis)
    void _galario_apply_phase_sampled(dreal dRA, dreal dDec, int nd, void* u, void* v, void* vis)
    void _galario_reduce_chi2(int nd, void* vis_obs_re, void* vis_obs_im, void* vis, void* vis_obs_w, dreal* chi2)

cdef extern from "galario.h":
    int  galario_threads_per_block(int num);
    void galario_init();
    void galario_threads(int num);
    void galario_cleanup();
    void galario_free(void*);
    void galario_use_gpu(int device_id)
    int  galario_ngpus()

cdef extern from "fftw3.h":
    void fftw_free(void*)


cdef class ArrayWrapper:
    """Wrap an array allocated in C that has to be deleted by `galario_free`.

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
        galario_free(self.data_ptr)


def _init():
    """ Initializes FFTW threads """
    galario_init()


def _cleanup():
    """ Cleans up FFTW threads """
    galario_cleanup()


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
    return galario_ngpus()


def use_gpu(int device_id):
    """
    Pin the GPU to be used for the computation.

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
    we recommend to start from `device_id`=0 and simultaneously check which GPU is used with
    `watch -n0.1 nvidia-smi` .

    """
    galario_use_gpu(device_id)

def threads(int num=1):
    galario_threads(num)

def threads_per_block(int num=16):
    """
    Set the number of threads per block on each of the block dimensions to be used.

    Typical call signature::

        threads_per_block(num=16)

    Parameters
    ----------
    num : int, optional
        Number of threads per block on each of the block dimensions, default is 16.
        1D kernels will be launched with `num` threads per block.
        2D kernels will be launched with `num*num` threads per block.

    Notes
    -----
    The CUDA documentation suggests starting with small `num` values, multiples of 2.
    GPU cards with compute capability between 2 and 6.2 have maximum number of
    threads per block of 1024, thus implying that the maximum `num` value is 32.

    Check the maximum number of threads per block of your GPU here by running the `deviceQuery` command.

    """
    return galario_threads_per_block(num)


# ############################################################################ #
#                                                                              #
#                                    CHECKS                                    #
#                                                                              #
# ############################################################################ #

def check_image(image):
    """ Checks whether the image is square has even number of cells on both sides. """
    nx, ny = image.shape
    assert nx == ny, "Expect a square image but got shape {}".format(image.shape)
    assert nx % 2 == 0, "Expect an even size but got shape {}".format(image.shape)

    return True


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


def check_uvplane(u, v, nxy, duv, maxuv_factor, minuv_factor):
    """
    Check whether the setup of the (u, v) plane satisfies Nyquist criteria for (u, v) plane sampling.

    Typical call signature::

        check_uvplane(u, v, nxy, duv, maxuv_factor, minuv_factor)

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
    duv : float
        Size of the cell in the (u, v) plane, assumed uniform and equal on both u and v directions.
        **units**: wavelength
    maxuv_factor : float
        Nyquist rate: numerical factor that ensures the Nyquist criterion is satisfied when sampling
        the synthetic visibilities at the specified (u, v) locations. Must be larger than 2.
        The maximum (u, v)-distance covered is `maxuv_factor` times the maximum (u, v)-distance
        of the observed visibilities.
        **units**: pure number
    minuv_factor : float
        Size of the field of view covered by the (u, v) plane grid w.r.t. the field
        of view covered by the image. Recommended to be larger than 3 for better results.
        **units**: pure number

    """
    assert len(u) == len(v), "Wrong array length: u, v must have same length."
    assert maxuv_factor > 2., "Expected maxuv_factor > 2 to ensure correct Nyquist sampling."
    assert minuv_factor > 3., "Expected minuv_factor > 3 to ensure the image covers the field of view of the data."

    uvdist = np.hypot(u, v)
    min_uv = np.min(uvdist)/minuv_factor
    max_uv = np.max(uvdist) * 2. * maxuv_factor
    # the factor of 2 comes from the fact that the FFT sample frequencies from -0.5 to 0.5 times max_uv

    assert duv <= min_uv, "The image does not cover the full field of view of the observations: try increasing nxy or dxy."
    assert duv*nxy >= max_uv, "The uv plane setup does not fulfil Nyquist sampling: try decreasing nxy or dxy."

    return True


def get_image_size(u, v, dist, dxy=None, maxuv_factor=2.2, minuv_factor=3.1):
    """
    Compute the recommended image size given the (u, v) locations.

    Typical call signature::

        nxy, dxy = get_image_size(u, v, dist, dxy=None, maxuv_factor=2.2, minuv_factor=3.1)

    Parameters
    ----------
    u : array_like, float
        u coordinate of the visibility points where the FT has to be sampled.
        **units**: wavelength
    v : array_like, float
        v coordinate of the visibility points where the FT has to be sampled.
        The length of v must be equal to the length of u.
        **units**: wavelength
    dist : float
        Distance to the source.
        **units**: cm
    dxy : float, optional
        Image cell size, assumed equal and uniform in both x and y direction.
        **units**: cm
    maxuv_factor : float, optional
        See :func:`check_uvplane <.check_uvplane>`.
    minuv_factor : float, optional
        See :func:`check_uvplane <.check_uvplane>`.

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

    min_uv = np.min(uvdist)/minuv_factor
    if not dxy:
        max_uv = np.max(uvdist) * 2. * maxuv_factor
        dxy = dist/max_uv
    else:
        max_uv = dist/dxy

    nxy = int(2 ** np.ceil(np.log2(max_uv/min_uv)))

    duv = get_uvcell_size(nxy, dxy, dist)

    check_uvplane(u, v, nxy, duv, maxuv_factor, minuv_factor)

    if not dxy:
        return nxy, dxy
    else:
        return nxy


def get_uvcell_size(nxy, dxy, dist):
    """
    Computes the cell size in the (u, v) space given the size of the image.

    Assumes that the image is a square matrix of side `nxy` with linear (x, y) coordinate axes.

    Typical call signature::

        duv = get_uvcell_size(nxy, dxy, dist)

    Parameters
    ----------
    dxy : float
        Image cell size, assumed equal and uniform in both x and y direction.
        **units**: cm
    nxy : int
        Size of the image.
        **units**: pixel
    dist : float
        Distance to the object in the image
        **units**: cm

    Returns
    ------
    get_uvcell_size : float
        The (u, v) cell size
        **units**: observing wavelength

    """
    return dist / dxy / nxy


# ############################################################################ #
#                                                                              #
#                               SCIENTIFIC APIs                                #
#                                                                              #
# ############################################################################ #

def sampleImage(dreal[:,::1] image, dxy, dist, dreal[::1] u, dreal[::1] v,
                dRA=0., dDec=0., PA=0., uvcheck=False, minuv_factor=3.1, maxuv_factor=2.2):
    """
    Compute the synthetic visibilities of a model image at the specified (u, v) locations.

    The 2D surface brightness in `image` is Fourier transformed and sampled in the
    (u, v) locations given in the `u` and `v` arrays.

    Typical call signature::

        vis = sampleImage(image, dxy, dist, u, v, dRA=0, dDec=0, PA=0, uvcheck=False)

    Parameters
    ----------
    image : 2D array_like, float
        Square matrix of shape (nxy, nxy) containing the 2D surface brightness of the model.
        Assume the x-axis (R.A.) increases from right (West) to left (East)
        and the y-axis (Dec.) increases from bottom (South) to top (North).
        `nxy` must be even.
        **units**: Jy/pixel
    u : array_like, float
        u coordinate of the visibility points where the FT has to be sampled.
        **units**: wavelength
    v : array_like, float
        v coordinate of the visibility points where the FT has to be sampled.
        The length of v must be equal to the length of u.
        **units**: wavelength
    dxy : float
        Size of the image cell, assumed equal and uniform in both x and y direction.
        **units**: cm
    dist : float
        Distance to the source.
        **units**: cm
    dRA : float, optional
        R.A. offset w.r.t. the phase center by which the image is translated.
        If dRA > 0 translate the image towards the left (East). Default is 0.
        **units**: arcsecond
    dDec : float, optional
        Dec. offset w.r.t. the phase center by which the image is translated.
        If dDec > 0 translate the image towards the top (North). Default is 0.
        **units**: arcsecond
    PA : float, optional
        Position Angle, defined East of North. Default is 0.
        **units**: degree
    uvcheck : bool, optional
        If True, check whether `image` and `dxy` satisfy Nyquist criterion for computing
        the synthetic visibilities in the (u, v) locations provided.
        Default is False since the check might take time. For executions where speed is important, set to False.
    maxuv_factor : float, optional
        See :func:`check_uvplane() <.check_uvplane>`.
        **units**: pure number
    minuv_factor : float, optional
        See :func:`check_uvplane() <.check_uvplane>`.
        **units**: pure number

    Returns
    -------
    vis : array_like, complex
        Synthetic visibilities sampled in the (u, v) locations given in `u` and `v`.
        **units**: Jy

    """
    check_image(image)
    nxy = image.shape[0]

    duv = get_uvcell_size(nxy, dxy, dist)

    if uvcheck:
        check_uvplane(u, v, nxy, duv, maxuv_factor, minuv_factor)

    PA *= deg
    dRA *= arcsec
    dDec *= arcsec

    vis = np.zeros(len(u), dtype=complex_dtype)
    _galario_sample_image(nxy, nxy, <void*>&image[0,0], dRA, dDec, duv, PA, len(u), <void*>&u[0], <void*>&v[0], <void*>np.PyArray_DATA(vis))

    return vis


def sampleProfile(dreal[::1] intensity, Rmin, dR, nxy, dxy, dist, dreal[::1] u, dreal[::1] v,
                  dRA=0., dDec=0., inc=0., PA=0., uvcheck=False, minuv_factor=3.1, maxuv_factor=2.2):
    """
    Compute the synthetic visibilities of a model with an axisymmetric brightness profile.

    The brightness profile `intensity` is used to build a 2D image of the model, which is
    then Fourier transformed and sampled in the (u, v) locations given in the `u` and `v` arrays.

    The image is created as in :func:`sweep() <.sweep>` assuming that the x-axis (R.A.) increases
    from right (West) to left (East) and the y-axis (Dec.) increases from bottom (South) to top (North).

    Typical call signature::

        vis = sampleProfile(intensity, Rmin, dR, nxy, dxy, dist, u, v, dRA=0, dDec=0, inc=0, PA=0, uvcheck=False)

    Parameters
    ----------
    intensity : (M,) array_like, float
        Array containing the radial brightness profile of the model.
        The profile is assumed to be sampled on a linear radial grid starting
        at `Rmin` with spacing `dR`.
        **units**: Jy/sr
    Rmin : float
        Inner edge of the radial grid, i.e. the radius where the brightness is intensity[0].
        **units**: cm
    dR : float
        Size of the cell of the radial grid, assumed linear.
        **units**: cm
    nxy : int
        Side of the square model image, which is internally computed.
        **units**: pixel
    dxy : float
        Size of the image cell, assumed equal and uniform in both x and y direction.
        **units**: cm
    dist : float
        Distance to the source.
        **units**: cm
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
        **units**: arcsecond
    dDec : float, optional
        Dec. offset w.r.t. the phase center by which the image is translated.
        If dDec > 0 translate the image towards the top (North). Default is 0.
        **units**: arcsecond
    inc : float, optional
        Inclination of the image plane along a North-South (top-bottom) axis.
        If inc=0. the image is face-on; if inc=90. the image is edge-on.
        **units**: degree
    PA : float, optional
        Position Angle, defined East of North. Default is 0.
        **units**: degree
    uvcheck : bool, optional
        If True, check whether `image` and `dxy` satisfy Nyquist criterion for computing
        the synthetic visibilities in the (u, v) locations provided.
        Default is False since the check might take time. For executions where speed is important, set to False.
    maxuv_factor : float, optional
        See :func:`check_uvplane() <.check_uvplane>`.
        **units**: pure number
    minuv_factor : float, optional
        See :func:`check_uvplane() <.check_uvplane>`.
        **units**: pure number

    Returns
    -------
    vis : array_like, complex
        Synthetic visibilities sampled in the (u, v) locations given in `u` and `v`.
        **units**: Jy

    See also
    --------
    :func:`sweep() <.sweep>`

    """
    assert Rmin < dxy, "For the interpolation of the image center, expect Rmin < dxy, but got Rmin={}, dxy={}".format(Rmin, dxy)
    duv = get_uvcell_size(nxy, dxy, dist)

    if uvcheck:
        check_uvplane(u, v, nxy, duv, maxuv_factor, minuv_factor)

    PA *= deg
    inc *= deg
    dRA *= arcsec
    dDec *= arcsec

    vis = np.zeros(len(u), dtype=complex_dtype)
    _galario_sample_profile(len(intensity), <void*>&intensity[0], Rmin, dR, dxy, nxy, dist, inc, dRA, dDec, duv, PA, len(u), <void*>&u[0], <void*>&v[0], <void*>np.PyArray_DATA(vis))

    return vis


def chi2Image(dreal[:,::1] image, dxy, dist, dreal[::1] u, dreal[::1] v,
              dreal[::1] vis_obs_re, dreal[::1] vis_obs_im, dreal[::1] vis_obs_w,
              dRA=0., dDec=0., PA=0., uvcheck=False, minuv_factor=3.1, maxuv_factor=2.2):
    """
    Compute the chi square of a model image given the observed visibilities.

    The chi square is computed from the observed and synthetic visibilities as::

        chi2 = sum(w * ((vis_obs_re-vis_re)^2 + (vis_obs_im-vis_im)^2))

    where vis_re, vis_im are the real and imaginary part of the synthetic visibilities
    that are computed internally.
    The synthetic visibilities as in :func:`sampleImage <.sampleImage>`.

    Typical call signature::

        chi2 = chi2Image(image, dxy, dist, u, v, vis_obs_re, vis_obs_im, vis_obs_w, dRA=0, dDec=0, PA=0, uvcheck=False)

    Parameters
    ----------
    image : 2D array_like, float
        Square matrix of shape (nxy, nxy) containing the 2D surface brightness of the model.
        Assume the x-axis (R.A.) increases from right (West) to left (East)
        and the y-axis (Dec.) increases from bottom (South) to top (North).
        `nxy` must be even.
        **units**: Jy/pixel
    dxy : float
        Size of the image cell, assumed equal and uniform in both x and y direction.
        **units**: cm
    dist : float
        Distance to the source.
        **units**: cm
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
        **units**: arcsecond
    dDec : float, optional
        Dec. offset w.r.t. the phase center by which the image is translated.
        If dDec > 0 translate the image towards the top (North). Default is 0.
        **units**: arcsecond
    PA : float, optional
        Position Angle, defined East of North. Default is 0.
        **units**: degree
    uvcheck : bool, optional
        If True, check whether `image` and `dxy` satisfy Nyquist criterion for computing
        the synthetic visibilities in the (u, v) locations provided.
        Default is False since the check might take time. For executions where speed is important, set to False.
    maxuv_factor : float, optional
        See :func:`check_uvplane() <.check_uvplane>`.
        **units**: pure number
    minuv_factor : float, optional
        See :func:`check_uvplane() <.check_uvplane>`.
        **units**: pure number

    Returns
    -------
    chi2: float
        The chi square, not normalized.

    See also
    --------
    :func:`sampleImage() <.sampleImage>`

    """
    check_obs(vis_obs_re, vis_obs_im, vis_obs_w, u=u, v=v)
    check_image(image)
    nxy = image.shape[0]

    duv = get_uvcell_size(nxy, dxy, dist)

    if uvcheck:
        check_uvplane(u, v, nxy, duv, maxuv_factor, minuv_factor)

    cdef dreal chi2
    PA *= deg
    dRA *= arcsec
    dDec *= arcsec

    _galario_chi2_image(image.shape[0], image.shape[1], <void*>&image[0,0], dRA, dDec, duv, PA, len(u), <void*> &u[0],  <void*> &v[0],  <void*>&vis_obs_re[0], <void*>&vis_obs_im[0], <void*>&vis_obs_w[0], &chi2)

    return chi2


def chi2Profile(dreal[::1] intensity, Rmin, dR, nxy, dxy, dist, dreal[::1] u, dreal[::1] v,
                dreal[::1] vis_obs_re, dreal[::1] vis_obs_im, dreal[::1] vis_obs_w,
                dRA=0., dDec=0., inc=0., PA=0., uvcheck=False, minuv_factor=3.1, maxuv_factor=2.2):
    """
    Compute the chi square of a model with an axisymmetric brightness profile
    given the observed visibilities.

    The image is created from the intensity profile as in :func:`sweep <.sweep>`.
    The synthetic visibilities are computed as in :func:`sampleProfile <.sampleProfile>`.
    The chi square is computed from the observed and synthetic visibilities as::

        chi2 = sum(w * ((vis_obs_re-vis_re)^2 + (vis_obs_im-vis_im)^2))

    where vis_re, vis_im are the real and imaginary part of the synthetic visibilities
    that are computed internally.

    Typical call signature::

        chi2 = chi2Profile(intensity, Rmin, dR, nxy, dxy, dist, u, v, vis_obs_re, vis_obs_im, vis_obs_w, dRA=0, dDec=0, inc=0, PA=0, uvcheck=False)

    Parameters
    ----------
    intensity : array_like, float
        Array containing the radial brightness profile of the model.
        The profile is assumed to be sampled on a linear radial grid starting
        at `Rmin` with spacing `dR`.
        **units**: Jy/sr
    Rmin : float
        Inner edge of the radial grid, i.e. the radius where the brightness is `intensity[0]`.
        **units**: cm
    dR : float
        Size of the cell of the radial grid, assumed linear.
        **units**: cm
    nxy : int
        Side of the square model image, which is internally computed.
        **units**: pixel
    dxy : float
        Size of the image cell, assumed equal and uniform in both x and y direction.
        **units**: cm
    dist : float
        Distance to the source.
        **units**: cm
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
        **units**: arcsecond
    dDec : float, optional
        Dec. offset w.r.t. the phase center by which the image is translated.
        If dDec > 0 translate the image towards the top (North). Default is 0.
        **units**: arcsecond
    inc : float, optional
        Inclination of the image plane along a North-South (top-bottom) axis.
        If inc=0. the image is face-on; if inc=90. the image is edge-on.
        **units**: degree
    PA : float, optional
        Position Angle, defined East of North. Default is 0.
        **units**: degree
    uvcheck : bool, optional
        If True, check whether `image` and `dxy` satisfy Nyquist criterion for computing
        the synthetic visibilities in the (u, v) locations provided.
        Default is False since the check might take time. For executions where speed is important, set to False.
    maxuv_factor : float, optional
        See :func:`check_uvplane() <.check_uvplane>`.
        **units**: pure number
    minuv_factor : float, optional
        See :func:`check_uvplane() <.check_uvplane>`.
        **units**: pure number

    Returns
    -------
    chi2: float
        The chi square, not normalized.

    See also
    --------
    :func:`sampleProfile() <.sampleProfile>`, :func:`sweep() <.sweep>`

    """
    check_obs(vis_obs_re, vis_obs_im, vis_obs_w, u=u, v=v)
    assert Rmin < dxy, "For the interpolation of the image center, expect Rmin < dxy, but got Rmin={}, dxy={}".format(Rmin, dxy)

    duv = get_uvcell_size(nxy, dxy, dist)

    if uvcheck:
        check_uvplane(u, v, nxy, duv, maxuv_factor, minuv_factor)

    cdef dreal chi2
    inc *= deg
    PA *= deg
    dRA *= arcsec
    dDec *= arcsec

    _galario_chi2_profile(len(intensity), <void*> &intensity[0], Rmin, dR, dxy, nxy, dist, inc, dRA, dDec, duv, PA, len(u), <void*> &u[0],  <void*> &v[0],  <void*>&vis_obs_re[0], <void*>&vis_obs_im[0], <void*>&vis_obs_w[0], &chi2)

    return chi2


def sweep(dreal[::1] intensity, Rmin, dR, nxy, dxy, dist, inc=0.):
    """
    Create a 2D model image from an axisymmetric brightness profile.

    The image is created assuming that the x-axis (R.A.) increases
    from right (West) to left (East) and the y-axis (Dec.) increases
    from bottom (South) to top (North).

    Typical call signature::

        image = sweep(intensity, Rmin, dR, nxy, dxy, dist, inc=0)

    Parameters
    ----------
    intensity : 2D array_like, float
        Array containing the radial brightness profile of the model.
        The brightness profile is assumed to be sampled on a linear radial grid
        starting at `Rmin` and with spacing `dR`.
        **units**: Jy/sr
    Rmin : float
        Inner edge of the radial grid, i.e. the radius where the brightness is intensity[0].
        **units**: cm
    dR : float
        Size of the cell of the radial grid, assumed linear.
        **units**: cm
    nxy : int
        Side of the square model image.
        **units**: pixel
    dxy : float
        Size of the image cell, assumed equal and uniform in both x and y direction.
        **units**: cm
    dist : float
        Distance to the source.
        **units**: cm
    inc : float, optional
        Inclination of the image plane along a North-South (top-bottom) axis.
        If inc=0. the image is face-on; if inc=90. the image is edge-on.
        **units**: degree

    Returns
    -------
    image : (nxy, nxy) array_like, float
        Image of the surface brightness.
        **units**: Jy/pixel

    """
    assert Rmin < dxy, "For the interpolation algorithm, Rmin must be smaller than dxy. " \
                       "Currently Rmin={}\t dxy={}".format(Rmin, dxy)

    inc *= deg
    image = np.empty((nxy, nxy//2+1), dtype=complex_dtype, order='C')

    _galario_sweep(len(intensity), <void*>&intensity[0], Rmin, dR, nxy, dxy, dist, inc, <void*>np.PyArray_DATA(image))

    # return a copy so is C-Continuous and can be used in sampleImage()
    return image.view(dtype=real_dtype)[:, :-2].copy()


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
        **units**: degree
    dRA : float, optional
        R.A. offset.
        **units**: arcsecond
    dDec : float, optional
        Dec. offset.
        **units**: arcsecond
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

    PA *= deg

    cdef dreal dRArot
    cdef dreal dDecrot
    urot = np.copy(u, order='C')
    vrot = np.copy(v, order='C')

    _galario_uv_rotate(PA, dRA, dDec, &dRArot, &dDecrot, nd,
                       <void*> &u[0], <void*> &v[0],
                       <void*>np.PyArray_DATA(urot), <void*>np.PyArray_DATA(vrot))

    return dRArot, dDecrot, urot, vrot


def interpolate(dcomplex[:,::1] r2cFT, duv, dreal[::1] u, dreal[::1] v):
    """
    Interpolate the R2C Fourier transform of a model image in (u, v) locations.

    Typical call signature::

        vis = interpolate(r2cFT, duv, u, v)

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

    Returns
    -------
    vis : array_like, complex
        Samples of the image in the given (u, v) locations.
        **units**: Jy

    """
    vis = np.empty(len(u), dtype=complex_dtype, order='C')

    _galario_interpolate(r2cFT.shape[0], r2cFT.shape[1], <void*>&r2cFT[0,0], len(u), <void*>&u[0], <void*>&v[0], duv, <void*>np.PyArray_DATA(vis))

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
        **units**: arcseconds
    dDec : float
        Declination offset.
        **units**: arcseconds
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
    dRA *= arcsec
    dDec *= arcsec

    vis_out = np.copy(vis, order='C')
    _galario_apply_phase_sampled(dRA, dDec, len(vis), <void*> &u[0], <void*> &v[0], <void*>np.PyArray_DATA(vis_out))

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

    cdef dreal chi2
    _galario_reduce_chi2(len(vis), <void*>&vis_obs_re[0], <void*>&vis_obs_im[0], <void*>&vis[0], <void*>&vis_obs_w[0], &chi2)

    return chi2



def _fft2d(dreal[:,::1] image):
    """ Wrapper for the 2D Real to Complex FFT """
    # require contiguous arrays with stride=1 in buffer[::1]
    check_image(image)
    nx, ny = image.shape[0], image.shape[1]
    cdef void* res = _galario_copy_input(nx, ny, <void*>&image[0,0])

    _galario_fft2d(nx, ny, res)

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
    check_image(matrix)
    nx, ny = matrix.shape[0], matrix.shape[1]
    assert nx % 2 == 0 and ny % 2 == 0, "Expect even matrix size but got {}".format(matrix.shape)

    cdef void* res = _galario_copy_input(nx, ny, <void*>&matrix[0,0])

    _galario_fftshift(nx, ny, res)

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
    _galario_fftshift_axis0(matrix.shape[0], matrix.shape[1], <void*>&matrix[0,0])
