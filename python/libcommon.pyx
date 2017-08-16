cimport numpy as np
import numpy as np
from cpython cimport PyObject, Py_INCREF

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

include "galario_config.pxi"

# CONSTANTS
arcsec = 4.84813681109536e-06       # radians
deg = 0.017453292519943295          # radians
CGS_to_Jy = 1e23                    # 1 Jy = 1.0e-23 erg/(s cm^2 Hz)
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
    void _galario_sample_profile(int nr, void* ints, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal dist, dreal inc, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis)
    void _galario_sample_image(int nx, int ny, void* image, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis)
    void _galario_chi2_profile(int nr, void* ints, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal dist, dreal inc, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis_obs_re, void* vis_obs_im, void* vis_obs_w, dreal* chi2)
    void _galario_chi2_image(int nx, int ny, void* image, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis_obs_re, void* vis_obs_im, void* vis_obs_w, dreal* chi2)
    void _galario_sweep(int nr, void* ints, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal dist, dreal inc, void* image)
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



########################
## HELPER FUNCTIONS  ##
########################

def _check_image(image):
    nx, ny = image.shape
    assert nx == ny, "Expect a square image but got shape {}".format(image.shape)
    assert nx % 2 == 0

    return True

def _check_obs(vis_obs_re, vis_obs_im, vis_obs_w, vis=None, u=None, v=None):
    nd = len(vis_obs_re)
    assert len(vis_obs_im) == nd, "Wrong array length: vis_obs_im."
    assert len(vis_obs_w) == nd, "Wrong array length: vis_obs_w."
    if vis:
        assert len(vis) == nd, "Wrong array length: vis."
    if u:
        assert len(u) == nd, "Wrong array length: u"
    if v:
        assert len(v) == nd, "Wrong array length: v"


def _check_uv(u, v):
    assert len(u) == len(v), "Wrong array length: u, v."


# TODO: review this function
def get_image_size(dist, u, v, max_f=4., min_f=3.):
    """ dist: cm;  u and v in lambda. Returns dxy: same units as dist"""
    uvdist = np.hypot(u, v)
    max_uv = np.max(uvdist)*max_f
    min_uv = np.min(uvdist)/min_f

    nxy = int(2**np.ceil(np.log2(max_uv/min_uv)))
    dxy = dist/max_uv

    return nxy, dxy

# TODO: pass dxy and dist instead of duv.
def sampleImage(dreal[:,::1] image, dRA, dDec, duv, dreal[::1] u, dreal[::1] v, PA=0.):
    """
    Computes the synthetic visibilities at the (u, v) locations for a given 2D surface brightness.

    The image of the object surface brightness provided in `image` is Fourier transformed
    and sampled in the (u, v) coordinates defined by the `u` and `v` arrays.

    If provided, the image is offset by dRA (dDec) in Right Ascension (Declination).
    If provided, the image is rotated East of North by an angle PA.

    Typical call signature::

        sample(image, dRA, dDec, duv, u, v)

    Parameters
    ----------
    image : array_like, float
        Matrix of shape (nx, ny) containing the object surface brightness.
        units: Jy/pixel
    dRA : float
        Right Ascension offset by which the image has to be translated.
        units: arcseconds
    dDec : float
        Declination offset by which the image has to be translated.
        units: arcseconds
    duv: float
        uv cell size in the Fourier space, assumed uniform and equal on u and v directions.
        units: observing wavelength
    u: array_like, float
        u coordinate of the visibility points where the FT has to be sampled.
        units: observing wavelength
    v: array_like, float
        u coordinate of the visibility points where the FT has to be sampled.
        units: observing wavelength

    Returns
    -------
    vis: array_like, complex
        Synthetic visibilities.
        units: Jy

    Example
    -------
    Typical usage::

        from galario import sample, uvcell_size

        wle = 0.88e-3  # observing wavelength (m)
        nx = image.shape[0]  # size of the square matrix containing the image.
        dx = 1.49e14  # cm
        dist = 3.1e20  # cm
        duv = uvcell_size(dist, dx, nx)
        vis = sample(image, dRA, dDec, duv, u/wle, v/wle)
        Re_V = vis.real
        Im_V = vis.imag

    """
    PA *= deg
    dRA *= arcsec
    dDec *= arcsec

    _check_image(image)
    vis = np.zeros(len(u), dtype=complex_dtype)
    _galario_sample_image(image.shape[0], image.shape[1], <void*>&image[0,0], dRA, dDec, duv, PA, len(u), <void*>&u[0], <void*>&v[0], <void*>np.PyArray_DATA(vis))

    return vis


def sampleProfile(dreal[::1] ints, Rmin, dR, dist, dRA, dDec, dreal[::1] u, dreal[::1] v, inc=0., dxy=None, nxy=None, duv=None, PA=0.):
    """
    Computes the synthetic visibilities of a brightness profile.

    Typical call signature::

        sampleProfile(f, Rmin, dR, dxy, nxy, dist, dRA, dDec, u, v)

    Parameters
    ----------
    ints : array_like, float
        Array containing the brightness intensity radial profile of the model.
        units: Jy/???.
    Rmin : float
        units: cm
    dR : float
        units: cm
    dxy : float
        units: cm
    nxy : int

    dist : float
        units: cm

    dRA : float
        Right Ascension offset by which the image has to be translated.
        units: arcseconds
    dDec : float
        Declination offset by which the image has to be translated.
        units: arcseconds
    u: array_like, float
        u coordinate of the visibility points where the FT has to be sampled.
        (units: observing wavelength).
    v: array_like, float
        u coordinate of the visibility points where the FT has to be sampled.
        (units: observing wavelength).

    Returns
    -------
    vis: array_like, complex
        Synthetic visibilities.
        units: Jy

    Example
    -------
    Typical usage::

        from galario import sample, uvcell_size

        wle = 0.88e-3  # observing wavelength (m)
        nx = image.shape[0]  # size of the square matrix containing the image.
        dx = 1.49e14  # cm
        dist = 3.1e20  # cm
        duv = uvcell_size(dist, dx, nx)
        vis = sample(image, dRA, dDec, duv, u/wle, v/wle)
        Re_V = vis.real
        Im_V = vis.imag

    """
    # TODO @Marco think better the input parameters, which are optional or not
    # if not dxy and not nxy:
    #     nxy, dxy = get_image_size(dist, u, v)
    # else:
    #     # user must provide both of them
    #     # do checks that dxy, nxy, dist satisfy Nyquist sampling given the u, v data points.
    #     pass

    # if not duv:
    #     # _check_image(image)
    #     duv = uvcell_size(dxy, nxy, dist)
    PA *= deg
    dRA *= arcsec
    dDec *= arcsec

    vis = np.zeros(len(u), dtype=complex_dtype)
    _galario_sample_profile(len(ints), <void*>&ints[0], Rmin, dR, dxy, nxy, dist, inc, dRA, dDec, duv, PA, len(u), <void*>&u[0], <void*>&v[0], <void*>np.PyArray_DATA(vis))

    return vis


def sweep(dreal[::1] ints, Rmin, dR, nxy, dxy, dist, inc=0.):
    """
    Create a 2D surface brightness image from an axisymmetric brightness profile.

    The brightness profile `ints` is assumed to be sampled on a linear radial grid
    starting at `Rmin` with spacing `dR`. The output matrix has shape `(nx, nx)`.
    It is assumed the astronomical convention of x-axis increasing from right (West)
    to left (East) and y-axis increasing from bottom (South) to top (North).

    Parameters
    ----------
    ints : array_like, float
        Brightness profile.
        units: Jy/sr
    Rmin : float
        Inner edge of the radial grid, i.e. the radius where the brightness is ints[0].
        units: cm
    dR : float
        Cell size of the radial grid, assumed linear.
        units: cm
    nxy : int
        Side of the output image.
        units: pixel
    dxy : float
        Cell size of the image.
        units: cm
    dist : float
        Distance to the object.
        units: cm
    inc : float, optional
        Inclination of the image plane along a North-South (top-bottom) axis.
        inc=0. is face-on, inc=90. is edge-on.
        units: degree

    Returns
    -------
    image : ndarray of shape (nx, nx), float
        Image of the surface brightness.
        units: Jy/pixel

    """
    assert Rmin < dxy, "For the interpolation algorithm, Rmin must be smaller than dxy. " \
                       "Currently Rmin={}\t dxy={}".format(Rmin, dxy)
    image = np.empty((nxy, nxy//2+1), dtype=complex_dtype, order='C')
    _galario_sweep(len(ints), <void*>&ints[0], Rmin, dR, nxy, dxy, dist, inc, <void*>np.PyArray_DATA(image))

    # TODO @Fred: image.view(dtype=real_dtype)[:, :-2] is *not* C-Continuous.
    # ensuring the output of sweep is C-contiguous allowes users to use it in sampleImage()
    return  image.view(dtype=real_dtype)[:, :-2]


def apply_phase_sampled(dRA, dDec, dreal[::1] u, dreal[::1] v, dcomplex[::1] vis):
    """
    Apply phase to sampled visibility points as to translate the image in the real
    space by an offset dRA along Right Ascension (R.A.) and dDec along Declination.
    R.A. increases towards left (East), thus dRA>0 translates the image towards East.

    Parameters
    ----------
    dRA : float
        Right Ascension offset.
        units: arcseconds
    dDec : float
        Declination offset.
        units: arcseconds
    u : array_like, float
        u-coordinates of visibility points.
        units: observing wavelength
    v : array_like, float
        v-coordinates of visibility points.
        units: observing wavelength
    vis : array_like, complex
        complex visibilities, of form Real(Vis) + i*Imag(Vis).
        units: Jy

    Returns
    -------
    vis_out : array_like, complex
        shifted complex visibilities
        units: arbitrary, same as vis

    TODO change `vis` name into `vis`

    """
    dRA *= arcsec
    dDec *= arcsec

    vis_out = np.copy(vis, order='C')
    _galario_apply_phase_sampled(dRA, dDec, len(vis), <void*> &u[0], <void*> &v[0], <void*>np.PyArray_DATA(vis_out))

    return vis_out


# require contiguous arrays with stride=1 in buffer[::1]
def fft2d(dreal[:,::1] image):
    _check_image(image)
    nx, ny = image.shape[0], image.shape[1]
    cdef void* res = _galario_copy_input(nx, ny, <void*>&image[0,0])

    _galario_fft2d(nx, ny, res)

    return ArrayWrapper().as_ndarray(nx, ny, res)


def fftshift(dreal[:,::1] image):
    _check_image(image)
    nx, ny = image.shape[0], image.shape[1]
    cdef void* res = _galario_copy_input(nx, ny, <void*>&image[0,0])

    _galario_fftshift(nx, ny, res)
    return ArrayWrapper().as_ndarray(nx, ny, res)


def fftshift_axis0(dcomplex[:,::1] matrix):
    """Swaps the upper and lower halves of a matrix.

    This equivalent to np.fft.fftshift(matrix, axes=0).

    `matrix.shape[0]` has to be even.

    """
    # TODO unify checks for evenness, move into separate function
    assert matrix.shape[0] % 2 == 0, "Axis 0 of `matrix` has to be even "
    _galario_fftshift_axis0(matrix.shape[0], matrix.shape[1], <void*>&matrix[0,0])


def interpolate(dcomplex[:,::1] image, dreal[::1] u, dreal[::1] v, duv):
    vis = np.empty(len(u), dtype=complex_dtype)
    _galario_interpolate(image.shape[0], image.shape[1], <void*>&image[0,0], len(u), <void*>&u[0], <void*>&v[0], duv, <void*>np.PyArray_DATA(vis))
    return vis


def reduce_chi2(dreal[::1] vis_obs_re, dreal[::1] vis_obs_im, dcomplex[::1] vis, dreal[::1] vis_obs_w):
    _check_obs(vis_obs_re, vis_obs_im, vis_obs_w, vis)

    cdef dreal chi2
    _galario_reduce_chi2(len(vis), <void*>&vis_obs_re[0], <void*>&vis_obs_im[0], <void*>&vis[0], <void*>&vis_obs_w[0], &chi2)

    return chi2


def chi2Image(dreal[:,::1] image, dreal dRA, dreal dDec, dreal duv, dreal[::1] u, dreal[::1] v, dreal[::1] vis_obs_re, dreal[::1] vis_obs_im, dreal[::1] vis_obs_w, PA=0.):
    _check_image(image)
    _check_obs(vis_obs_re, vis_obs_im, vis_obs_w, u=u, v=v)

    cdef dreal chi2
    PA *= deg
    dRA *= arcsec
    dDec *= arcsec

    _galario_chi2_image(image.shape[0], image.shape[1], <void*>&image[0,0], dRA, dDec, duv, PA, len(u), <void*> &u[0],  <void*> &v[0],  <void*>&vis_obs_re[0], <void*>&vis_obs_im[0], <void*>&vis_obs_w[0], &chi2)

    return chi2


def chi2Profile(dreal[::1] ints, Rmin, dR, nxy, dxy, dist, inc, dRA, dDec, dreal duv, dreal[::1] u, dreal[::1] v, dreal[::1] vis_obs_re, dreal[::1] vis_obs_im, dreal[::1] vis_obs_w, PA=0.):
    _check_obs(vis_obs_re, vis_obs_im, vis_obs_w, u=u, v=v)

    cdef dreal chi2
    PA *= deg
    dRA *= arcsec
    dDec *= arcsec

    _galario_chi2_profile(len(ints), <void*> &ints[0], Rmin, dR, dxy, nxy, dist, inc, dRA, dDec, duv, PA, len(u), <void*> &u[0],  <void*> &v[0],  <void*>&vis_obs_re[0], <void*>&vis_obs_im[0], <void*>&vis_obs_w[0], &chi2)

    return chi2


def init():
    galario_init()


def cleanup():
    galario_cleanup()


def ngpus():
    return galario_ngpus()


def use_gpu(int device_id):
    galario_use_gpu(device_id)


def threads_per_block(int num=0):
    return galario_threads_per_block(num)


def uv_rotate(PA, dRA, dDec, dreal[::1] u, dreal[::1] v):
    """
    Rotate (u, v) point coordinates and dRA, dDec angular offsets by a position angle PA.
    Applies a cartesian rotation of angle PA to the pair (dRA, dDec) and to the arrays (u, v).

    Typical call signature::

        uv_rotate(PA, dRA, dDec, u, v)

    Parameters
    ----------
    PA : float
        Position Angle, defined East of North.
        units: degree
    dRA : float
        Right Ascension offset.
    dDec : float
        Declination offset.
    u : array_like, float
        u coordinates of the visibility points.
    v : array_like, float
        v coordinates of the visibility points.

    Returns
    -------
    dRArot : float
        Rotated Right Ascension offset.
    dDecrot : float
        Rotated Declination offset.
    urot : array_like, float
        Rotated u coordinates of the visibility points.
    vrot : array_like, float
        Rotated v coordinates of the visibility points.

    Notes
    -----
    The units of the returned values are the same of the input values.

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


# TODO: review this
def uvcell_size(dx, nx, dist):
    """
    Computes the cell size in the Fourier (uv) space,
    given the properties of the image matrix.

    Assumes that the image is a **square** matrix of size (nx, nx)
    with linearly spaced (x, y) coordinate axes.

    Parameters
    ----------
    dx : float
        Image Cell size (units: same as dist).
    nx : int
        Size of the image.
    dist : float
        Distance to the object in the image (units: same as dx).

    Returns
    ------
    The uv cell size (units: observing wavelength).

    """
    return dist / dx / nx
