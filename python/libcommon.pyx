cimport numpy as np
import numpy as np
from cpython cimport Py_INCREF

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

include "galario_config.pxi"

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
    void* _galario_copy_input(int nx, int ny, void* realdata);
    void* _galario_fft2d(int nx, int ny, void* data)
    void _galario_fftshift(int nx, int ny, void* data)
    void _galario_fftshift_axis0(int nx, int ny, void* data);
    void _galario_interpolate(int nx, int ncol, void* data, int nd, void* u, void* v, dreal duv, void* fint)
    void _galario_apply_phase_sampled(dreal dRA, dreal dDec, int nd, void* u, void* v, void* fint)
    void _galario_reduce_chi2(int nd, void* fobs_re, void* fobs_im, void* fint, void* weights, dreal* chi2)
    void _galario_sample(int nx, int ny, void* data, dreal dRA, dreal dDec, dreal duv, int nd, void* u, void* v, void* fint)
    void _galario_sweep(int nr, void* ints, dreal Rmin, dreal dR, int nrow, int ncol, dreal dxy, dreal inc, void* image)
    void _galario_sampleProfile(int nr, void* ints, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal dist, dreal inc, dreal dRA, dreal dDec, dreal duv, int nd, void* u, void* v, void* fint)
    void _galario_chi2(int nx, int ny, void* data, dreal dRA, dreal dDec, dreal duv, int nd, void* u, void* v, void* fobs_re, void* fobs_im, void* weights, dreal* chi2)

cdef extern from "galario.h":
    int  galario_threads_per_block(int num);
    void galario_init();
    void galario_cleanup();
    void galario_free(void*);
    void galario_use_gpu(int device_id)
    int  galario_ngpus()

cdef extern from "fftw3.h":
    void fftw_free(void*)

# constants
sec2rad = 1./3600.*np.pi/180.    # from arcsec to radians

def _check_data(data):
    # assert data.shape[0] == data.shape[1], "Expect a square image but got shape %s" % data.shape
    return True

def _check_obs(fobs_re, fobs_im, weights, fint=None, u=None, v=None):
    nd = len(fobs_re)
    assert len(fobs_im) == nd, "Wrong array length: fobs_im."
    assert len(weights) == nd, "Wrong array length: weights."
    if fint is not None:
        assert len(fint) == nd, "Wrong array length: fint."
    if u is not None:
        assert len(u) == nd, "Wrong array length: u"
    if v is not None:
        assert len(v) == nd, "Wrong array length: v"


def _check_uv(u, v):
    assert len(u) == len(v), "Wrong array length: u, v."


cdef class ArrayWrapper:
    """Wrap an array created by `fftw_alloc`"""
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
        self.set_data(nx, ny, data_ptr)
        cdef np.npy_intp shape[2]
        shape[:] = (self.nx, int(self.ny/2)+1)

        # Create a 2D array, of length `nx*ny/2+1`
        ndarray = np.PyArray_SimpleNewFromData(2, shape, complex_typenum, self.data_ptr)

        # without this, data would be cleaned up right away
        # TODO If nobody calls Py_DECREF, will the memory ever be deallocated?
        Py_INCREF(self)
        return ndarray

    def __dealloc__(self):
        """ Frees the array. This is called by Python when all the
        references to the object are gone. """
        print("Deallocating array")
        galario_free(self.data_ptr)


def sample(dreal[:,::1] data, dRA, dDec, duv, dreal[::1] u, dreal[::1] v):
    """
    Performs Fourier transform, translation by (dRA, dDec) and sampling in (u, v) locations of a given image.

    # TODO: add that FT operations are done in-place, and padding might be required
    #       if user creates smaller images.

    Typical call signature::

        sample(image, dRA, dDec, duv, u, v)

    Parameters
    ----------
    image : array_like, float
        Matrix of size (nx, ny) containing the object brightness distribution.
        units: Jy/pixel.
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
        (units: observing wavelength).
    v: array_like, float
        u coordinate of the visibility points where the FT has to be sampled.
        (units: observing wavelength).

    Returns
    -------
    fint: array_like, complex
        Sampled values of the translated Fourier transform of data.

    Example
    -------
    Typical usage::

        from galario import sample, uvcell_size

        wle = 0.88e-3  # observing wavelength (m)
        nx = image.shape[0]  # size of the square matrix containing the image.
        dx = 1.49e14  # cm
        dist = 3.1e20  # cm
        duv = uvcell_size(dist, dx, nx)
        fint = sample(image, dRA, dDec, duv, u/wle, v/wle)
        Re_V = fint.real
        Im_V = fint.imag

    """
    _check_data(data)
    fint = np.zeros(len(u), dtype=complex_dtype)
    _galario_sample(data.shape[0], data.shape[1], <void*>&data[0,0], dRA, dDec, duv, len(u), <void*>&u[0], <void*>&v[0], <void*>np.PyArray_DATA(fint))

    return fint


def get_image_size(dist, u, v, max_f=4., min_f=3.):
    """ dist: cm;  u and v in lambda"""
    uvdist = np.hypot(u, v)
    max_uv = np.max(uvdist)*max_f
    min_uv = np.min(uvdist)/min_f

    nxy = 2**np.ceil(np.log2(max_uv/min_uv))
    dxy = dist/max_uv

    return dxy, nxy


def sampleProfile(dreal[::1] ints, Rmin, dR, dist, dRA, dDec, dreal[::1] u, dreal[::1] v, inc=0., dxy=None, nxy=None):
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
    fint: array_like, complex
        Sampled values of the translated Fourier transform of data.

    Example
    -------
    Typical usage::

        from galario import sample, uvcell_size

        wle = 0.88e-3  # observing wavelength (m)
        nx = image.shape[0]  # size of the square matrix containing the image.
        dx = 1.49e14  # cm
        dist = 3.1e20  # cm
        duv = uvcell_size(dist, dx, nx)
        fint = sample(image, dRA, dDec, duv, u/wle, v/wle)
        Re_V = fint.real
        Im_V = fint.imag

    """
    if not dxy and not nxy:
        dxy, nxy = get_image_size(dist, u, v)
    else:
        # user must provide both of them
        # do checks that dxy, nxy, dist satisfy Nyquist sampling given the u, v data points.
        pass

    # _check_data(data)
    duv = uvcell_size(dxy, nxy, dist)
    fint = np.zeros(len(u), dtype=complex_dtype)
    _galario_sampleProfile(len(ints), <void*>&ints[0], Rmin, dR, dxy, nxy, dist, inc, dRA, dDec, duv, len(u), <void*>&u[0], <void*>&v[0], <void*>np.PyArray_DATA(fint))

    return fint

def sweep(dreal[::1] ints, Rmin, dR, nx, ny, dxy, inc):
    """
    sweep

    Remove last column(s) for a (nx, ny) image

    """
    image = np.empty((nx, ny//2+1), dtype=complex_dtype, order='C')
    _galario_sweep(len(ints), <void*>&ints[0], Rmin, dR, nx, ny, dxy, inc, <void*>np.PyArray_DATA(image))

    # skip last two padding columns
    return image.view(dtype=real_dtype)[:, :-2]

def apply_phase_sampled(dRA, dDec, dreal[::1] u, dreal[::1] v, dcomplex[::1] fint):
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
    fint : array_like, complex
        complex visibilities, of form Real(Vis) + i*Imag(Vis).
        units: Jy

    Returns
    -------
    fint_out : array_like, complex
        shifted complex visibilities
        units: arbitrary, same as fint

    TODO change `fint` name into `vis`

    """
    dRA *= sec2rad
    dDec *= sec2rad

    fint_out = np.copy(fint, order='C')
    _galario_apply_phase_sampled(dRA, dDec, len(fint), <void*> &u[0], <void*> &v[0], <void*>np.PyArray_DATA(fint_out))

    return fint_out


# require contiguous arrays with stride=1 in buffer[::1]
def fft2d(dreal[:,::1] data):
    _check_data(data)
    nx, ny = data.shape[0], data.shape[1]
    cdef void* res = _galario_copy_input(nx, ny, <void*>&data[0,0])

    _galario_fft2d(nx, ny, res)

    # Use a custom delete function to free the array http://gael-varo1quaux.info/programming/cython-example-of-exposing-c-computed-arrays-in-python-without-data-copies.html
    return ArrayWrapper().as_ndarray(nx, ny, res)


def fftshift(dreal[:,::1] data):
    _check_data(data)
    nx, ny = data.shape[0], data.shape[1]
    cdef void* res = _galario_copy_input(nx, ny, <void*>&data[0,0])

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


def interpolate(dcomplex[:,::1] data, dreal[::1] u, dreal[::1] v, duv):
    fint = np.empty(len(u), dtype=complex_dtype)
    _galario_interpolate(data.shape[0], data.shape[1], <void*>&data[0,0], len(u), <void*>&u[0], <void*>&v[0], duv, <void*>np.PyArray_DATA(fint))
    return fint


def reduce_chi2(dreal[::1] fobs_re, dreal[::1] fobs_im, dcomplex[::1] fint, dreal[::1] weights):
    _check_obs(fobs_re, fobs_im, weights, fint)

    cdef dreal chi2
    _galario_reduce_chi2(len(fint), <void*>&fobs_re[0], <void*>&fobs_im[0], <void*>&fint[0], <void*>&weights[0], &chi2)

    return chi2


def chi2(dreal[:,::1] data, dRA, dDec, dreal duv, dreal[::1] u, dreal[::1] v, dreal[::1] fobs_re, dreal[::1] fobs_im, dreal[::1] weights):
    _check_data(data)
    _check_obs(fobs_re, fobs_im, weights)

    cdef dreal chi2

    _galario_chi2(data.shape[0], data.shape[1], <void*>&data[0,0], dRA, dDec, duv, len(u), <void*> &u[0],  <void*> &v[0],  <void*>&fobs_re[0], <void*>&fobs_im[0], <void*>&weights[0], &chi2)

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


# TODO update to rectangular images
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
