cimport numpy as np
import numpy as np

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
    void* _galario_fft2d(int nx, void* data)
    void _galario_fftshift(int nx, void* data)
    void _galario_fftshift_axis0(int nx, int ny, void* data);
    void _galario_interpolate(int nx, void* data, int nd, void* u, void* v, void* fint)
    void _galario_apply_phase_2d(int nx, void* data, dreal dRA, dreal dDec)
    void _galario_apply_phase_sampled(dreal dRA, dreal dDec, int nd, void* u, void* v, void* fint)
    void _galario_get_uv_idx(int nx, dreal du, int nd, void* u, void* v, void* indu, void* indv)
    void _galario_get_uv_idx_R2C(int nx, dreal du, int nd, void* u, void* v, void* indu, void* indv);
    void _galario_reduce_chi2(int nd, void* fobs_re, void* fobs_im, void* fint, void* weights, dreal* chi2)
    void _galario_sample(int nx, void* data, dreal dRA, dreal dDec, dreal du, int nd, void* u, void* v, void* fint);
    void _galario_chi2(int nx, void* data, dreal dRA, dreal dDec, dreal du, int nd, void* u, void* v, void* fobs_re, void* fobs_im, void* weights, dreal* chi2)

cdef extern from "galario.h":
    int  galario_threads_per_block(int num);
    void galario_acc_init();
    void galario_acc_cleanup();
    void galario_use_gpu(int device_id)
    int  galario_ngpus()

def _check_data(data):
    assert data.shape[0] == data.shape[1], "Expect a square image but got shape %s" % data.shape


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


def sample(dreal[:,::1] data, dRA, dDec, du, dreal[::1] u, dreal[::1] v):
    """
    Performs Fourier transform, translation by (dRA, dDec) and sampling in (u, v) locations of a given image.

    # TODO: add that FT operations are done in-place, and padding might be required
    #       if user creates smaller images.

    Typical call signature::

      sample(image, dRA, dDec, du, u, v)

    Parameters
    ----------
    image: 2d array_like, float
        Square matrix of size (nx, nx) containing the object brightness distribution.
        units: Jy/pixel.
    dRA: float
        X-axis offset by which the image has to be translated.
        units: arcseconds
    dDec: float
        Y-axis offset by which the image has to be translated.
        units: arcseconds
    du: float
        uv cell size in the Fourier space.
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
        from galario import sample, uvcell_size

        wle = 0.88e-3  # observing wavelength (m)
        nx = image.shape[0]  # size of the square matrix containing the image.
        dx = 1.49e14  # cm
        dist = 3.1e20  # cm
        du = uvcell_size(dist, dx, nx)
        fint = sample(image, dRA, dDec, du, u/wle, v/wle)
        Re_V = fint.real
        Im_V = fint.imag

    """
    _check_data(data)
    fint = np.zeros(len(u), dtype=complex_dtype)
    _galario_sample(len(data), <void*>&data[0,0], dRA, dDec, du, len(u), <void*>&u[0], <void*>&v[0], <void*>np.PyArray_DATA(fint))

    return fint

# TODO wrap memory with custom deleter for fftw_free as in here
# http://gael-varoquaux.info/programming/cython-example-of-exposing-c-computed-arrays-in-python-without-data-copies.html

# require contiguous arrays with stride=1 in buffer[::1]
def fft2d(dreal[:,::1] data):
    _check_data(data)

    cdef void* res = _galario_fft2d(data.shape[0], <void*>&data[0,0])

    # follow https://stackoverflow.com/questions/25102409/
    cdef extern from "numpy/arrayobject.h":
        void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

    cdef np.npy_intp size[2]
    size[:] = [data.shape[0], int(data.shape[0]/2)+1]
    cdef np.ndarray[dcomplex, ndim=2] arr = np.PyArray_SimpleNewFromData(2, size, complex_typenum, res)
    PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)

    return arr

def fftshift(dcomplex[:,::1] data):
    assert data.shape[0] == data.shape[1], "Wrong data shape."

    _galario_fftshift(data.shape[0], <void*>&data[0,0])


def fftshift_axis0(dcomplex[:,::1] data):
    """
    Swaps the upper and lower halves of a matrix,
    equivalent to np.fft.fftshift(data, axes=0).

    data.shape[0] has to be even.

    """
    assert data.shape[0] %  2 == 0, "Axis 0 of data has to be even "
    _galario_fftshift_axis0(data.shape[0], data.shape[1], <void*>&data[0,0])


def interpolate(dcomplex[:,::1] data, dreal[::1] u, dreal[::1] v):
    fint = np.empty(len(u), dtype=complex_dtype)
    _galario_interpolate(len(data), <void*>&data[0,0], len(u), <void*>&u[0], <void*>&v[0], <void*>np.PyArray_DATA(fint))
    return fint


def apply_phase_2d(dcomplex[:,:] data, dRA, dDec):
    assert data.shape[0] == data.shape[1], "Wrong data shape."

    _galario_apply_phase_2d(data.shape[0], <void*>&data[0,0], dRA, dDec)


def apply_phase_sampled(dRA, dDec, dreal[::1] u, dreal[::1] v, dcomplex[::1] fint):

    _galario_apply_phase_sampled(dRA, dDec, len(fint), <void*> &u[0], <void*> &v[0], <void*> &fint[0])


def get_uv_idx(nx, du, dreal[::1] u, dreal[::1] v):
    assert len(u) == len(v), "Wrong array length: u, v."

    indu = np.zeros(len(u), dtype=real_dtype)
    indv = np.zeros(len(u), dtype=real_dtype)
    _galario_get_uv_idx(nx, du, len(u), <void*> &u[0],  <void*> &v[0],
                <void*>np.PyArray_DATA(indu), <void*>np.PyArray_DATA(indv))

    return indu, indv

def get_uv_idx_R2C(nx, du, dreal[::1] u, dreal[::1] v):
    assert len(u) == len(v), "Wrong array length: u, v."

    indu = np.zeros(len(u), dtype=real_dtype)
    indv = np.zeros(len(u), dtype=real_dtype)
    _galario_get_uv_idx_R2C(nx, du, len(u), <void*> &u[0],  <void*> &v[0],
                <void*>np.PyArray_DATA(indu), <void*>np.PyArray_DATA(indv))

    return indu, indv

def reduce_chi2(dreal[::1] fobs_re, dreal[::1] fobs_im, dcomplex[::1] fint, dreal[::1] weights):
    _check_obs(fobs_re, fobs_im, weights, fint)

    cdef dreal chi2
    _galario_reduce_chi2(len(fint), <void*>&fobs_re[0], <void*>&fobs_im[0], <void*>&fint[0], <void*>&weights[0], &chi2)

    return chi2


def chi2(dreal[:,::1] data, dRA, dDec, dreal du, dreal[::1] u, dreal[::1] v, dreal[::1] fobs_re, dreal[::1] fobs_im, dreal[::1] weights):
    _check_data(data)
    _check_obs(fobs_re, fobs_im, weights)

    cdef dreal chi2

    _galario_chi2(data.shape[0], <void*>&data[0,0], dRA, dDec, du, len(u), <void*> &u[0],  <void*> &v[0],  <void*>&fobs_re[0], <void*>&fobs_im[0], <void*>&weights[0], &chi2)

    return chi2


def init():
    galario_acc_init()


def cleanup():
    galario_acc_cleanup()


def ngpus():
    return galario_ngpus()


def use_gpu(int device_id):
    galario_use_gpu(device_id)


def threads_per_block(int num=0):
    return galario_threads_per_block(num)


def uvcell_size(dx, nx, dist):
    """
    Computes the cell size in the Fourier (uv) space,
    given the properties of the image matrix.

    Assumes that the image is a **square** matrix of size (nx, nx)
    with linearly spaced (x, y) coordinate axes.

    Parameters
    ----------
    dx: float
        Image Cell size (units: same as dist).
    nx: int
        Size of the image.
    dist: float
        Distance to the object in the image (units: same as dx).

    Return
    ------
    The uv cell size (units: observing wavelength).

    """
    return dist / dx / nx
