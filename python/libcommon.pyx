cimport numpy as np
import numpy as np

include "galario_config.pxi"

IF DOUBLE_PRECISION:
    ctypedef double dreal
    real_dtype = np.float64

    ctypedef double complex dcomplex
    complex_dtype = np.complex128
ELSE:
    ctypedef float dreal
    real_dtype = np.float32

    ctypedef float complex dcomplex
    complex_dtype = np.complex64

cdef extern from "galario.hpp":
    # todo avoid void*
    void C_fft2d(int nx, void* data)
    void C_fftshift(int nx, void* data)
    void C_fftshift_fft2d_fftshift(int nx, void* data)
    void C_interpolate(int nx, void* data, int nd, void* u, void* v, void* fint)
    void C_apply_phase_2d(int nx, void* data, dreal x0, dreal y0)
    void C_acc_rotix(int nx, void* pixel_centers, int nd, void* u, void* v, void* indu, void* indv)
    void C_reduce_chi2(int nd, void* fobs_re, void* fobs_im, void* fint, void* weights, dreal* chi2)
    void C_acc_init()
    void C_acc_cleanup()
    void C_chi2(int nx, void* data, dreal x0, dreal y0, void* vpixel_centers, int nd, void* u, void* v, void* fobs_re, void* fobs_im, void* weights, dreal* chi2)
    int C_ngpus()
    void C_use_gpu(int device_id)

# require contiguous arrays with stride=1 in buffer[::1]
def fft2d(dcomplex[:,::1] data):
    assert data.shape[0] == data.shape[1], "Wrong data shape."

    C_fft2d(data.shape[0], <void*>&data[0,0])

def fftshift(dcomplex[:,::1] data):
    assert data.shape[0] == data.shape[1], "Wrong data shape."

    C_fftshift(data.shape[0], <void*>&data[0,0])

def fftshift_fft2d_fftshift(dcomplex[:,::1] data, shift=True):
    assert data.shape[0] == data.shape[1], "Wrong data shape."
    assert isinstance(shift, bool), "Wrong type: shift must be boolean."

    if shift is True:
        C_fftshift_fft2d_fftshift(data.shape[0], <void*>&data[0,0])
    else:
        C_fft2d(data.shape[0], <void*>&data[0,0])

def interpolate(dcomplex[:,::1] data, dreal[::1] u, dreal[::1] v):
    fint = np.empty(len(u), dtype=complex_dtype)
    C_interpolate(len(data), <void*>&data[0,0], len(u), <void*>&u[0], <void*>&v[0], <void*>np.PyArray_DATA(fint))
    return fint

def apply_phase_2d(dcomplex[:,:] data, x0, y0):
    assert data.shape[0] == data.shape[1], "Wrong data shape."

    C_apply_phase_2d(data.shape[0], <void*>&data[0,0], x0, y0)

def acc_rotix(dreal[::1] pixel_centers, dreal[::1] u, dreal[::1] v):
    assert len(u) == len(v), "Wrong array length: u, v."

    indu = np.zeros(len(u), dtype=real_dtype)
    indv = np.zeros(len(u), dtype=real_dtype)
    C_acc_rotix(len(pixel_centers), <void*> &pixel_centers[0], len(u), <void*> &u[0],  <void*> &v[0],
                <void*>np.PyArray_DATA(indu), <void*>np.PyArray_DATA(indv))

    return indu, indv

def reduce_chi2(dreal[::1] fobs_re, dreal[::1] fobs_im, dcomplex[::1] fint, dreal[::1] weights):
    nd = len(fobs_re)
    assert len(fobs_im) == nd
    assert len(weights) == nd
    assert len(fint) == nd

    cdef dreal chi2
    C_reduce_chi2(nd, <void*>&fobs_re[0], <void*>&fobs_im[0], <void*>&fint[0], <void*>&weights[0], &chi2)

    return chi2

def chi2(dcomplex[:,::1] data, x0, y0, dreal[::1] pixel_centers, dreal[::1] u, dreal[::1] v, dreal[::1] fobs_re, dreal[::1] fobs_im, dreal[::1] weights):
    assert data.shape[0] == data.shape[1], "Wrong data shape."
    assert len(u) == len(v), "Wrong array length: u, v."
    nd = len(fobs_re)
    assert len(fobs_im) == nd, "Wrong array length: fobs_im."
    assert len(weights) == nd, "Wrong array length: weights."

    cdef dreal chi2

    C_chi2(data.shape[0], <void*>&data[0,0], x0, y0, <void*> &pixel_centers[0], len(u), <void*> &u[0],  <void*> &v[0],  <void*>&fobs_re[0], <void*>&fobs_im[0], <void*>&weights[0], &chi2)

    return chi2

def ngpus():
    return C_ngpus()

def use_gpu(int device_id):
    C_use_gpu(device_id)
