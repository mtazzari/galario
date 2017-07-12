#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import, unicode_literals)

import numpy as np
import os
import pytest

from utils import *

import galario

if galario.HAVE_CUDA and int(pytest.config.getoption("--gpu")):
    from galario import double_cuda as g_double
    from galario import single_cuda as g_single
else:
    from galario import double as g_double
    from galario import single as g_single

# PARAMETERS FOR MULTIPLE TEST EXECUTIONS
par1 = {'wle_m': 0.003, 'x0_arcsec': 0.4, 'y0_arcsec': 4.}
par2 = {'wle_m': 0.00088, 'x0_arcsec': -3.5, 'y0_arcsec': 7.2}
par3 = {'wle_m': 0.00088, 'x0_arcsec': 0., 'y0_arcsec': 0.}


# use last gpu if available. Check `watch -n 0.1 nvidia-smi` to see which gpu is
# used during test execution.
ngpus = g_double.ngpus()
g_double.use_gpu(max(0, ngpus-1))

g_double.threads_per_block()



########################################################
#                                                      #
#                      TESTS                           #
#                                                      #
########################################################
@pytest.mark.parametrize("size, real_type, tol, acc_lib",
                         [(1024, 'float32', 1.e-4, g_single),
                          (1024, 'float64', 1.e-13, g_double)],
                         ids=["SP", "DP"])
def test_uv_idx(size, real_type, tol, acc_lib):
    nsamples = 10
    maxuv = 1000.

    uv = pixel_coordinates(maxuv, size).astype(real_type)
    udat, vdat = create_sampling_points(nsamples, maxuv/4.8)
    assert len(udat) == nsamples
    assert len(vdat) == nsamples
    udat = udat.astype(real_type)
    vdat = vdat.astype(real_type)

    ui, vi = get_uv_idx_n(uv, uv, udat, vdat, len(uv))
    ui = ui.astype(real_type)
    vi = vi.astype(real_type)

    ui1, vi1 = acc_lib.get_uv_idx(size, maxuv/size, udat, vdat)

    np.testing.assert_allclose(ui1, ui, rtol=tol)
    np.testing.assert_allclose(vi1, vi, rtol=tol)


@pytest.mark.parametrize("size, real_type, complex_type, rtol, atol, acc_lib",
                         [(1024, 'float32', 'complex64',  1e-7,  1e-5, g_single),
                          (1024, 'float64', 'complex128', 1e-16, 1e-8, g_double)],
                         ids=["SP", "DP"])
def test_interpolate(size, real_type, complex_type, rtol, atol, acc_lib):
    nsamples = 10000
    maxuv = 1000.

    reference_image = create_reference_image(size=size, kernel='gaussian', dtype=real_type)
    udat, vdat = create_sampling_points(nsamples, maxuv/2.2)
    # this factor has to be > than 2 because the matrix cover between -maxuv/2 to +maxuv/2,
    # therefore the sampling points have to be contained inside.

    udat = udat.astype(real_type)
    vdat = vdat.astype(real_type)

    # no rotation
    uv = pixel_coordinates(maxuv, size)
    uroti, vroti = get_uv_idx_n(uv, uv, udat, vdat, len(uv))

    uroti = uroti.astype(real_type)
    vroti = vroti.astype(real_type)

    ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(reference_image))).astype(complex_type)

    # fortran
    ReInt = int_bilin(ft.real, uroti, vroti)
    ImInt = int_bilin(ft.imag, uroti, vroti)

    # gpu
    complexInt = acc_lib.interpolate(ft,
                                     uroti.astype(real_type),
                                     vroti.astype(real_type))

    np.testing.assert_allclose(ReInt, complexInt.real, rtol, atol)
    np.testing.assert_allclose(ImInt, complexInt.imag, rtol, atol)



@pytest.mark.parametrize("size, complex_type, rtol, atol, acc_lib",
                         [(1024, 'complex64', 1.e-5, 1e-3, g_single),
                          (1024, 'complex128', 1.e-16, 1e-8, g_double)],
                         ids=["SP", "DP"])
def test_FFT(size, complex_type, rtol, atol, acc_lib):

    reference_image = create_reference_image(size=size, kernel='gaussian', dtype=complex_type)

    ft = np.fft.fft2(reference_image)

    # create a copy of reference_image because galario makes in-place FFT
    ref_complex = reference_image.copy()
    acc_lib.fft2d(ref_complex)

    # print()
    # print(np.min(np.abs(ft.real)), np.max(np.abs(ft.real)))

    # some real parts can be very close to zero, so we need atol > 0!
    np.testing.assert_allclose(ft.real, ref_complex.real, rtol, atol)
    np.testing.assert_allclose(ft.imag, ref_complex.imag, rtol, atol)


@pytest.mark.parametrize("size, complex_type, rtol, atol, acc_lib",
                         [(1000, 'complex64',  1e-7, 1e-3, g_single),
                          (1000, 'complex128', 1.e-14, 1e-8, g_double)],
                         ids=["SP", "DP"])
def test_shift_fft_shift(size, complex_type, rtol, atol, acc_lib):

    reference_image = create_reference_image(size=size, kernel='gaussian', dtype=complex_type)
    cpu_shift_fft_shift = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(reference_image)))

    # create a copy of reference_image because galario makes in-place FFT
    ref_complex = reference_image.copy()
    acc_lib.fftshift_fft2d_fftshift(ref_complex)

    # see https://github.com/mtazzari/galario/issues/28
    # print()
    # print('%.15e' % cpu_shift_fft_shift.real[0,0])
    # print('%.15e' % ref_complex.real[0,0])
    # print('---')
    np.testing.assert_allclose(cpu_shift_fft_shift, ref_complex, rtol, atol)


@pytest.mark.parametrize("size, complex_type, tol, acc_lib",
                         [(1024, 'complex64', 1.e-8, g_single),
                          (1024, 'complex128', 1.e-16, g_double)],
                         ids=["SP", "DP"])
def test_shift(size, complex_type, tol, acc_lib):

    reference_image = create_reference_image(size=size, kernel='gaussian', dtype=complex_type)

    npshifted = np.fft.fftshift(reference_image)

    ref_complex = reference_image.copy()
    acc_lib.fftshift(ref_complex)

    np.testing.assert_allclose(npshifted, ref_complex, rtol=tol)


@pytest.mark.parametrize("real_type, complex_type, rtol, atol, acc_lib, pars",
                         [('float32', 'complex64',  1.e-7,  1e-5, g_single, par1),
                          ('float64', 'complex128', 1.e-16, 1e-15, g_double, par1),
                          ('float32', 'complex64',  1.e-3,  1e-5, g_single, par2),
                          ('float64', 'complex128', 1.e-16, 1e-14, g_double, par2),
                          ('float32', 'complex64',  1.e-7,  1e-5, g_single, par3),
                          ('float64', 'complex128', 1.e-16, 1e-15, g_double, par3)],
                         ids=["SP_par1", "DP_par1",
                              "SP_par2", "DP_par2",
                              "SP_par3", "DP_par3"])
def test_apply_phase_2d(real_type, complex_type, rtol, atol, acc_lib, pars):

    wle_m = pars.get('wle_m', 0.003)
    x0_arcsec = pars.get('x0_arcsec', 0.4)
    y0_arcsec = pars.get('y0_arcsec', 10.)

    # generate the samples
    nsamples = 1000
    maxuv_generator = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)

    # compute the matrix size and maxuv given the sampling points
    size, minuv, maxuv = matrix_size(udat, vdat)
    # print("size:{0}, minuv:{1}, maxuv:{2}".format(size, minuv, maxuv))

    # create reference image (complex)
    ref_complex = create_reference_image(size=size, kernel='gaussian', dtype=complex_type)

    # compute the shift
    shifted_original_static = Fourier_shift_static(ref_complex, x0_arcsec, y0_arcsec, wle_m, maxuv)

    # GPU version
    factor = sec2rad/wle_m*maxuv
    acc_lib.apply_phase_2d(ref_complex, x0_arcsec * factor, y0_arcsec * factor)

    np.testing.assert_allclose(shifted_original_static, ref_complex, rtol, atol)


@pytest.mark.parametrize("real_type, complex_type, rtol, atol, acc_lib, pars",
                         [('float32', 'complex64',  1.e-7,  1e-5, g_single, par1),
                          ('float64', 'complex128', 1.e-16, 1e-14, g_double, par1),
                          ('float32', 'complex64',  1.e-3,  1e-5, g_single, par2),
                          ('float64', 'complex128', 1.e-16, 1e-14, g_double, par2),
                          ('float32', 'complex64',  1.e-7,  1e-5, g_single, par3),
                          ('float64', 'complex128', 1.e-16, 1e-15, g_double, par3)],
                         ids=["SP_par1", "DP_par1",
                              "SP_par2", "DP_par2",
                              "SP_par3", "DP_par3"])
def test_apply_phase_sampled(real_type, complex_type, rtol, atol, acc_lib, pars):

    x0_arcsec = pars.get('x0_arcsec', 0.4)
    y0_arcsec = pars.get('y0_arcsec', 10.)

    # generate the samples
    nsamples = 1000
    maxuv_generator = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)

    # generate mock visibility values
    fint = np.zeros(nsamples, dtype=complex_type)
    fint.real = np.random.random(nsamples) * 10.
    fint.imag = np.random.random(nsamples) * 30.

    fint_numpy = Fourier_shift_array(udat, vdat, fint.copy(), x0_arcsec, y0_arcsec)

    acc_lib.apply_phase_sampled(x0_arcsec*sec2rad , y0_arcsec*sec2rad, udat, vdat, fint)

    np.testing.assert_allclose(fint_numpy.real, fint.real, rtol, atol)
    np.testing.assert_allclose(fint_numpy.imag, fint.imag, rtol, atol)





@pytest.mark.parametrize("nsamples, real_type, tol, acc_lib",
                         [(1000, 'float32', 1.e-6, g_single),
                          (1000, 'float64', 1.e-15, g_double)],
                         ids=["SP", "DP"])
def test_reduce_chi2(nsamples, real_type, tol, acc_lib):

    x, y, w = generate_random_vis(nsamples, real_type)
    chi2_ref = np.sum(((x.real - y.real) ** 2. + (x.imag - y.imag)**2.) * w)

    chi2_loc = acc_lib.reduce_chi2(x.real.copy(order='C'), x.imag.copy(order='C'), y.copy(), w)

    # print("Chi2_ref:{0}  Chi2_acc:{1}".format(chi2_ref, chi2_loc))
    # print("Absolute diff: {0}".format(chi2_loc-chi2_ref))
    # print("Relative diff: {0}".format((chi2_loc-chi2_ref)*2./(chi2_loc+chi2_ref)))

    np.testing.assert_allclose(chi2_ref, chi2_loc, rtol=tol)


# huge inaccuracy in single precision for larger images
@pytest.mark.parametrize("nsamples, real_type, complex_type, rtol, atol, acc_lib, pars",
                         [(100, 'float32', 'complex64',  1e-4,  1e-3, g_single, par1),
                          (1000, 'float64', 'complex128', 1e-14, 1e-10, g_double, par1)],
                         ids=["SP_par1", "DP_par1"])
def test_loss(nsamples, real_type, complex_type, rtol, atol, acc_lib, pars):
    # try to find out where precision is lost

    wle_m = pars.get('wle_m', 0.003)
    x0_arcsec = pars.get('x0_arcsec', 0.4)
    y0_arcsec = pars.get('y0_arcsec', 10.)

    # generate the samples
    maxuv_generator = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)

    # compute the matrix size and maxuv
    size, minuv, maxuv = matrix_size(udat, vdat)
    uv = pixel_coordinates(maxuv, size)

    # create model complex image (it happens to have 0 imaginary part)
    reference_image = create_reference_image(size=size, kernel='gaussian', dtype=complex_type)
    ref_real = reference_image.real.copy()

    ###
    # shift - FFT - shift
    ###
    cpu_shift_fft_shift = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(reference_image)))

    shift_acc = reference_image.copy()
    acc_lib.fftshift_fft2d_fftshift(shift_acc)

    np.testing.assert_allclose(cpu_shift_fft_shift.real, shift_acc.real, rtol, atol)
    np.testing.assert_allclose(cpu_shift_fft_shift.imag, shift_acc.imag, rtol, atol)

    ###
    # phase
    ###
    uroti, vroti = get_uv_idx_n(uv, uv, udat, vdat, size)
    ReInt = int_bilin(cpu_shift_fft_shift.real, uroti, vroti).astype(real_type)
    ImInt = int_bilin(cpu_shift_fft_shift.imag, uroti, vroti).astype(real_type)
    fint = ReInt + 1j*ImInt
    fint_acc = fint.copy()
    fint_shifted = Fourier_shift_array(udat/wle_m, vdat/wle_m, fint, x0_arcsec, y0_arcsec)
    acc_lib.apply_phase_sampled(x0_arcsec*sec2rad, y0_arcsec*sec2rad, udat/wle_m, vdat/wle_m, fint_acc)


    # lose some absolute precision here  --> not anymore
    # atol *= 2
    np.testing.assert_allclose(fint_shifted.real, fint_acc.real, rtol, atol)
    np.testing.assert_allclose(fint_shifted.imag, fint_acc.imag, rtol, atol)
    # but continue with previous tolerance
    # atol /= 2

    ###
    # rotation indices
    ###
    uroti, vroti = get_uv_idx_n(uv, uv, udat, vdat, size)
    ui1, vi1 = acc_lib.get_uv_idx(size, maxuv/size, udat.astype(real_type), vdat.astype(real_type))

    np.testing.assert_allclose(ui1, uroti, rtol, atol)
    np.testing.assert_allclose(vi1, vroti, rtol, atol)


    ###
    # interpolation
    ###
    ReInt = int_bilin(cpu_shift_fft_shift.real, uroti, vroti).astype(real_type)
    ImInt = int_bilin(cpu_shift_fft_shift.imag, uroti, vroti).astype(real_type)
    complexInt = acc_lib.interpolate(shift_acc, uroti.astype(real_type), vroti.astype(real_type))

    np.testing.assert_allclose(ReInt, complexInt.real, rtol, atol)
    np.testing.assert_allclose(ImInt, complexInt.imag, rtol, atol)

    ###
    # now all steps in one function
    ###
    sampled = acc_lib.sample(ref_real, x0_arcsec, y0_arcsec,
                             maxuv/size/wle_m, udat/wle_m, vdat/wle_m)

    # a lot of precision lost. Why? --> not anymore
    # rtol = 1
    # atol = 0.5
    np.testing.assert_allclose(fint_shifted.real, sampled.real, rtol, atol)
    np.testing.assert_allclose(fint_shifted.imag, sampled.imag, rtol, atol)


# single precision difference can be -1.152496e-01 vs 1.172152e+00 for large 1000x1000 images!!
@pytest.mark.parametrize("nsamples, real_type, complex_type, rtol, atol, acc_lib, pars",
                         [(1000, 'float32', 'complex64',  1e-3,  1e-4, g_single, par1),
                          (1000, 'float64', 'complex128', 1e-12, 1e-11, g_double, par1),
                          (1000, 'float32', 'complex64',  1e-3,  1e-3, g_single, par2), ## large x0, y0 induce larger errors
                          (1000, 'float64', 'complex128', 1e-12, 1e-10, g_double, par2),
                          (1000, 'float32', 'complex64',  1e-3,  1e-5, g_single, par3),
                          (1000, 'float64', 'complex128', 1e-12, 1e-11, g_double, par3)],
                         ids=["SP_par1", "DP_par1",
                              "SP_par2", "DP_par2",
                              "SP_par3", "DP_par3"])
def test_sample(nsamples, real_type, complex_type, rtol, atol, acc_lib, pars):
    # go for fairly low precision when we add up many large numbers, we loose precision
    # TODO: perhaps implement the test with more realistic values of chi2 ~ 1

    wle_m = pars.get('wle_m', 0.003)
    x0_arcsec = pars.get('x0_arcsec', 0.4)
    y0_arcsec = pars.get('y0_arcsec', 10.)

    # generate the samples
    maxuv_generator = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)

    # compute the matrix size and maxuv
    size, minuv, maxuv = matrix_size(udat, vdat)
    uv = pixel_coordinates(maxuv, size, dtype=real_type)

    # create model image (it happens to have 0 imaginary part)
    reference_image = create_reference_image(size=size, kernel='gaussian', dtype=complex_type)
    ref_real = reference_image.real.copy()

    # CPU version
    cpu_shift_fft_shift = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(reference_image)))
    uroti, vroti = get_uv_idx_n(uv, uv, udat, vdat, size)
    ReInt = int_bilin(cpu_shift_fft_shift.real, uroti, vroti).astype(real_type)
    ImInt = int_bilin(cpu_shift_fft_shift.imag, uroti, vroti).astype(real_type)
    fint = ReInt + 1j*ImInt
    fint_shifted = Fourier_shift_array(udat/wle_m, vdat/wle_m, fint, x0_arcsec, y0_arcsec)

    # GPU
    sampled = acc_lib.sample(ref_real, x0_arcsec, y0_arcsec,
                             maxuv/size/wle_m, udat/wle_m, vdat/wle_m)

    np.testing.assert_allclose(fint_shifted.real, sampled.real, rtol, atol)
    np.testing.assert_allclose(fint_shifted.imag, sampled.imag, rtol, atol)


@pytest.mark.parametrize("nsamples, real_type, complex_type, rtol, atol, acc_lib, pars",
                         [(1000, 'float32', 'complex64', 8.e-3, 8.e-3, g_single, par1),
                          (1000, 'float64', 'complex128', 1.e-14, 1.e-14, g_double, par1),
                          (1000, 'float32', 'complex64', 5.e-2, 8.e-3, g_single, par2),
                          (1000, 'float64', 'complex128', 1.e-10, 1.e-10, g_double, par2),
                          (1000, 'float32', 'complex64', 8.e-3, 8.e-3,g_single, par3),
                          (1000, 'float64', 'complex128', 1.e-14, 1.e-14, g_double, par3)],
                         ids=["SP_par1", "DP_par1",
                              "SP_par2", "DP_par2",
                              "SP_par3", "DP_par3"])
def test_chi2(nsamples, real_type, complex_type, rtol, atol, acc_lib, pars):
    # go for fairly low precision when we add up many large numbers, we loose precision
    # TODO: perhaps implement the test with more realistic values of chi2 ~ 1

    wle_m = pars.get('wle_m', 0.003)
    x0_arcsec = pars.get('x0_arcsec', 0.4)
    y0_arcsec = pars.get('y0_arcsec', 10.)

    # generate the samples
    maxuv_generator = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)
    x, _, w = generate_random_vis(nsamples, real_type)

    # compute the matrix size and maxuv
    size, minuv, maxuv = matrix_size(udat, vdat)
    #print("size:{0}, minuv:{1}, maxuv:{2}".format(size, minuv, maxuv))
    uv = pixel_coordinates(maxuv, size).astype(real_type)

    # create model image (it happens to have 0 imaginary part)
    ref_complex = create_reference_image(size=size, kernel='gaussian', dtype=complex_type)
    ref_real = ref_complex.real.copy()

    # CPU version
    cpu_shift_fft_shift = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ref_complex)))

    # compute interpolation and chi2
    uroti, vroti = get_uv_idx_n(uv, uv, udat, vdat, size)
    ReInt = int_bilin(cpu_shift_fft_shift.real, uroti, vroti).astype(real_type)
    ImInt = int_bilin(cpu_shift_fft_shift.imag, uroti, vroti).astype(real_type)
    fint = ReInt + 1j*ImInt
    fint_shifted = Fourier_shift_array(udat/wle_m, vdat/wle_m, fint, x0_arcsec, y0_arcsec)

    chi2_ref = np.sum(((fint_shifted.real - x.real)**2. + (fint_shifted.imag - x.imag)**2.) * w)

    # GPU
    chi2_cuda = acc_lib.chi2(ref_real, x0_arcsec, y0_arcsec,
                             maxuv/size/wle_m, udat/wle_m, vdat/wle_m, x.real.copy(), x.imag.copy(), w)

    np.testing.assert_allclose(chi2_ref, chi2_cuda, rtol=rtol, atol=atol)


# a test case for profiling. Avoid python calls as much as possible.
def test_profile():
    nsamples = 512
    real_type = 'float64'
    complex_type = 'complex128'

    wle_m = par1.get('wle_m', 0.003)
    x0_arcsec = par1.get('x0_arcsec', 0.4)
    y0_arcsec = par1.get('y0_arcsec', 10.)

    # generate the samples
    maxuv_generator = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)
    x, _, w = generate_random_vis(nsamples, real_type)

    # compute the matrix size and maxuv
    size, minuv, maxuv = matrix_size(udat, vdat, force_nx=4096)
    #print("size:{0}, minuv:{1}, maxuv:{2}".format(size, minuv, maxuv))
    uv = pixel_coordinates(maxuv, size).astype(real_type)

    # create model image (it happens to have 0 imaginary part)
    reference_image = create_reference_image(size=size, kernel='gaussian', dtype=real_type)
    ref_complex = reference_image.copy()

    chi2_cuda = g_double.chi2(ref_complex, x0_arcsec, y0_arcsec,
                             maxuv/size/wle_m, udat/wle_m, vdat/wle_m, x.real.copy(), x.imag.copy(), w)
