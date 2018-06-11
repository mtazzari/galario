###############################################################################
# This file is part of GALARIO:                                               #
# Gpu Accelerated Library for Analysing Radio Interferometer Observations     #
#                                                                             #
# Copyright (C) 2017-2018, Marco Tazzari, Frederik Beaujean, Leonardo Testi.  #
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import, unicode_literals)

import numpy as np
import pytest

from utils import *

import galario
from galario import deg, arcsec

if galario.HAVE_CUDA and int(pytest.config.getoption("--gpu")):
    from galario import double_cuda as g_double
    from galario import single_cuda as g_single
else:
    from galario import double as g_double
    from galario import single as g_single

# PARAMETERS FOR MULTIPLE TEST EXECUTIONS
par1 = {'dRA': 0., 'dDec': 0.4, 'PA': 2., 'nxy': 1024}
par2 = {'dRA': -3.5, 'dDec': 7.2, 'PA': -23., 'nxy': 2048}
par3 = {'dRA': 2.3, 'dDec': 3.2, 'PA': 88., 'nxy': 4096}
par4 = {'dRA': 0., 'dDec': 0., 'PA': 145., 'nxy': 1024}


# use last gpu if available. Check `watch -n 0.1 nvidia-smi` to see which gpu is
# used during test execution.
ngpus = g_double.ngpus()
g_double.use_gpu(0) #max(0, ngpus-1))

g_double.threads()


########################################################
#                                                      #
#                      TESTS                           #
#                                                      #
########################################################

@pytest.mark.parametrize("Rmin, dR, nrad, nxy, dxy, inc, profile_mode, real_type",
                          [(1e-6, 0.001, 2000, 1024, 0.2, 20., 'Gauss', 'float64'),
                           (1e-6, 0.001, 2000, 2048, 0.2, 44.23, 'Cos-Gauss', 'float64'),
                           (1e-6, 0.001, 2000, 2048, 0.5, 20., 'Gauss', 'float64'),
                           (1e-6, 0.001, 2000, 1024, 0.3, 20., 'Gauss', 'float64')],
                          ids=["{}".format(i) for i in range(4)])
def test_intensity_sweep(Rmin, dR, nrad, nxy, dxy, inc, profile_mode, real_type):
    """
    Test the image creation algorithm, `sweep`.

    """
    Rmin *= arcsec
    dR *= arcsec
    dxy *= arcsec
    inc = np.radians(inc)

    # compute radial profile
    intensity = radial_profile(Rmin, dR, nrad, profile_mode, dtype=real_type, gauss_width=dxy*6)

    nrow, ncol = nxy, nxy

    image_ref = sweep_ref(intensity, Rmin, dR, nrow, ncol, dxy, inc, dtype_image=real_type)

    image_sweep_galario = g_double.sweep(intensity, Rmin, dR, nxy, dxy, inc)

    image_prototype = g_sweep_prototype(intensity, Rmin, dR, nrow, ncol, dxy, inc, dtype_image=real_type)

    # uncomment for debugging
    # plot images
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.matshow(image_sweep_galario)
    # plt.savefig("./test_intensity_sweep_galario.pdf")
    # plt.clf()
    # plt.matshow(image_ref)
    # plt.savefig("./test_intensity_sweep_ref.pdf")

    # plot cuts - benchmark
    # for line_no in [0, nx//2-1, nx//2, nx//2+1, nx-1]:
    #     imin, imax = nx//2 - 10, nx//2 + 10
    #     # plt.plot(image_ref[line_no, imin:imax], '.-', label=line_no)
    #     # plt.plot(image_g_sweep_prototype[line_no, imin:imax], '.--', ms=3, lw=0.3, label=line_no)
    #     plt.plot(image_ref[line_no, imin:imax]-image_g_sweep_prototype[line_no, imin:imax], '.--', ms=3, lw=0.3, label=line_no)
    # plt.legend()
    # plt.savefig("./profile_intensity_ref.pdf")
    # plt.clf()

    assert_allclose(image_ref, image_prototype, rtol=1.e-12, atol=0)
    assert_allclose(image_prototype, image_sweep_galario, rtol=1.e-12, atol=0)


@pytest.mark.parametrize("nsamples, real_type, rtol, atol, acc_lib, pars",
                          [(1000, 'float64', 1e-6, 0, g_double, par1),
                          (1000, 'float64',  1e-6, 0, g_double, par2),
                          (1000, 'float64',  1e-6, 0, g_double, par3),
                          (1000, 'float64',  1e-6, 0, g_double, par4)],
                         ids=["{}".format(i) for i in range(4)])
def test_R2C_vs_C2C(nsamples, real_type, rtol, atol, acc_lib, pars):
    """
    Test the (current) R2C implementation against the (old) C2C one.
    # it is possible that 0.0002% of points differ at rtol>1e-5

    """
    dRA = pars['dRA']
    dDec = pars['dDec']
    PA = pars['PA']
    nxy = pars['nxy']

    # generate the samples
    maxuv_generator = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)

    # compute the matrix nxy and maxuv
    _, minuv, maxuv = matrix_size(udat, vdat)
    du = maxuv/nxy

    # create model image (it happens to have 0 imaginary part)
    reference_image = create_reference_image(nxy, -5., 2., dtype=real_type)
    ref_real = reference_image.copy()

    # CPU version
    PA *= deg
    dRA *= arcsec
    dDec *= arcsec
    dRArot, dDecrot, urot, vrot = apply_rotation(PA, dRA, dDec, udat, vdat)
    dRArot_g, dDecrot_g, urot_g, vrot_g = acc_lib.uv_rotate(PA, dRA, dDec, udat, vdat)

    np.testing.assert_allclose(dRArot, dRArot_g)
    np.testing.assert_allclose(dDecrot, dDecrot_g)
    np.testing.assert_allclose(urot, urot_g)
    np.testing.assert_allclose(vrot, vrot_g)

    #  1) C2C (numpy)
    fft_c2c_shifted = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(reference_image.copy())))
    uroti_c2c, vroti_c2c = uv_idx(urot, vrot, du, nxy/2.)
    ReInt_c2c = int_bilin_MT(fft_c2c_shifted.real, uroti_c2c, vroti_c2c)
    ImInt_c2c = int_bilin_MT(fft_c2c_shifted.imag, uroti_c2c, vroti_c2c)
    vis_c2c = ReInt_c2c + 1j*ImInt_c2c
    vis_c2c_shifted = apply_phase_array(urot, vrot, vis_c2c, dRArot, dDecrot)

    # CPU/GPU version (galario)
    dxy = 1./nxy/du
    vis_galario = acc_lib.sampleImage(ref_real, dxy, udat, vdat, dRA=dRA, dDec=dDec, PA=PA)

    # check python c2c vs galario
    assert_allclose(vis_galario.real, vis_c2c_shifted.real, rtol, atol)
    assert_allclose(vis_galario.imag, vis_c2c_shifted.imag, rtol, np.abs(np.mean(vis_galario.real))*rtol)


# single precision less precise if code compiled with `-ffast-math`, otherwise rtol=1e-7 passes
@pytest.mark.parametrize("size, real_type, complex_type, rtol, atol, acc_lib",
                         [(1024, 'float32', 'complex64',  2e-4,  1e-5, g_single),
                          (1024, 'float64', 'complex128', 1e-16, 1e-8, g_double)],
                         ids=["SP", "DP"])
def test_interpolate(size, real_type, complex_type, rtol, atol, acc_lib):
    """
    Test the interpolation of the output FT.

    """
    nsamples = 10000
    maxuv = 1000.

    reference_image = create_reference_image(size=size, dtype=real_type)
    udat, vdat = create_sampling_points(nsamples, maxuv/2.2)
    # this factor has to be > than 2 because the matrix cover between -maxuv/2 to +maxuv/2,
    # therefore the sampling points have to be contained inside.

    udat = udat.astype(real_type)
    vdat = vdat.astype(real_type)

    # no rotation
    du = maxuv/size
    uroti, vroti = uv_idx_r2c(udat, vdat, du, size/2.)

    uroti = uroti.astype(real_type)
    vroti = vroti.astype(real_type)

    ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(reference_image))).astype(complex_type)

    ReInt = int_bilin_MT(ft.real, uroti, vroti)
    ImInt = int_bilin_MT(ft.imag, uroti, vroti)
    uneg = udat < 0.
    ImInt[uneg] *= -1.

    complexInt = acc_lib.interpolate(ft, du,
                                     udat.astype(real_type),
                                     vdat.astype(real_type))

    assert_allclose(ReInt, complexInt.real, rtol, atol)
    assert_allclose(ImInt, complexInt.imag, rtol, atol)


@pytest.mark.parametrize("size, real_type, rtol, atol, acc_lib",
                         [(1024, 'float32', 1.e-5, 1e-3, g_single),
                          (1024, 'float64', 1.e-16, 1e-8, g_double)],
                         ids=["SP", "DP"])
def test_FFT(size, real_type, rtol, atol, acc_lib):
    """
    Test the Real to Complex FFTW/cuFFT against numpy Complex to Complex.

    """
    reference_image = create_reference_image(size=size, dtype=real_type)

    ft = np.fft.fft2(reference_image)

    acc_res = acc_lib._fft2d(reference_image)

    # outputs of different shape because np doesn't use the redundancy y[i] == y[n-i] for i>0
    np.testing.assert_equal(ft.shape[0], acc_res.shape[0])
    np.testing.assert_equal(acc_res.shape[1], int(acc_res.shape[0]/2)+1)

    # some real parts can be very close to zero, so we need atol > 0!
    # only get the 0-th and the first half of columns to compare to compact FFTW output
    assert_allclose(unique_part(ft).real, acc_res.real, rtol, atol)
    assert_allclose(unique_part(ft).imag, acc_res.imag, rtol, atol)


@pytest.mark.parametrize("size, real_type, tol, acc_lib",
                         [(1024, 'float32', 1.e-8, g_single),
                          (1024, 'float64', 1.e-16, g_double)],
                         ids=["SP", "DP"])
def test_shift_axes01(size, real_type, tol, acc_lib):
    """
    Test the 1st shift to be applied to the input image before the FFT.

    """
    # just a create a runtime-typical image with a big offset disk
    reference_image = create_reference_image(size=size, x0=size/10., y0=-size/10.,
                                            sigma_x=3.*size, sigma_y=2.*size, dtype=real_type)

    npshifted = np.fft.fftshift(reference_image)

    ref_complex = reference_image.copy()
    acc_shift_real = acc_lib._fftshift(ref_complex)

    # interpret complex array as real and skip last two columns
    real_view = acc_shift_real.view(dtype=real_type)[:, :-2]

    assert_allclose(npshifted, real_view, rtol=tol)


@pytest.mark.parametrize("size, complex_type, tol, acc_lib",
                         [(1024, 'complex64', 1.e-8, g_single),
                          (1024, 'complex128', 1.e-16, g_double)],
                         ids=["SP", "DP"])
def test_shift_axis0(size, complex_type, tol, acc_lib):
    """
    Test the 2nd shift to be applied to the output of the FFT.

    """
    #  the reference image has the shape of the typical output of FFTW R2C,
    #  but acc_lib.fftshift_axis0() works for every matrix size.
    reference_image = np.random.random((size, int(size/2)+1)).astype(complex_type)

    # numpy reference
    npshifted = np.fft.fftshift(reference_image, axes=0)

    ref_complex = reference_image.copy()
    acc_lib._fftshift_axis0(ref_complex)
    assert_allclose(npshifted, ref_complex, rtol=tol)


@pytest.mark.parametrize("real_type, complex_type, rtol, atol, acc_lib, pars",
                         [('float32', 'complex64',  1.e-7,  1e-5, g_single, par1),
                          ('float64', 'complex128', 1.e-16, 1e-13, g_double, par1),
                          ('float32', 'complex64',  1.e-3,  1e-5, g_single, par2),
                          ('float64', 'complex128', 1.e-16, 1e-13, g_double, par2),
                          ('float32', 'complex64',  1.e-7,  1e-5, g_single, par3),
                          ('float64', 'complex128', 1.e-16, 1e-13, g_double, par3)],
                         ids=["SP_par1", "DP_par1",
                              "SP_par2", "DP_par2",
                              "SP_par3", "DP_par3"])
def test_apply_phase_vis(real_type, complex_type, rtol, atol, acc_lib, pars):
    """
    Test apply phase to visibilities

    """
    dRA = pars.get('dRA', 0.4)
    dDec = pars.get('dDec', 10.)

    dRA *= arcsec
    dDec *= arcsec

    # generate the samples
    nsamples = 10000
    maxuv_generator = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)

    # generate mock visibility values
    fint = np.zeros(nsamples, dtype=complex_type)
    fint.real = np.random.random(nsamples) * 10.
    fint.imag = np.random.random(nsamples) * 30.

    fint_numpy = apply_phase_array(udat, vdat, fint.copy(), dRA, dDec)

    fint_shifted = acc_lib.apply_phase_vis(dRA, dDec, udat, vdat, fint)

    assert_allclose(fint_numpy.real, fint_shifted.real, rtol, atol)
    assert_allclose(fint_numpy.imag, fint_shifted.imag, rtol, atol)


@pytest.mark.parametrize("nsamples, real_type, tol, acc_lib",
                         [(1000, 'float32', 1.e-6, g_single),
                          (1000, 'float64', 1.e-15, g_double)],
                         ids=["SP", "DP"])
def test_reduce_chi2(nsamples, real_type, tol, acc_lib):
    """
    Test chi2 reduction

    """
    x, y, w = generate_random_vis(nsamples, real_type)
    chi2_ref = np.sum(((x.real - y.real) ** 2. + (x.imag - y.imag)**2.) * w)

    chi2_loc = acc_lib.reduce_chi2(x.real.copy(order='C'), x.imag.copy(order='C'), w, y.copy())

    assert_allclose(chi2_ref, chi2_loc, rtol=tol)


@pytest.mark.parametrize("nsamples, real_type, rtol, acc_lib",
                         [(1000, 'float32', 1.e-4, g_single),
                          (1000, 'float64', 1.e-10, g_double)],
                         ids=["SP", "DP"])
def test_image_origin(nsamples, real_type, rtol, acc_lib):
    def model1(R):
        y = (np.exp(-(R / (0.2 * arcsec)) ** 2) + 0.3 * np.exp(
            -((R - 0.4 * arcsec) / ((0.15 * arcsec))) ** 2))
        return 1e12 * y

    def model2(R):
        y = 1 * np.exp(
            -((R - 1 * arcsec) / (0.5 * arcsec)) ** 2) + 0.7 * np.exp(
            -((R - 2.5 * arcsec) / (0.25 * arcsec)) ** 2) + 0.2 * np.exp(
            -((R - 3.5 * arcsec) / (0.15 * arcsec)) ** 2)

        return 1e12 * y

    def model3(R):
        y = 1 * np.exp(-((R - 0.5 * arcsec) / ((0.1 * arcsec))) ** 2)
        return 1e12 * y

    def model4(R):
        y = 1 * (R / 2. / arcsec) ** -0.05 * np.exp(-(R / 2. / arcsec) ** 4)
        return 1e12 * y

    # u, v points
    maxuv_generator = 3e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator,
                                        dtype=real_type)
    nxy, dxy = 4096, 6.42956326721e-08

    # radial grid
    Rmin = 0.00001 * arcsec
    dR = 0.0001 * arcsec
    nrad = 2000
    gridrad = np.linspace(Rmin, Rmin + dR * (nrad - 1), nrad)

    # create sample image with origin='upper'
    image_asym = sweep_ref(model1(gridrad), Rmin, dR, nxy, nxy, dxy,  0  * deg, Dx=-50.*dxy,  Dy=66.*dxy,   dtype_image=real_type) + \
                 sweep_ref(model2(gridrad), Rmin, dR, nxy, nxy, dxy, 20. * deg, Dx=+150.*dxy, Dy=+250.*dxy, dtype_image=real_type) + \
                 sweep_ref(model3(gridrad), Rmin, dR, nxy, nxy, dxy, 35. * deg, Dx=-110.*dxy, Dy=-100.*dxy, dtype_image=real_type) + \
                 sweep_ref(model4(gridrad), Rmin, dR, nxy, nxy, dxy, 44. * deg, Dx=-110.*dxy, Dy=-100.*dxy, dtype_image=real_type)

    # create sample image with origin='lower'
    image_asym2 = sweep_ref(model1(gridrad), Rmin, dR, nxy, nxy, dxy,   0 * deg, Dx=-50.*dxy,  Dy=66.*dxy,   dtype_image=real_type, origin='lower') + \
                  sweep_ref(model2(gridrad), Rmin, dR, nxy, nxy, dxy, 20. * deg, Dx=+150.*dxy, Dy=+250.*dxy, dtype_image=real_type, origin='lower') + \
                  sweep_ref(model3(gridrad), Rmin, dR, nxy, nxy, dxy, 35. * deg, Dx=-110.*dxy, Dy=-100.*dxy, dtype_image=real_type, origin='lower') + \
                  sweep_ref(model4(gridrad), Rmin, dR, nxy, nxy, dxy, 44. * deg, Dx=-110.*dxy, Dy=-100.*dxy, dtype_image=real_type, origin='lower')

    # check that the images are flipped and rolled when diffent origin option is used
    assert_allclose(image_asym2, np.roll(np.flipud(image_asym), 1, 0), atol=0, rtol=rtol)

    # remove spurious values
    image_asym[0, :] = 0.
    image_asym[:, 0] = 0.
    image_asym[np.where(image_asym < 1e-10)] = 0.

    # remove spurious values
    image_asym2[0, :] = 0.
    image_asym2[:, 0] = 0.
    image_asym2[np.where(image_asym2 < 1e-10)] = 0.

    # Compute visibilities of ORIGINAL image with CURRENT GALARIO algorithm (only: origin='upper')
    vis_C_upper_image_upper = acc_lib.sampleImage(image_asym, dxy, udat, vdat, dRA=0.5, dDec=-3., PA=10.)

    # Compute visibilities of ORIGINAL image with NEW algorithm, origin='upper'
    vis_py_upper_image_upper = py_sampleImage(image_asym, dxy, udat, vdat, dRA=0.5, dDec=-3., PA=10., origin='upper')

    # Compute visibilities of LOWER ORIGIN image with NEW algorithm, origin='lower'
    vis_py_lower_image_lower = py_sampleImage(image_asym2, dxy, udat, vdat, dRA=0.5, dDec=-3., PA=10., origin='lower')

    # Compute with C implementation
    vis_C_lower_image_lower = acc_lib.sampleImage(image_asym2, dxy, udat, vdat, dRA=0.5, dDec=-3., PA=10., origin='lower')

    # check that they produce all the same visibilities
    assert_allclose(vis_py_upper_image_upper, vis_C_lower_image_lower, atol=0., rtol=rtol)
    assert_allclose(vis_py_upper_image_upper, vis_C_upper_image_upper, atol=0., rtol=rtol)
    assert_allclose(vis_py_lower_image_lower, vis_C_upper_image_upper, atol=0., rtol=rtol)
    assert_allclose(vis_C_lower_image_lower, vis_C_upper_image_upper, atol=0., rtol=rtol)



@pytest.mark.parametrize("nsamples, real_type, rtol, atol, acc_lib, pars",
                          [(int(1e3), 'float64', 1e-6, 0, g_double, par1),
                          (int(1e3), 'float64', 1e-6, 0, g_double, par2),
                          (int(1e3), 'float64', 1e-6, 0, g_double, par3),
                          (int(1e3), 'float64', 1e-6, 0, g_double, par4)],
                         ids=["{}".format(i) for i in range(4)])
def test_all(nsamples, real_type, rtol, atol, acc_lib, pars):
    """
    Main test function: tests Python vs galario implementation of sampleImage,
    sampleProfile, chi2Image, chi2Profile.
    It also cross-checks all the galario results among themselves.

    For the imaginary part the test has atol=np.abs(np.mean(vis_g_sampleImage.real))*rtol.
    The reason is that for symmetric images the imaginary part of FFT can fluctuate quite a lot
    this manual absolute tolerance checks that such fluctuations are small compared to the real part.

    """
    dRA = pars['dRA']
    dDec = pars['dDec']
    PA = pars['PA']
    nxy = pars['nxy']

    # generate the samples
    maxuv_generator = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)

    _, minuv, maxuv = matrix_size(udat, vdat)

    dxy = 1. / maxuv # pixel size (rad)
    # create intensity profile and model image
    Rmin, dR, nrad, inc, profile_mode, real_type = dxy/100., dxy/10.5, 10000, 20., 'Gauss', 'float64',
    dRA *= arcsec
    dDec *= arcsec
    PA *= deg
    inc *= deg

    intensity = radial_profile(Rmin, dR, nrad, profile_mode, dtype=real_type, gauss_width=dxy*10)
    reference_image = sweep_ref(intensity, Rmin, dR, nxy, nxy, dxy, inc, dtype_image=real_type)

    # test sampleImage
    vis_py_sampleImage = py_sampleImage(reference_image, dxy, udat, vdat, PA=PA, dRA=dRA, dDec=dDec)
    vis_g_sampleImage = acc_lib.sampleImage(reference_image, dxy, udat, vdat, PA=PA, dRA=dRA, dDec=dDec)

    assert_allclose(vis_py_sampleImage.real, vis_g_sampleImage.real, rtol=rtol, atol=atol)
    assert_allclose(vis_py_sampleImage.imag, vis_g_sampleImage.imag, rtol=rtol, atol=np.abs(np.mean(vis_g_sampleImage.real))*rtol)

    # test sampleProfile
    vis_py_sampleProfile = py_sampleProfile(intensity.copy(), Rmin, dR, nxy, dxy, udat, vdat, inc=inc, dRA=dRA, dDec=dDec, PA=PA)
    vis_g_sampleProfile = acc_lib.sampleProfile(intensity, Rmin, dR, nxy, dxy, udat, vdat, inc=inc, dRA=dRA, dDec=dDec, PA=PA)

    # check galario vs python implementation
    assert_allclose(vis_g_sampleProfile.real, vis_py_sampleProfile.real, rtol=rtol, atol=atol)
    assert_allclose(vis_g_sampleProfile.imag, vis_py_sampleProfile.imag, rtol=rtol, atol=np.abs(np.mean(vis_g_sampleProfile.real))*rtol)

    # cross-check galario sampleProfile vs sampleImage
    assert_allclose(vis_g_sampleImage.real, vis_g_sampleProfile.real, rtol=rtol, atol=atol)
    assert_allclose(vis_g_sampleImage.imag, vis_g_sampleProfile.imag, rtol=rtol, atol=np.abs(np.mean(vis_g_sampleProfile.real))*rtol)

    # test chi2Image
    x, _, w = generate_random_vis(nsamples, real_type)

    chi2_pychi2Image = py_chi2Image(reference_image, dxy, udat, vdat, x.real.copy(), x.imag.copy(), w, dRA=dRA, dDec=dDec)
    chi2_g_chi2Image = acc_lib.chi2Image(reference_image, dxy, udat, vdat, x.real.copy(), x.imag.copy(), w, dRA=dRA, dDec=dDec)

    # test chi2Profile
    chi2_pychi2Profile = py_chi2Profile(intensity, Rmin, dR, nxy, dxy, udat, vdat, x.real.copy(), x.imag.copy(), w, inc=inc, dRA=dRA, dDec=dDec)
    chi2_g_chi2Profile = acc_lib.chi2Profile(intensity, Rmin, dR, nxy, dxy, udat, vdat, x.real.copy(), x.imag.copy(), w, inc=inc, dRA=dRA, dDec=dDec)

    # check galario vs python implementation
    assert_allclose(chi2_pychi2Profile, chi2_g_chi2Profile, rtol=rtol, atol=atol)
    assert_allclose(chi2_pychi2Image, chi2_g_chi2Image, rtol=rtol, atol=atol)

    # cross-check galario chi2Profile vs chi2Image
    assert_allclose(chi2_g_chi2Profile, chi2_g_chi2Image, rtol=rtol, atol=atol)



# huge inaccuracy in single precision for larger images
@pytest.mark.parametrize("nsamples, real_type, complex_type, rtol, atol, acc_lib, pars",
                         [(100, 'float32', 'complex64',  1e-4,  1e-3, g_single, par1),
                          (1000, 'float64', 'complex128', 1e-14, 1e-10, g_double, par1)],
                         ids=["SP_par1", "DP_par1"])
def test_loss(nsamples, real_type, complex_type, rtol, atol, acc_lib, pars):
    # try to find out where precision is lost

    dRA = pars.get('dRA', 0.4)
    dDec = pars.get('dDec', 10.)

    # generate the samples
    maxuv_generator = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)

    # compute the matrix size and maxuv
    size, minuv, maxuv = matrix_size(udat, vdat)

    # create model complex image (it happens to have 0 imaginary part)
    reference_image = create_reference_image(size=size, dtype=real_type)

    ###
    # shift real
    ###
    py_shift_real = np.fft.fftshift(reference_image)
    acc_shift_real = acc_lib._fftshift(reference_image)

    # interpret complex array as real and skip last two columns
    real_view = acc_shift_real.view(dtype=real_type)[:, :-2]

    # shifting the values should make no difference, so ask for high precision
    assert_allclose(py_shift_real, real_view, rtol=1e-15, atol=1e-15)

    ###
    # FFT
    ###
    py_fft = np.fft.fft2(py_shift_real)
    # use the real input!
    acc_fft = acc_lib._fft2d(py_shift_real)

    assert_allclose(unique_part(py_fft).real, acc_fft.real, rtol, atol)
    assert_allclose(unique_part(py_fft).imag, acc_fft.imag, rtol, atol)

    ###
    # shift complex
    ###
    py_shift_cmplx = np.fft.fftshift(py_fft, axes=0)
    acc_lib._fftshift_axis0(acc_fft)
    assert_allclose(unique_part(py_shift_cmplx).real, acc_fft.real, rtol, atol)
    assert_allclose(unique_part(py_shift_cmplx).imag, acc_fft.imag, rtol, atol)

    ###
    # phase
    ###
    du = maxuv/size
    uroti, vroti = uv_idx(udat, vdat, du, size/2.)
    ReInt = int_bilin_MT(py_shift_cmplx.real, uroti, vroti).astype(real_type)
    ImInt = int_bilin_MT(py_shift_cmplx.imag, uroti, vroti).astype(real_type)
    fint = ReInt + 1j*ImInt
    fint_acc = fint.copy()
    fint_shifted = apply_phase_array(udat, vdat, fint, dRA, dDec)
    fint_acc_shifted = acc_lib.apply_phase_vis(dRA, dDec, udat, vdat, fint_acc)


    # lose some absolute precision here  --> not anymore. Really? check by decreasing rtol, atol
    # atol *= 2
    assert_allclose(fint_shifted.real, fint_acc_shifted.real, rtol, atol)
    assert_allclose(fint_shifted.imag, fint_acc_shifted.imag, rtol, atol)
    # but continue with previous tolerance
    # atol /= 2

    ###
    # interpolation
    ###
    uroti, vroti = uv_idx_r2c(udat, vdat, du, size/2.)
    ReInt = int_bilin_MT(py_shift_cmplx.real, uroti, vroti).astype(real_type)
    ImInt = int_bilin_MT(py_shift_cmplx.imag, uroti, vroti).astype(real_type)
    uneg = udat < 0.
    ImInt[uneg] *= -1.

    complexInt = acc_lib.interpolate(py_shift_cmplx.astype(complex_type),
                                     du,
                                     udat.astype(real_type),
                                     vdat.astype(real_type))

    assert_allclose(ReInt, complexInt.real, rtol, atol)
    assert_allclose(ImInt, complexInt.imag, rtol, atol)

    ###
    # now all steps in one function
    # -> MT removed this because there is already a test for sample and here it is not clear what is the reference.
    ###
    # sampled = acc_lib.sampleImage(ref_real, dRA, dDec, du, udat, vdat)
    #
    # # a lot of precision lost. Why? --> not anymore
    # # rtol = 1
    # # atol = 0.5
    # assert_allclose(fint_shifted.real, sampled.real, rtol, atol)
    # assert_allclose(fint_shifted.imag, sampled.imag, rtol, atol)

def test_exception():
    """
    Make sure exceptions propagate from C++ to python
    """
    with pytest.raises(ValueError, message="Image can't be too small"):
        g_double._fft2d(np.ones((1, 1), dtype=np.float64))

    with pytest.raises(ValueError, message="Unequal image lengths are not permitted"):
        g_double._fft2d(np.ones((10, 9), dtype=np.float64))

    with pytest.raises(ValueError, message="Odd image lengths are not permitted"):
        g_double._fft2d(np.ones((9, 9), dtype=np.float64))


@pytest.mark.parametrize("nxy, inc, dxy, Dx, Dy, real_type, tol, acc_lib",
                         [(1000, 20., 2e-3, -2., 0., 'float32', 1.e-6, g_single),
                          (1000, 33.4, 1e-8, 0.23, -1.23, 'float64', 1.e-15, g_double)],
                         ids=["SP", "DP"])
def test_get_coords_meshgrid(nxy, inc, dxy, Dx, Dy, real_type, tol, acc_lib):

    ncol, nrow = nxy, nxy

    # create the referencemesh grid
    inc_cos = np.cos(inc)
    x = (np.linspace(0.5, -0.5 + 1./float(ncol), ncol, dtype=real_type)) * dxy * ncol
    y = (np.linspace(0.5, -0.5 + 1./float(nrow), nrow, dtype=real_type)) * dxy * nrow

    # we shrink the x axis, since PA is the angle East of North of the
    # the plane of the disk (orthogonal to the angular momentum axis)
    # PA=0 is a disk with vertical orbital node (aligned along North-South)
    x_m, y_m = np.meshgrid((x - Dx)/ inc_cos, y - Dy)
    R_m = np.sqrt(x_m ** 2. + y_m ** 2.)

    x_test, y_test, x_m_test, y_m_test, R_m_test = acc_lib.get_coords_meshgrid(nrow, ncol, dxy, inc, Dx=Dx, Dy=Dy, origin='upper')

    assert_allclose(x, x_test, atol=0, rtol=tol)
    assert_allclose(y, y_test, atol=0, rtol=tol)
    assert_allclose(x_m, x_m_test, atol=0, rtol=tol)
    assert_allclose(y_m, y_m_test, atol=0, rtol=tol)
    assert_allclose(R_m, R_m_test, atol=0, rtol=tol)

