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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import, unicode_literals)

import numpy as np

from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.integrate import trapezoid, quadrature

__all__ = ["py_sampleImage", "py_sampleProfile", "py_chi2Profile", "py_chi2Image",
           "radial_profile", "g_sweep_prototype", "sweep_ref",
           "create_reference_image", "create_sampling_points", "uv_idx",
           "uv_idx_r2c", "int_bilin_MT", "matrix_size",
           "apply_phase_array", "generate_random_vis",
           "unique_part", "assert_allclose", "apply_rotation"]


def py_sampleImage(reference_image, dxy, udat, vdat, dRA=0., dDec=0., PA=0., origin='upper'):
    """
    Python implementation of sampleImage.

    """
    if origin == 'upper':
        v_origin = 1.
    elif origin == 'lower':
        v_origin = -1.

    nxy = reference_image.shape[0]

    dRA *= 2.*np.pi
    dDec *= 2.*np.pi

    du = 1. / (nxy*dxy)

    # Real to Complex transform
    fft_r2c_shifted = np.fft.fftshift(
                        np.fft.rfft2(
                            np.fft.fftshift(reference_image)), axes=0)

    # apply rotation
    cos_PA = np.cos(PA)
    sin_PA = np.sin(PA)

    urot = udat * cos_PA - vdat * sin_PA
    vrot = udat * sin_PA + vdat * cos_PA

    dRArot = dRA * cos_PA - dDec * sin_PA
    dDecrot = dRA * sin_PA + dDec * cos_PA

    # interpolation indices
    uroti = np.abs(urot)/du
    vroti = nxy/2. + v_origin * vrot/du
    uneg = urot < 0.
    vroti[uneg] = nxy/2 - v_origin * vrot[uneg]/du

    # coordinates of FT
    u_axis = np.linspace(0., nxy // 2, nxy // 2 + 1)
    v_axis = np.linspace(0., nxy - 1, nxy)

    # We use RectBivariateSpline to do only linear interpolation, which is faster
    # than interp2d for our case of a regular grid.
    # RectBivariateSpline does not work for complex input, so we need to run it twice.
    f_re = RectBivariateSpline(v_axis, u_axis, fft_r2c_shifted.real, kx=1, ky=1, s=0)
    ReInt = f_re.ev(vroti, uroti)
    f_im = RectBivariateSpline(v_axis, u_axis, fft_r2c_shifted.imag, kx=1, ky=1, s=0)
    ImInt = f_im.ev(vroti, uroti)
    f_amp = RectBivariateSpline(v_axis, u_axis, np.abs(fft_r2c_shifted), kx=1, ky=1, s=0)
    AmpInt = f_amp.ev(vroti, uroti)

    # correct for Real to Complex frequency mapping
    uneg = urot < 0.
    ImInt[uneg] *= -1.
    PhaseInt = np.angle(ReInt + 1j*ImInt)

    # apply the phase change
    theta = urot*dRArot + vrot*dDecrot
    vis = AmpInt * (np.cos(theta+PhaseInt) + 1j*np.sin(theta+PhaseInt))

    return vis


def py_sampleProfile(intensity, Rmin, dR, nxy, dxy, udat, vdat, dRA=0., dDec=0., PA=0, inc=0.):
    """
    Python implementation of sampleProfile.

    """
    inc_cos = np.cos(inc)

    nrad = len(intensity)
    gridrad = np.linspace(Rmin, Rmin + dR * (nrad - 1), nrad)

    ncol, nrow = nxy, nxy
    # create the mesh grid
    x = (np.linspace(0.5, -0.5 + 1./float(ncol), ncol)) * dxy * ncol
    y = (np.linspace(0.5, -0.5 + 1./float(nrow), nrow)) * dxy * nrow

    # we shrink the x axis, since PA is the angle East of North of the
    # the plane of the disk (orthogonal to the angular momentum axis)
    # PA=0 is a disk with vertical orbital node (aligned along North-South)
    x_axis, y_axis = np.meshgrid(x / inc_cos, y)
    x_meshgrid = np.sqrt(x_axis ** 2. + y_axis ** 2.)

    # convert to Jansky
    sr_to_px = dxy**2.
    intensity *= sr_to_px
    f = interp1d(gridrad, intensity, kind='linear', fill_value=0.,
                 bounds_error=False, assume_sorted=True)
    intensmap = f(x_meshgrid)

    intensmap[nrow//2, ncol//2] = central_pixel(intensity, Rmin, dR, dxy)

    vis = py_sampleImage(intensmap, dxy, udat, vdat, PA=PA, dRA=dRA, dDec=dDec)

    return vis


def py_chi2Image(reference_image, dxy, udat, vdat, vis_obs_re, vis_obs_im, weights, dRA=0., dDec=0., PA=0.):
    """
    Python implementation of chi2Image.

    """
    vis = py_sampleImage(reference_image, dxy, udat, vdat, PA=PA, dRA=dRA, dDec=dDec)

    chi2 = np.sum(((vis.real - vis_obs_re)**2. + (vis.imag - vis_obs_im)**2.)*weights)

    return chi2


def py_chi2Profile(intensity, Rmin, dR, nxy, dxy, udat, vdat, vis_obs_re, vis_obs_im, weights, dRA=0., dDec=0., PA=0, inc=0.):
    """
    Python implementation of chi2Profile.

    """
    vis = py_sampleProfile(intensity, Rmin, dR, nxy, dxy, udat, vdat, inc=inc, PA=PA, dRA=dRA, dDec=dDec)

    chi2 = np.sum(((vis.real - vis_obs_re)**2. + (vis.imag - vis_obs_im)**2.)*weights)

    return chi2


def radial_profile(Rmin, delta_R, nrad, mode='Gauss', dtype='float64', gauss_width=100):
    """ Compute a radial brightness profile. Returns intensity in Jy/sr """
    gridrad = np.linspace(Rmin, Rmin + delta_R * (nrad - 1), nrad).astype(dtype)

    if mode == 'Gauss':
        # a simple Gaussian
        intensity = np.exp(-(gridrad/gauss_width)**2)
    elif mode == 'Cos-Gauss':
        # a cos-tapered Gaussian
        intensity = np.cos(2.*np.pi*gridrad/(gauss_width))**2. * np.exp(-(gridrad/gauss_width)**2)

    return intensity


def central_pixel(I, Rmin, dR, dxy):
    """
    Compute brightness in the central pixel as the average flux in the pixel.

    """
    # with quadrature method: tends to over-estimate it
    # area = np.pi*((dxy/2.)**2-Rmin**2)
    # flux, _ = quadrature(lambda z: f(z)*z, Rmin, dxy/2., tol=1.49e-25, maxiter=200)
    # flux *= 2.*np.pi
    # intensmap[int(nrow/2+Dy/dxy), int(ncol/2-Dx/dxy)] = flux/area

    # with trapezoidal rule: it's the same implementation as in galario.cpp
    iIN = int(np.floor((dxy / 2 - Rmin) // dR))
    flux = 0.
    for i in range(1, iIN):
        flux += (Rmin + dR * i) * I[i]

    flux *= 2.
    flux += Rmin * I[0] + (Rmin + iIN * dR) * I[iIN]
    flux *= dR

    # add flux between Rmin+iIN*dR and dxy/2
    I_interp = (I[iIN + 1] - I[iIN]) / (dR) * (dxy / 2. - (Rmin + dR * (iIN))) + \
               I[iIN]  # brightness at R=dxy/2
    flux += ((Rmin + iIN * dR) * I[iIN] + dxy / 2. * I_interp) * (
                dxy / 2. - (Rmin + iIN * dR))

    # flux *= 2 * np.pi / 2.  # to complete trapezoidal rule (***)
    area = (dxy / 2.) ** 2 - Rmin ** 2
    # area *= np.pi  # elides (***)

    return flux / area


def g_sweep_prototype(I, Rmin, dR, nrow, ncol, dxy, inc, dtype_image='float64'):
    """ Prototype of the sweep function for galario. """
    assert Rmin <= dxy, "Rmin must be smaller or equal than dxy"
    image = np.zeros((nrow, ncol), dtype=dtype_image)

    nrad = len(I)
    irow_center = nrow // 2
    icol_center = ncol // 2
    inc_cos = np.cos(inc)

    # radial extent in number of image pixels covered by the profile
    rmax = min(int(np.ceil((Rmin+nrad*dR)/dxy)), irow_center)
    row_offset = irow_center-rmax
    col_offset = icol_center-rmax
    for irow in range(rmax*2):
        for jcol in range(rmax*2):
            x = (rmax - jcol) * dxy
            y = (rmax - irow) * dxy
            rr = np.sqrt((x/inc_cos)**2. + (y)**2.)

            # interpolate 1D
            iR = int(np.floor((rr-Rmin) / dR))
            if iR >= nrad-1:
                image[irow+row_offset, jcol+col_offset] = 0.
            else:
                image[irow+row_offset, jcol+col_offset] = I[iR] + (rr - iR * dR - Rmin) * (I[iR + 1] - I[iR]) / dR

    # central pixel
    image[irow_center, icol_center] = central_pixel(I, Rmin, dR, dxy)

    sr_to_px = dxy**2.
    image *= sr_to_px

    return image


def sweep_ref(I, Rmin, dR, nrow, ncol, dxy, inc, Dx=0., Dy=0., dtype_image='float64', origin='upper'):
    """
    Compute the intensity map (i.e. the image) given the radial profile I(R).
    We assume an axisymmetric profile.
    The origin of the output image is in the upper left corner.

    Parameters
    ----------
    I: 1D float array
        Intensity radial profile I(R).
    Rmin : float
        Inner edge of the radial grid. At R=Rmin the intensity is intensity[0].
        For R<Rmin the intensity is assumed to be 0.
        **units**: rad
    dR : float
        Size of the cell of the radial grid, assumed linear.
        **units**: rad
    nrow : int
        Number of rows of the output image.
        **units**: pixel
    ncol : int
        Number of columns of the output image.
        **units**: pixel
    dxy : float
        Size of the image cell, assumed equal and uniform in both x and y direction.
        **units**: rad
    inc : float
        Inclination along North-South axis.
        **units**: rad
    Dx : optional, float
        Right Ascension offset (positive towards East, left).
        **units**: rad
    Dy : optional, float
        Declination offset (positive towards North, top).
        **units**: rad
    dtype_image : optional, str
        numpy dtype specification for the output image.
    origin: ['upper' | 'lower'], optional, default: 'upper'
        Set the [0,0] index of the array in the upper left or lower left corner of the axes.

    Returns
    -------
    intensmap: 2D float array
        The intensity map, sweeped by 2pi.

    """
    if origin == 'upper':
        v_origin = 1.
    elif origin == 'lower':
        v_origin = -1.

    inc_cos = np.cos(inc)

    nrad = len(I)
    gridrad = np.linspace(Rmin, Rmin + dR * (nrad - 1), nrad)

    # create the mesh grid
    x = (np.linspace(0.5, -0.5 + 1./float(ncol), ncol)) * dxy * ncol
    y = (np.linspace(0.5, -0.5 + 1./float(nrow), nrow)) * dxy * nrow * v_origin

    # we shrink the x axis, since PA is the angle East of North of the
    # the plane of the disk (orthogonal to the angular momentum axis)
    # PA=0 is a disk with vertical orbital node (aligned along North-South)
    xxx, yyy = np.meshgrid((x - Dx) / inc_cos, (y - Dy))
    x_meshgrid = np.sqrt(xxx ** 2. + yyy ** 2.)

    f = interp1d(gridrad, I, kind='linear', fill_value=0.,
                 bounds_error=False, assume_sorted=True)
    intensmap = f(x_meshgrid)

    # central pixel: compute the average brightness
    intensmap[int(nrow / 2 + Dy / dxy * v_origin), int(ncol / 2 - Dx / dxy)] = central_pixel(I, Rmin, dR, dxy)

    # convert to Jansky
    intensmap *= dxy**2.

    return intensmap.astype(dtype_image)


def create_reference_image(size, x0=10., y0=-3., sigma_x=50., sigma_y=30., dtype='float64',
                           reverse_xaxis=False, correct_axes=True, sizey=None, **kwargs):
    """
    Creates a reference image: a gaussian intensity with elliptical
    """
    inc_cos = np.cos(0./180.*np.pi)

    delta_x = 1.
    x = (np.linspace(0., size - 1, size) - size / 2.) * delta_x

    if sizey:
        y = (np.linspace(0., sizey-1, sizey) - sizey/2.) * delta_x
    else:
        y = x.copy()

    if reverse_xaxis:
        xx, yy = np.meshgrid(-x, y/inc_cos)
    elif correct_axes:
        xx, yy = np.meshgrid(-x, -y/inc_cos)
    else:
        xx, yy = np.meshgrid(x, y/inc_cos)

    image = np.exp(-(xx-x0)**2./sigma_x - (yy-y0)**2./sigma_y)

    return image.astype(dtype)


def create_sampling_points(nsamples, maxuv=1., dtype='float64'):
    # TODO make this generator smarter
    assert isinstance(nsamples, int)

    minuv = maxuv/100.  # change to 10000 to have nxy=4096
    np.random.seed(42)
    # columns are non contiguous arrays => copy
    uvdist = np.random.uniform(low=minuv, high=maxuv, size=nsamples)
    phi = np.random.uniform(low=0., high=2.*np.pi, size=nsamples)

    u = uvdist * np.cos(phi)
    v = uvdist * np.sin(phi)

    return u.astype(dtype), v.astype(dtype)


def uv_idx(udat, vdat, du, half_size):
    """
    For C2C transform.
    uv coordinates to pixel coordinates in range [0, npixels].
    Assume image is square, same boundary in u and v direction.
    """
    return half_size + udat/du, half_size + vdat/du


def uv_idx_r2c(udat, vdat, du, half_size):
    """
    For R2C transform.
    uv coordinates to pixel coordinates in range [0, npixels].
    Assume image is square, same boundary in u and v direction.
    """
    indu = np.abs(udat) / du
    indv = half_size + vdat / du
    uneg = udat < 0.
    indv[uneg] = half_size - vdat[uneg] / du

    return indu, indv


def int_bilin_MT(f, x, y):
    # assume x, y are in pixel
    vis_int = np.zeros(len(x))

    for i in range(len(x)):
        t = y[i] - np.floor(y[i])
        u = x[i] - np.floor(x[i])
        y0 = f[int(np.floor(y[i])), int(np.floor(x[i]))]
        y1 = f[int(np.floor(y[i])) + 1, int(np.floor(x[i]))]
        y2 = f[int(np.floor(y[i])) + 1, int(np.floor(x[i])) + 1]
        y3 = f[int(np.floor(y[i])), int(np.floor(x[i])) + 1]

        vis_int[i] = t * u * (y0 - y1 + y2 - y3)
        vis_int[i] += t * (y1 - y0)
        vis_int[i] += u * (y3 - y0)
        vis_int[i] += y0

    return vis_int


def matrix_size(udat, vdat, **kwargs):

    maxuv_factor = kwargs.get('maxuv_factor', 4.8)
    minuv_factor = kwargs.get('minuv_factor', 4.)

    uvdist = np.sqrt(udat**2 + vdat**2)

    maxuv = max(uvdist)*maxuv_factor
    minuv = min(uvdist)/minuv_factor

    minpix = np.uint(maxuv/minuv)

    Nuv = kwargs.get('force_nx', int(2**np.ceil(np.log2(minpix))))

    return Nuv, minuv, maxuv


def apply_phase_array(u, v, vis_int, x0, y0):
    """
    Performs a translation in the real space by applying a phase shift in the Fourier space.
    This function applies the shift to data points sampling the Fourier transform of an image.

    Parameters
    ----------
    u, v: 1D float array
        Coordinates of points in the Fourier space. units: observing wavelength
    vis_int: 1D float array, complex
        Fourier Transform sampled in the (u, v) points.
        Re, Im, u, v must have the same length.
    x0, y0: floats, rad
        Shifts in the real space.

    Returns
    -------
    vis_int_shifted: 1D float array, complex
        Phase-shifted of the Fourier Transform sampled in the (u, v) points.

    """
    x0 *= 2.*np.pi
    y0 *= 2.*np.pi

    # construct the phase change
    theta = u*x0 + v*y0

    # apply the phase change
    vis_int_shifted = vis_int * (np.cos(theta) + 1j*np.sin(theta))

    return vis_int_shifted


def generate_random_vis(nsamples, dtype):
    x = 3. * np.random.uniform(low=0., high=1., size=nsamples).astype(dtype) + 2.8 +\
        1j * np.random.uniform(low=0., high=1., size=nsamples).astype(dtype) + 8.2
    y = 8. * np.random.uniform(low=0.5, high=3., size=nsamples).astype(dtype) + 5.7 +\
        1j * np.random.uniform(low=0., high=6., size=nsamples).astype(dtype) + 21.2

    w = np.random.uniform(low=0., high=1e4, size=nsamples).astype(dtype)
    w /= w.sum()

    return x, y, w


def apply_rotation(PA, dRA, dDec, udat, vdat):
    """ Rotates the RA, Dec offsets and the udat and vdat coordinates by Position Angle PA """
    # PA: rad

    cos_PA = np.cos(PA)
    sin_PA = np.sin(PA)

    urot = udat * cos_PA - vdat * sin_PA
    vrot = udat * sin_PA + vdat * cos_PA

    dRArot = dRA * cos_PA - dDec * sin_PA
    dDecrot = dRA * sin_PA + dDec * cos_PA

    return dRArot, dDecrot, urot, vrot


def unique_part(array):
    """Extract the unique part of a real-to-complex Fourier transform"""
    return array[:, 0:int(array.shape[1]/2)+1]


def assert_allclose(x, y, rtol=1e-10, atol=1e-8):
    """Drop in replacement for `numpy.testing.assert_allclose` that shows the nonmatching elements"""
    if np.isscalar(x) and np.isscalar(y) == 1:
        return np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)

    if x.shape != y.shape:
        raise AssertionError("Shape mismatch: %s vs %s" % (str(x.shape), str(y.shape)))

    d = ~np.isclose(x, y, rtol, atol)
    if np.any(d):
        miss = np.where(d)[0]
        raise AssertionError("""Mismatch of %d elements (%g %%) at the level of rtol=%g, atol=%g
    %s
    %s
    %s""" % (len(miss), len(miss)/x.size, rtol, atol, repr(miss), str(x[d]), str(y[d])))
