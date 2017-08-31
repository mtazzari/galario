#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import, unicode_literals)

import numpy as np
import pyfftw

from scipy.interpolate import interp1d, RectBivariateSpline
from galario import arcsec, pc, au, deg

__all__ = ["py_sampleImage", "py_sampleProfile", "py_chi2Profile", "py_chi2Image",
           "radial_profile", "g_sweep_prototype", "sweep_ref",
           "create_reference_image", "create_sampling_points", "uv_idx",
           "uv_idx_r2c", "int_bilin_MT", "matrix_size",
           "apply_phase_array", "generate_random_vis",
           "unique_part", "assert_allclose", "apply_rotation"]


def py_sampleImage(reference_image, dxy, dist, udat, vdat, PA=0., dRA=0., dDec=0.):
    """
    Python implementation of sampleImage.

    """
    nxy = reference_image.shape[0]

    PA *= deg
    dRA *= 2.*np.pi * arcsec
    dDec *= 2.*np.pi * arcsec
    du = dist / nxy / dxy

    # Real to Complex transform
    fft_r2c_shifted = np.fft.fftshift(
                        pyfftw.interfaces.numpy_fft.rfft2(
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
    vroti = nxy/2. + vrot/du
    uneg = urot < 0.
    vroti[uneg] = nxy/2 - vrot[uneg]/du

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

    # correct for Real to Complex frequency mapping
    uneg = urot < 0.
    ImInt[uneg] *= -1.

    # apply the phase change
    theta = urot*dRArot + vrot*dDecrot
    vis = (ReInt + 1j*ImInt) * (np.cos(theta) + 1j*np.sin(theta))

    return vis


def py_sampleProfile(intensity, Rmin, dR, nxy, dxy, dist, udat, vdat, inc=0., PA=0, dRA=0., dDec=0.):
    """
    Python implementation of sampleProfile.

    """
    inc *= deg
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
    sr_to_px = (dxy/dist)**2.
    intensity *= sr_to_px
    f = interp1d(gridrad, intensity, kind='linear', fill_value=0.,
                 bounds_error=False, assume_sorted=True)
    intensmap = f(x_meshgrid)

    f_center = interp1d(gridrad, intensity, kind='linear', fill_value='extrapolate',
                 bounds_error=False, assume_sorted=True)
    intensmap[int(nrow/2), int(ncol/2)] = f_center(0.)

    vis = py_sampleImage(intensmap, dxy, dist, udat, vdat, PA=PA, dRA=dRA, dDec=dDec)

    return vis


def py_chi2Image(reference_image, dxy, dist, udat, vdat, vis_obs_re, vis_obs_im, weights, PA=0., dRA=0., dDec=0.):
    """
    Python implementation of chi2Image.

    """
    vis = py_sampleImage(reference_image, dxy, dist, udat, vdat, PA=PA, dRA=dRA, dDec=dDec)

    chi2 = np.sum(((vis.real - vis_obs_re)**2. + (vis.imag - vis_obs_im)**2.)*weights)

    return chi2



def py_chi2Profile(intensity, Rmin, dR, nxy, dxy, dist, udat, vdat, vis_obs_re, vis_obs_im, weights, inc=0., PA=0, dRA=0., dDec=0.):
    """
    Python implementation of chi2Profile.

    """
    vis = py_sampleProfile(intensity, Rmin, dR, nxy, dxy, dist, udat, vdat, inc=inc, PA=PA, dRA=dRA, dDec=dDec)

    chi2 = np.sum(((vis.real - vis_obs_re)**2. + (vis.imag - vis_obs_im)**2.)*weights)

    return chi2

def radial_profile(Rmin, delta_R, nrad, mode='Gauss', dtype='float64', gauss_width=100):
    """ Compute a radial brightness profile. Returns int in Jy/sr """
    gridrad = np.linspace(Rmin, Rmin + delta_R * (nrad - 1), nrad).astype(dtype)

    if mode == 'Gauss':
        # a simple Gaussian
        ints = np.exp(-(gridrad/delta_R/gauss_width)**2)
    elif mode == 'Cos-Gauss':
        # a cos-tapered Gaussian
        ints = np.cos(2.*np.pi*gridrad/(50.*delta_R))**2. * np.exp(-(gridrad/delta_R/80)**2)

    return ints


def g_sweep_prototype(I, Rmin, dR, nrow, ncol, dxy, dist, inc, dtype_image='float64'):
    """ Prototype of the sweep function for galario. """
    assert Rmin <= dxy, "Rmin must be smaller or equal than dxy"
    image = np.zeros((nrow, ncol), dtype=dtype_image)

    nrad = len(I)
    irow_center = int(nrow / 2)
    icol_center = int(ncol / 2)
    inc_cos = np.cos(inc/180.*np.pi)

    # radial extent in number of image pixels covered by the profile
    rmax = min(np.int(np.ceil((Rmin+nrad*dR)/dxy)), irow_center)
    row_offset = irow_center-rmax
    col_offset = icol_center-rmax
    for irow in range(rmax*2):
        for jcol in range(rmax*2):
            x = (rmax - jcol) * dxy
            y = (rmax - irow) * dxy
            rr = np.sqrt((x/inc_cos)**2. + (y)**2.)

            # interpolate 1D
            iR = np.int(np.floor((rr-Rmin) / dR))
            if iR >= nrad-1:
                image[irow+row_offset, jcol+col_offset] = 0.
            else:
                image[irow+row_offset, jcol+col_offset] = I[iR] + (rr - iR * dR - Rmin) * (I[iR + 1] - I[iR]) / dR

    # central pixel
    if Rmin != 0.:
        image[irow_center, icol_center] = I[0] + Rmin * (I[0] - I[1]) / dR

    sr_to_px = (dxy/dist)**2.
    image *= sr_to_px

    return image


def sweep_ref(I, Rmin, dR, nrow, ncol, dxy, dist, inc, Dx=0., Dy=0., dtype_image='float64'):
    """
    Compute the intensity map (i.e. the image) given the radial profile I(R)=ints.
    We assume an axisymmetric profile.

    Parameters
    ----------
    I: 1D float array
        Intensity radial profile I(R).
    gridrad: array
        Radial grid
    inc: float
        Inclination, degree
    Returns
    -------
    intensmap: 2D float array
        Image of the disk, i.e. the intensity map.

    """
    inc = inc/180.*np.pi
    inc_cos = np.cos(inc)

    nrad = len(I)
    gridrad = np.linspace(Rmin, Rmin + dR * (nrad - 1), nrad)

    # create the mesh grid
    x = (np.linspace(0.5, -0.5 + 1./float(ncol), ncol)) * dxy * ncol
    y = (np.linspace(0.5, -0.5 + 1./float(nrow), nrow)) * dxy * nrow

    # we shrink the x axis, since PA is the angle East of North of the
    # the plane of the disk (orthogonal to the angular momentum axis)
    # PA=0 is a disk with vertical orbital node (aligned along North-South)
    xxx, yyy = np.meshgrid((x - Dx * dxy) / inc_cos,
                           (y - Dy * dxy))
    x_meshgrid = np.sqrt(xxx ** 2. + yyy ** 2.)

    f = interp1d(gridrad, I, kind='linear', fill_value=0.,
                 bounds_error=False, assume_sorted=True)
    intensmap = f(x_meshgrid)

    f_center = interp1d(gridrad, I, kind='linear', fill_value='extrapolate',
                 bounds_error=False, assume_sorted=True)
    intensmap[int(nrow/2), int(ncol/2)] = f_center(0.)

    # convert to Jansky
    sr_to_px = (dxy/dist)**2.
    intensmap *= sr_to_px

    return intensmap.astype(dtype_image)


def create_reference_image(size, x0=10., y0=-3., sigma_x=50., sigma_y=30., dtype='float64',
                           reverse_xaxis=False, correct_axes=True, sizey=None, **kwargs):
    """
    Creates a reference image: a gaussian brightness with elliptical
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
    fint = np.zeros(len(x))

    for i in range(len(x)):
        t = y[i] - np.floor(y[i])
        u = x[i] - np.floor(x[i])
        y0 = f[np.int(np.floor(y[i])), np.int(np.floor(x[i]))]
        y1 = f[np.int(np.floor(y[i])) + 1, np.int(np.floor(x[i]))]
        y2 = f[np.int(np.floor(y[i])) + 1, np.int(np.floor(x[i])) + 1]
        y3 = f[np.int(np.floor(y[i])), np.int(np.floor(x[i])) + 1]

        fint[i] = t * u * (y0 - y1 + y2 - y3)
        fint[i] += t * (y1 - y0)
        fint[i] += u * (y3 - y0)
        fint[i] += y0

    return fint


def matrix_size(udat, vdat, **kwargs):

    maxuv_factor = kwargs.get('maxuv_factor', 4.8)
    minuv_factor = kwargs.get('minuv_factor', 4.)

    uvdist = np.sqrt(udat**2 + vdat**2)

    maxuv = max(uvdist)*maxuv_factor
    minuv = min(uvdist)/minuv_factor

    minpix = np.uint(maxuv/minuv)

    Nuv = kwargs.get('force_nx', int(2**np.ceil(np.log2(minpix))))

    return Nuv, minuv, maxuv


def apply_phase_array(u, v, fint, x0, y0):
    """
    Performs a translation in the real space by applying a phase shift in the Fourier space.
    This function applies the shift to data points sampling the Fourier transform of an image.

    Parameters
    ----------
    u, v: 1D float array
        Coordinates of points in the Fourier space. units: observing wavelength
    fint: 1D float array, complex
        Fourier Transform sampled in the (u, v) points.
        Re, Im, u, v must have the same length.
    x0, y0: floats, arcsec
        Shifts in the real space.

    Returns
    -------
    fint_shifted: 1D float array, complex
        Phase-shifted of the Fourier Transform sampled in the (u, v) points.

    """
    # convert x0, y0 from arcsec to cm
    x0 *= arcsec
    y0 *= arcsec

    x0 *= 2.*np.pi
    y0 *= 2.*np.pi

    # construct the phase change
    theta = u*x0 + v*y0

    # apply the phase change
    fint_shifted = fint * (np.cos(theta) + 1j*np.sin(theta))

    return fint_shifted


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
    # PA: deg

    PA = PA / 180. * np.pi
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
