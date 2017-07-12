#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import, unicode_literals)

import numpy as np

__all__ = ["create_reference_image", "create_sampling_points", "uv_idx", 
           "pixel_coordinates", "get_uv_idx_n",
           "int_bilin", "matrix_size", "Fourier_shift_static", 
           "Fourier_shift_array", "generate_random_vis", 
           "sec2rad"]

sec2rad = np.pi/180./3600.  # from arcsec to radians

def create_reference_image(size, x0=10., y0=-3., sigma_x=50., sigma_y=30., dtype='float64', reverse_xaxis=False, correct_axes=True, **kwargs):
    """
    Creates a reference image: a gaussian brightness with elliptical
    """
    _ = kwargs.get('kernel', 0.)  # legacy: muted
    _ = kwargs.get('save', 0.)  # legacy: muted

    inc_cos = np.cos(0./180.*np.pi)

    delta_x = 1.
    x = (np.linspace(0., size-1, size) - size/2.) * delta_x


    if reverse_xaxis:
        xx, yy = np.meshgrid(-x, x/inc_cos)
    elif correct_axes:
        xx, yy = np.meshgrid(-x, -x/inc_cos)
    else:
        xx, yy = np.meshgrid(x, x/inc_cos)

    image = np.exp(-(xx-x0)**2./sigma_x - (yy-y0)**2./sigma_y)

    return image.astype(dtype)


def create_sampling_points(nsamples, maxuv=1., dtype='float64'):
    assert isinstance(nsamples, int)

    np.random.seed(42)
    # columns are non contiguous arrays => copy
    x = np.random.uniform(low=-maxuv, high=maxuv, size=(nsamples, 2))
    return x[:, 0].astype(dtype), x[:, 1].astype(dtype)


def uv_idx(udat, vdat, uv):
    """
    uv coordinates to pixel coordinates in range [0, npixels].
    Assume image is square, same boundary in u and v direction.

    Parameters
    ----------

    uv: nd array
    u values at which FFT is computed. Assumed identical for v.

    """
    umin = uv[0]
    du = uv[1] - uv[0]

    u = np.floor((udat - umin) / du)
    v = np.floor((vdat - umin) / du)

    return u + (udat - uv) / du , v + (vdat - uv) / du


def pixel_coordinates(maxuv, nx, dtype='float64'):
    """
    Compute the array that maps the pixels of the image to real uv-coordinates.
    The array contains the coordinate of the pixel centers (not the edges!).

    """
    return (np.linspace(0., nx-1, nx, dtype=dtype) - nx/2.) * maxuv/np.float(nx)


def get_uv_idx_n(ux, vx, ur, vr, size):

    ntot = len(ur)
    assert len(ur) == len(vr)
    uri = np.zeros(len(ur), dtype=ur.dtype)
    vri = np.zeros(len(vr), dtype=vr.dtype)

    for i in range(ntot):
        i2u = size-1
        i1u = 0
        # binary search: index of closest u element
        while i2u-i1u > 1:
            itu = i1u + int(np.real(i2u-i1u)/2.)
            if ux[itu] > ur[i]:
                i2u = itu
            else:
                i1u = itu

        i2v = size-1
        i1v=0
        while i2v-i1v > 1:
            itv=i1v+int(np.real(i2v-i1v)/2.)
            if vx[itv] > vr[i]:
                i2v = itv
            else:
                i1v = itv

        uri[i] = i1u + np.real(ur[i]-ux[i1u])/(ux[i2u]-ux[i1u])
        vri[i] = i1v + np.real(vr[i]-vx[i1v])/(vx[i2v]-vx[i1v])

    return uri, vri


def int_bilin(f, x, y):

    nd = len(x)

    fint = np.zeros(nd, dtype=f.dtype)

    for i in range(nd):
        jj = int(x[i])
        ii = int(y[i])
        dfj = f[ii + 1, jj] - f[ii, jj]           # x
        dfj1 = f[ii + 1, jj + 1] - f[ii, jj + 1]  # y
        # numpy has weird promotion rules. Use `trunc` instead of `int` to preserve types of `x` and `f`
        fix = f[ii, jj] + dfj * (x[i] - np.trunc(x[i]))
        fix1 = f[ii + 1, jj] + dfj1 * (x[i] - np.trunc(x[i]))
        fint[i] = fix + (fix1 - fix) * (y[i] - np.trunc(y[i]))

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


def Fourier_shift_static(ft_centered, x0, y0, wle, maxuv):
    """
    Performs a translation in the real space by applying a phase shift in the Fourier space.
    This function applies the shift to 2D arrays (i.e. images).

    Parameters
    ----------
    ft_centered: 2D float array, complex64
        Fourier transform
    x0, y0: floats, arcsec
        Shifts in the real space.

    Returns
    -------
    v_shifted: 2D float array, complex64
        Phase-shifted Fourier transform

    """
    nx = ft_centered.shape[0]
    # convert x0, y0 from arcsec to pixel

    sec2pixel = sec2rad/wle
    x0 *= sec2pixel
    y0 *= sec2pixel

    # construct the phase change
    spatial_freq = maxuv*np.fft.fftshift(np.fft.fftfreq(nx))*2.*np.pi
    uu, vv = np.meshgrid(spatial_freq, spatial_freq)
    uv_grid = uu*x0 + vv*y0
    cos_theta = np.cos(uv_grid)
    sin_theta = np.sin(uv_grid)

    # apply the phase change
    re_ft_c, im_ft_c = ft_centered.real, ft_centered.imag
    re_v_shifted = re_ft_c*cos_theta - im_ft_c*sin_theta
    imag_v_shifted = im_ft_c*cos_theta + re_ft_c*sin_theta

    v_shifted = re_v_shifted+1j*imag_v_shifted

    return v_shifted


def Fourier_shift_array(u, v, fint, x0, y0):
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
    x0 *= sec2rad
    y0 *= sec2rad

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
