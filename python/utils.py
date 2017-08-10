#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import, unicode_literals)

import numpy as np

__all__ = ["create_reference_image", "create_sampling_points", "uv_idx",
           "pixel_coordinates", "uv_idx_r2c", "int_bilin_MT",
           "matrix_size", "Fourier_shift_static",
           "Fourier_shift_array", "generate_random_vis",
           "sec2rad"]

sec2rad = np.pi/180./3600.  # from arcsec to radians


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
    assert isinstance(nsamples, int)

    np.random.seed(42)
    # columns are non contiguous arrays => copy
    x = np.random.uniform(low=-maxuv, high=maxuv, size=(nsamples, 2))
    return x[:, 0].astype(dtype), x[:, 1].astype(dtype)


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


def pixel_coordinates(maxuv, nx, dtype='float64'):
    """
    Compute the array that maps the pixels of the image to real uv-coordinates.
    The array contains the coordinate of the pixel centers (not the edges!).

    """
    return (np.linspace(0., nx-1, nx, dtype=dtype) - nx/2.) * maxuv/np.float(nx)

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
