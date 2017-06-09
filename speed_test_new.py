#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

# from pyvfit.imager import Imager
# from pyvfit.tests import create_sampling_points, create_reference_image
import numpy as np
import timeit
import multiprocessing
import sys
import time
import textwrap


# import optparse
# p = optparse.OptionParser()
# p.add_option("--cpu", action="store_true", dest="use_cpu", default=False,
#              help="Use CPU version of the accelerated library in pyvfit_libcpu")
#

sec2rad = 4.848136811e-06  # from arcsec to radians



# (options, args) = p.parse_args()
# USE_CPU = options.use_cpu

# now, I bypass the parsed options
# USE_CPU = False

# if not USE_CPU:
#     from pyvfit.cuda import pyvfit_libgpu as acc_lib
#     print("!!! Using GPU version !!!")
# else:
#     from pyvfit.cuda import pyvfit_libcpu as acc_lib
#     print("!!!! Using CPU version !!!")

# run_mode = "CPU" if USE_CPU==True else "GPU"

def save_init_test(size=1024, nsamples=100):
    image, uv, udat, vdat, obs_re, obs_im, obs_w = init_test(size, nsamples)
    np.save("image_{}".format(size), image)
    np.save("uv_{}".format(size), uv)
    np.save("udat_{}".format(nsamples), udat)
    np.save("vdat_{}".format(nsamples), vdat)
    np.save("obs_re_{}".format(nsamples), obs_re)
    np.save("obs_im_{}".format(nsamples), obs_im)
    np.save("obs_w_{}".format(nsamples), obs_w)

def load_init_test(size=1024, nsamples=100, double_prec=True):

    image = np.load("image_{}.npy".format(size))
    if not double_prec:
        ARR_TYPE = 'float32'
        image = image.astype('complex64')
    else:
        ARR_TYPE = 'float64'
        image = image.astype('complex128')

    uv = np.load("uv_{}.npy".format(size)).astype(ARR_TYPE)
    udat = np.load("udat_{}.npy".format(nsamples)).astype(ARR_TYPE)
    vdat = np.load("vdat_{}.npy".format(nsamples)).astype(ARR_TYPE)
    obs_re = np.load("obs_re_{}.npy".format(nsamples)).astype(ARR_TYPE)
    obs_im = np.load("obs_im_{}.npy".format(nsamples)).astype(ARR_TYPE)
    obs_w = np.load("obs_w_{}.npy".format(nsamples)).astype(ARR_TYPE)

    x0 = 0.5
    y0 = -21.
    rank = 0


    return image, uv, udat, vdat, obs_re, obs_im, obs_w, x0, y0, rank


def speed_test(x):

    image, uv, udat, vdat, obs_re, obs_im, obs_w, x0, y0, rank = x
    y = acc_lib.acc_everything(image, -x0, -y0, uv, udat, vdat,
                                            obs_re, obs_im, obs_w, rank)






####################### these can be imported from galario.test_galario
def pixel_coordinates(maxuv, nx):
    """
    Compute the array that maps the pixels of the image to real uv-coordinates.
    The array contains the coordinate of the pixel centers (not the edges!).

    """
    # TODO: should divide by /nx instead of /(nx-1)

    return (np.linspace(0., nx-1, nx) - nx/2.) * maxuv/(nx-1)

def get_rotix_n(ux, vx, ur, vr, size):

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
        ii = int(x[i])
        jj = int(y[i])
        dfj = f[ii + 1, jj] - f[ii, jj]           # x
        dfj1 = f[ii + 1, jj + 1] - f[ii, jj + 1]  # y
        fix = f[ii, jj] + dfj * (x[i] - int(x[i]))
        fix1 = f[ii + 1, jj] + dfj1 * (x[i] - int(x[i]))
        fint[i] = fix + (fix1 - fix) * (y[i] - int(y[i]))

    return fint


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
    uu, vv = np.meshgrid(spatial_freq*y0, spatial_freq*x0)
    uv_grid = uu+vv
    cos_theta = np.cos(uv_grid)
    sin_theta = -np.sin(uv_grid)

    # apply the phase change
    re_ft_c, im_ft_c = ft_centered.real, ft_centered.imag
    re_v_shifted = re_ft_c*cos_theta - im_ft_c*sin_theta
    imag_v_shifted = im_ft_c*cos_theta + re_ft_c*sin_theta

    v_shifted = re_v_shifted+1j*imag_v_shifted

    return v_shifted

def create_sampling_points(nsamples, maxuv=1., dtype='float64'):
    np.random.seed(42)

    # columns are non contiguous arrays => copy
    x = np.random.uniform(low=-maxuv, high=maxuv, size=(nsamples, 2))
    return x[:, 0].astype(dtype), x[:, 1].astype(dtype)

def create_reference_image(size=1024, kernel='gaussian', save=False, dtype='float64'):
    try:
        import astropy
    except ImportError:
        print("Please install astropy with:\n\tconda install astropy")
        exit(1)

    from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
    from astropy.modeling.models import Gaussian2D

    np.random.seed(42)

    gauss = Gaussian2D(1, 0, 0, 3, 3)

    # Fake image data including noise
    x = np.linspace(-100, 100, size)
    y = np.linspace(-100, 100, size)
    x, y = np.meshgrid(x, y)
    data_2D = gauss(x, y) + 0.1 * (np.random.rand(size, size) - 0.5)

    if kernel == 'tophat':
        kernel = Tophat2DKernel(30)
    elif kernel == 'gaussian':
        kernel = Gaussian2DKernel(2)

    reference_image = convolve(data_2D, kernel)

    # ensure it is positive
    reference_image += np.abs(reference_image.min())

    return reference_image.astype(dtype)

######################################


def setup_chi2(size, nsamples, real_type, complex_type):

    maxuv = 1000.
    wle_m = 0.003
    factor = 2.*np.pi*sec2rad/wle_m*maxuv
    x0_arcsec = 2.3 * factor
    y0_arcsec = -1.4 * factor

    reference_image = create_reference_image(size=size, kernel='gaussian', dtype=complex_type)
    udat, vdat = create_sampling_points(nsamples, maxuv/2.2, dtype=real_type)
    # this factor has to be > than 2 because the matrix cover between -maxuv/2 to +maxuv/2,
    # therefore the sampling points have to be contained inside.

    # no rotation
    uv = pixel_coordinates(maxuv, size).astype(real_type)

    # CPU version
    cpu_shift_fft_shift = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(reference_image)))
    fourier_shifted = Fourier_shift_static(cpu_shift_fft_shift, x0_arcsec, y0_arcsec, wle_m, maxuv)

    # compute interpolation and chi2
    uroti, vroti = get_rotix_n(uv, uv, udat, vdat, size)
    uroti = uroti.astype(real_type)
    vroti = vroti.astype(real_type)
    ReInt = int_bilin(fourier_shifted.real, uroti, vroti).astype(real_type)
    ImInt = int_bilin(fourier_shifted.imag, uroti, vroti).astype(real_type)

    w = np.random.uniform(low=0., high=1e4, size=nsamples).astype(real_type)
    w /= w.sum()


    return reference_image, x0_arcsec, y0_arcsec, uv, udat, vdat, ReInt, ImInt, w


def function_to_test(*args, **kwargs):

    acc_lib = kwargs['acc_lib']

    chi2_cuda = acc_lib.chi2(*args)


if __name__ == '__main__':
    # import timeit
    cycles = 3
    number = 1
    double_prec = True
    #
    str_headers = "\t".join(
        ["i", "size", "nsamples", "real", "complex", "Ttot", "Tavg", "Tstd",
         "Tmin"])
    print(str_headers)
    with open(sys.argv[1], 'a') as f:
        f.write(str_headers + "\n")

    nsamples = 1000

    modes = [["single", "float32", "complex64"],
             ["double", "float64", "complex128"]]
    i_tests = 0

    for mode in modes:
        acc_lib, real_type, complex_type = mode

        for size in [256, 512, 1024, 2048, 4096]:  # , 8192]:
            # for nsamples in [1000, 1000000]:
            i_tests += 1
            t = timeit.Timer('function_to_test(*x, acc_lib={})'.format(acc_lib),
                             setup=textwrap.dedent("""
                             from __main__ import function_to_test, setup_chi2; 
                             from galario import {4};
                             x = setup_chi2({0}, {1}, "{2}", "{3}")
                             """
                             .format(
                                 size, nsamples, real_type, complex_type, acc_lib)))

            t_results = t.repeat(cycles, number)

            #
            str_results = "\t".join(["{}".format(x) for x in
                                     [i_tests, size, nsamples, real_type, complex_type, np.sum(t_results),
                                      np.average(t_results), np.std(t_results),
                                      np.min(t_results)]])

            with open(sys.argv[1], 'a') as f:
                f.write(str_results + "\n")
            print(str_results)


