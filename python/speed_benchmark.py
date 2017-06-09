#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from test_galario import sec2rad, create_reference_image, create_sampling_points, Fourier_shift_static, int_bilin, get_rotix_n, pixel_coordinates
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


def setup_chi2(size, nsamples, real_type, complex_type):

    maxuv = 1000.
    wle_m = 0.003
    factor = 2.*np.pi*sec2rad/wle_m*maxuv
    x0_arcsec = 2.3 * factor
    y0_arcsec = -1.4 * factor

    # reference_image = create_reference_image(size=size, kernel='gaussian', dtype=complex_type)
    reference_image = np.ones((size, size), dtype=complex_type)
    udat, vdat = create_sampling_points(nsamples, maxuv/2.2, dtype=real_type)
    # this factor has to be > than 2 because the matrix cover between -maxuv/2 to +maxuv/2,
    # therefore the sampling points have to be contained inside.

    # no rotation
    uv = pixel_coordinates(maxuv, size).astype(real_type)

    # CPU version
    # cpu_shift_fft_shift = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(reference_image)))
    # fourier_shifted = Fourier_shift_static(cpu_shift_fft_shift, x0_arcsec, y0_arcsec, wle_m, maxuv)

    # compute interpolation and chi2
    uroti, vroti = get_rotix_n(uv, uv, udat, vdat, size)
    uroti = uroti.astype(real_type)
    vroti = vroti.astype(real_type)
    # ReInt = int_bilin(fourier_shifted.real, uroti, vroti).astype(real_type)
    # ImInt = int_bilin(fourier_shifted.imag, uroti, vroti).astype(real_type)
    ReInt = np.ones(nsamples, dtype=real_type)
    ImInt = np.ones(nsamples, dtype=real_type)

    w = np.random.uniform(low=0., high=1e4, size=nsamples).astype(real_type)
    w /= w.sum()

    return reference_image, x0_arcsec, y0_arcsec, uv, udat, vdat, ReInt, ImInt, w


def function_to_test(*args, **kwargs):

    acc_lib = kwargs['acc_lib']

    chi2_cuda = acc_lib.chi2(*args)


if __name__ == '__main__':
    cycles = 5
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

    modes = [["single_cuda", "float32", "complex64"],
             ["double_cuda", "float64", "complex128"]]
    i_tests = 0

    for mode in modes:
        acc_lib, real_type, complex_type = mode

        for size in [256, 512, 1024, 2048, 4096, 8192, 16384]:
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

            # call timeit `cycles` times. timeit returns sum of execution
            t_results = t.repeat(cycles, number)
            str_results = "\t".join(["{}".format(x) for x in
                                     [i_tests, size, nsamples, real_type, complex_type, np.sum(t_results),
                                      np.average(t_results), np.std(t_results),
                                      np.min(t_results)]])

            with open(sys.argv[1], 'a') as f:
                f.write(str_results + "\n")
            print(t_results)
            print(str_results)


