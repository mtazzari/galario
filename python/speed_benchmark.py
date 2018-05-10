#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import datetime
import timeit

from utils import generate_random_vis, create_reference_image, create_sampling_points
import galario
from galario import au, pc, cgs_to_Jy, arcsec
from utils import *

import argparse
p = argparse.ArgumentParser()
p.add_argument("--gpu", action="store_true", dest="USE_GPU", default=False,
             help="Use GPU version of galario")
p.add_argument("--cpu", action="store_true", dest="USE_CPU", default=True,
             help="Use CPU version of galario")
p.add_argument("--gpu_id", action="store", dest="gpu_id", default=0, type=int,
             help="Choose index of GPU if several are available.  Check `watch -n 0.1 nvidia-smi` to see which gpu is used during test execution")
p.add_argument("--timing", action="store_true", dest="TIMING", default=False,
             help="Time chi2()")
p.add_argument("--cycles", action="store", dest="cycles", default=5, type=int,
             help="Number of cycles in calls to test")
p.add_argument("--size", action="store", dest="size", default=4096, type=int,
             help="Square input image size")
p.add_argument("--nsamples", action="store", dest="nsamples", default=int(1e6), type=int,
             help="Number of uv points")
p.add_argument("--tpb", type=int, nargs='+', help="Threads per block on the GPU", default=[16])
p.add_argument("--ompnthreads", type=int, nargs='+', help="$OMP_NUM_THREADS", default=[1])
p.add_argument("--dtype", action="store", dest="dtype", default='float64',
             help="Data type of the input image")
p.add_argument("--output", action="store", dest="output", default="",
             help="Name of output file")
p.add_argument("--output_header", action="store_true", dest="output_header",
             help="Only create output file and print header, then quit.")
p.add_argument("--image", action="store_true", default=False,
             help="If True computes from Image, else computes from profile.")
p.add_argument("--use-py", action="store_true", dest="use_py",default=False,
             help="If True uses Python implementation of sample/chi2 Profile/Image.")
p.add_argument("--no-verbose", action="store_true", default=False,
               help="Print timing to stdout")

options = p.parse_args()
# print(options)

if options.USE_GPU:
    if galario.HAVE_CUDA:

        from galario import double_cuda as acc_lib_cuda

        acc_lib_cuda.use_gpu(options.gpu_id)

    else:
        print("Option --gpu not valid. galario.HAVE_CUDA is {}.".format(galario.HAVE_CUDA))
        options.USE_GPU = False

from galario import double as acc_lib_cpu


def setup_chi2Image(nxy, nsamples):

    # these number can be freely changed for this speed test
    dRA = -3.1
    dDec = 2.5
    PA = 80.

    maxuv_generator = 3e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype='float64')
    x, _, w = generate_random_vis(nsamples, options.dtype)

    _, _, maxuv = matrix_size(udat, vdat)
    dxy = 1 / maxuv

    # create model image (it happens to have 0 imaginary part)
    image_ref = create_reference_image(size=nxy, kernel='gaussian', dtype=options.dtype)

    return image_ref, dxy, udat, vdat, x.real.copy(), x.imag.copy(), w, dRA, dDec, PA


def setup_chi2Profile(nxy, nsamples):

    pars = {'wle_m': 0.00088, 'dRA': 2.3, 'dDec': 3.2, 'PA': 88.}
    Rmin, dR, nrad, inc, profile_mode = 0.001, 0.001, 2000, 20., 'Gauss'
    Rmin *= arcsec
    dR *= arcsec

    wle_m = pars['wle_m']
    dRA = pars['dRA']
    dDec = pars['dDec']
    PA = pars['PA']

    # generate the samples
    maxuv_generator = 3e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=options.dtype)
    x, _, w = generate_random_vis(nsamples, options.dtype)

    _, _, maxuv = matrix_size(udat, vdat)
    maxuv /= wle_m
    dxy = 1 / maxuv
    print(dxy)
    # compute the matrix size and maxuv
    # nxy, dxy = g_double.get_image_size(udat/wle_m, vdat/wle_m)

    # compute radial profile
    intensity = radial_profile(Rmin, dR, nrad, profile_mode, dtype=options.dtype, gauss_width=150.*arcsec)

    return intensity, Rmin, dR, nxy, dxy, udat/wle_m, vdat/wle_m, x.real.copy(), x.imag.copy(), w, dRA, dDec, inc, PA


def do_timing(options, input_data, gpu=False, tpb=0, omp_num_threads=0):
    if gpu:
        acc_lib_cuda.threads(tpb)
    else:
        acc_lib_cpu.threads(omp_num_threads)

    str_headers = "\t".join(["size", "nsamples", "real", "OMP", "tpb", "Ttot", "Tavg", "Tstd", "Tmin"])
    if options.output_header:
        with open(options.output, 'w') as f:
            f.write(str_headers + "\n")
        exit(0)

    cycles = options.cycles
    number = 1

    acc_lib = 'acc_lib_cuda' if gpu else 'acc_lib_cpu'

    if options.image:
        if options.use_py:
            t = timeit.Timer('from __main__ import input_data, py_chi2Image; py_chi2Image(*input_data)')
        else:
            t = timeit.Timer('from __main__ import input_data, {}; {}.chi2Image(*input_data)'.format(acc_lib, acc_lib))
    else:
        if options.use_py:
            t = timeit.Timer('from __main__ import input_data, py_chi2Profile; py_chi2Profile(*input_data)')
        else:
            t = timeit.Timer('from __main__ import input_data, {}; {}.chi2Profile(*input_data)'.format(acc_lib, acc_lib))

    if options.output:
        filename = options.output
    else:
        filename = "timings_"
        if gpu:
            filename += "GPU_{}".format(tpb)
        else:
            filename += "CPU_OMP_NUM_THREADS_{}".format(omp_num_threads)
            filename += "_{}.txt".format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

    t_results = t.repeat(cycles, number)
    # drop 1st call: invovles lots of overhead
    str_results = "{}\t{:e}\t{}\t{}\t{}\t{:e}\t{:e}\t{:e}\t{:e}".format(size, nsamples, options.dtype, omp_num_threads, tpb, np.sum(t_results[1:]),
                                                                            np.average(t_results[1:]), np.std(t_results[1:]), np.min(t_results))

    with open(filename, 'a') as f:
        # f.write(str_headers + "\n")
        f.write(str_results + "\n")
        f.write("# |--> timings: {}".format(t_results) + "\n")

    if not options.no_verbose:
        print(str_headers)
        print(str_results)
        print(t_results)
        print("Log saved in {}".format(filename))

if __name__ == '__main__':

    size = options.size
    nsamples = options.nsamples

    if options.image:
        input_data = setup_chi2Image(size, nsamples)
    else:
        input_data = setup_chi2Profile(size, nsamples)

    if options.USE_GPU:
        for t in options.tpb:
            do_timing(options, input_data, tpb=t, gpu=True)
    if options.USE_CPU:
        for nthreads in options.ompnthreads:
            do_timing(options, input_data, omp_num_threads=nthreads)
