#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import os
import datetime
import textwrap
import timeit
import optparse
import sys

from utils import generate_random_vis, create_reference_image, create_sampling_points
import galario
from galario import au, pc, cgs_to_Jy
from utils import *

import optparse
p = optparse.OptionParser()
p.add_option("--gpu", action="store_true", dest="USE_GPU", default=False,
             help="Use GPU version of galario")
p.add_option("--gpu_id", action="store", dest="gpu_id", default=0, type=int,
             help="Choose index of GPU if several are available.  Check `watch -n 0.1 nvidia-smi` to see which gpu is used during test execution")
p.add_option("--timing", action="store_true", dest="TIMING", default=False,
             help="Time chi2()")
p.add_option("--cycles", action="store", dest="cycles", default=5, type=int,
             help="Number of cycles in calls to test")
p.add_option("--size", action="store", dest="size", default=4096, type=int,
             help="Square input image size")
p.add_option("--tpb", action="store", dest="tpb", default=0, type=int,
             help="Threads per block on the GPU")
p.add_option("--dtype", action="store", dest="dtype", default='float64',
             help="Data type of the input image")
p.add_option("--output", action="store", dest="output", default="",
             help="Name of output file")
p.add_option("--output_header", action="store_true", dest="output_header",
             help="Only create output file and print header, then quit.")
p.add_option("--image", action="store_true", default=False,
             help="If True computes from Image, else computes from profile.")

(options, args) = p.parse_args()

if options.USE_GPU:
    if galario.HAVE_CUDA:

        from galario import double_cuda as acc_lib

        acc_lib.use_gpu(options.gpu_id)

        if options.tpb:
            acc_lib.threads_per_block(options.tpb)
    else:
        print("Option --gpu not valid. galario.HAVE_CUDA is {}. Terminating.".format(galario.HAVE_CUDA))

else:
    from galario import double as acc_lib


def setup_sampleImage(nxy, nsamples):

    sys.stdout.write("Setup sampleImage...")
    sys.stdout.flush()

    # these number can be freely changed for this speed test
    dRA = -3.1
    dDec = 2.5
    PA = 80.

    dist = 130. * pc
    maxuv_generator = 3e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype='float64')

    _, _, maxuv = matrix_size(udat, vdat)
    dxy = dist / maxuv

    # generate the samples
    maxuv = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv/2.2, dtype=options.dtype)

    # create model image (it happens to have 0 imaginary part)
    image_ref = create_reference_image(size=nxy, kernel='gaussian', dtype=options.dtype)

    print("...done")
    return image_ref, dxy, dist, udat, vdat, dRA, dDec, PA


def setup_chi2(size, nsamples):

    sys.stdout.write("Setup chi2Image...")
    sys.stdout.flush()

    image_ref, dxy, dist, udat, vdat, dRA, dDec, PA = setup_sampleImage(size, nsamples)
    x, _, w = generate_random_vis(nsamples, options.dtype)

    print("...done")

    return image_ref, dxy, dist, udat, vdat, x.real.copy(), x.imag.copy(), w, dRA, dDec

def setup_chi2Profile(nxy, nsamples):

    sys.stdout.write("Setup chi2Image...")
    sys.stdout.flush()

    pars = {'wle_m': 0.00088, 'dRA': 2.3, 'dDec': 3.2, 'PA': 88., 'nxy': 4096}
    Rmin, dR, nrad, inc, profile_mode, real_type = 0.1, 1., 500, 20., 'Gauss', 'float64'
    Rmin *= au
    dR *= au

    wle_m = pars['wle_m']
    dRA = pars['dRA']
    dDec = pars['dDec']
    PA = pars['PA']

    # generate the samples
    maxuv_generator = 3e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)
    x, _, w = generate_random_vis(nsamples, real_type)

    dist = 130. * pc

    _, _, maxuv = matrix_size(udat, vdat, maxuv_factor=3.)
    maxuv /= wle_m
    dxy = dist / maxuv

    # print(nxy, minuv, maxuv, duv, dxy/au)
    # compute the matrix size and maxuv
    # nxy, dxy = g_double.get_image_size(dist, udat/wle_m, vdat/wle_m)

    # compute radial profile
    ints = radial_profile(Rmin, dR, nrad, profile_mode, dtype=real_type, gauss_width=150.)

    print("...done")

    return ints, Rmin, dR, nxy, dxy, dist, udat/wle_m, vdat/wle_m, x.real.copy(), x.imag.copy(), w, inc/180.*np.pi, dRA, dDec

if __name__ == '__main__':
    str_headers = "\t".join(["size", "nsamples", "real", "OMP", "tpb", "Ttot", "Tavg", "Tstd", "Tmin"])
    if options.output_header:
        with open(options.output, 'w') as f:
            f.write(str_headers + "\n")
        exit(0)

    size = options.size
    nsamples = int(1e6)

    input_chi2 = setup_chi2(size, nsamples)
    input_chi2Profile = setup_chi2Profile(size, nsamples)

    if not options.TIMING:
        input_sample = setup_sampleImage(size, nsamples)

        acc_lib.sampleImage(*input_sample)

        acc_lib.chi2Image(*input_chi2)

    else:
        omp_num_threads = os.environ.get('OMP_NUM_THREADS', 0)

        cycles = options.cycles
        number = 1

        if options.image:
            t = timeit.Timer('from __main__ import input_chi2, acc_lib; acc_lib.chi2Image(*input_chi2)')
        else:
            t = timeit.Timer('from __main__ import input_chi2Profile, acc_lib; acc_lib.chi2Profile(*input_chi2Profile)')

        if options.output:
            filename = options.output
        else:
            filename = "timings_"
            if options.USE_GPU:
                filename += "GPU_{}".format(options.tpb)
            else:
                filename += "CPU_OMP_NUM_THREADS_{}".format(omp_num_threads)
            filename += "_{}.txt".format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

        t_results = t.repeat(cycles, number)
        # drop 1st call: invovles lots of overhead
        str_results = "{}\t{:e}\t{}\t{}\t{}\t{:e}\t{:e}\t{:e}\t{:e}".format(size, nsamples, options.dtype, omp_num_threads, options.tpb, np.sum(t_results[1:]),
                                  np.average(t_results[1:]), np.std(t_results[1:]), np.min(t_results))

        with open(filename, 'a') as f:
            # f.write(str_headers + "\n")
            f.write(str_results + "\n")
            f.write("# |--> timings: {}".format(t_results) + "\n")

        print(str_headers)
        print(str_results)
        print(t_results)
        print("Log saved in {}".format(filename))
