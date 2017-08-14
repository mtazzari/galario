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


def setup_sample(size, nsamples):

    sys.stdout.write("Setup sample...")
    sys.stdout.flush()

    # these number can be freely changed for this speed test
    dRA = -3.1
    dDec = 2.5
    wle_m = 1.

    # generate the samples
    maxuv = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv/2.2, dtype=options.dtype)
    x, _, w = generate_random_vis(nsamples, options.dtype)

    # create model image (it happens to have 0 imaginary part)
    ref_image = create_reference_image(size=size, kernel='gaussian', dtype=options.dtype)

    print("done")

    return ref_image, dRA, dDec, maxuv/size/wle_m, udat/wle_m, vdat/wle_m


def setup_chi2(size, nsamples):

    ref_image, dRA, dDec, maxuv, udat, vdat = setup_sample(size, nsamples)
    x, _, w = generate_random_vis(nsamples, options.dtype)

    return ref_image, dRA, dDec, maxuv, udat, vdat, x.real.copy(), x.imag.copy(), w


if __name__ == '__main__':
    str_headers = "\t".join(["size", "nsamples", "real", "OMP", "tpb", "Ttot", "Tavg", "Tstd", "Tmin"])
    if options.output_header:
        with open(options.output, 'w') as f:
            f.write(str_headers + "\n")
        exit(0)

    size = options.size
    nsamples = int(1e6)

    input_chi2 = setup_chi2(size, nsamples)

    if not options.TIMING:
        input_sample = setup_sample(size, nsamples)

        acc_lib.sample(*input_sample)

        acc_lib.chi2(*input_chi2)

    else:
        omp_num_threads = os.environ.get('OMP_NUM_THREADS', 1)

        cycles = options.cycles
        number = 1
        t = timeit.Timer('from __main__ import setup_chi2, input_chi2, acc_lib; acc_lib.chi2(*input_chi2)'.format(acc_lib))

        if options.output:
            filename = options.output
        else:
            filename = "timings_"
            if options.USE_GPU:
                filename += "GPU"
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
