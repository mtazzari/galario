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

from utils import generate_random_vis, create_reference_image, create_sampling_points
import galario

import optparse
p = optparse.OptionParser()
p.add_option("--gpu", action="store_true", dest="USE_GPU", default=False,
             help="Use GPU version of galario")
p.add_option("--timing", action="store_true", dest="TIMING", default=False,
             help="Time chi2()")

(options, args) = p.parse_args()

if options.USE_GPU:
    if galario.HAVE_CUDA:

        from galario import double_cuda as acc_lib

        # use last gpu if available. Check `watch -n 0.1 nvidia-smi` to see which gpu is
        # used during test execution.
        ngpus = acc_lib.ngpus()
        acc_lib.use_gpu(0) #max(0, ngpus - 1))

        acc_lib.threads_per_block()
    else:
        print("Option --gpu not valid. galario.HAVE_CUDA is {}. Terminating.".format(galario.HAVE_CUDA))

else:
    from galario import double as acc_lib


def setup_sample(size, nsamples, real_type):

    # these number can be freely changed for this speed test
    dRA = -3.1
    dDec = 2.5
    wle_m = 1.

    # generate the samples
    maxuv = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv/2.2, dtype=real_type)
    x, _, w = generate_random_vis(nsamples, real_type)

    # create model image (it happens to have 0 imaginary part)
    ref_image = create_reference_image(size=size, kernel='gaussian', dtype=real_type)

    return ref_image, dRA, dDec, maxuv/size/wle_m, udat/wle_m, vdat/wle_m


def setup_chi2(size, nsamples, real_type):

    ref_image, dRA, dDec, maxuv, udat, vdat = setup_sample(size, nsamples, real_type)
    x, _, w = generate_random_vis(nsamples, real_type)

    return ref_image, dRA, dDec, maxuv, udat, vdat, x.real.copy(), x.imag.copy(), w
  

if __name__ == '__main__':

    size = 4096
    nsamples = int(1e6)
    real_type = "float64"

    if not options.TIMING:
        input_sample = setup_sample(size, nsamples, real_type)
        input_chi2 = setup_chi2(size, nsamples, real_type)

        acc_lib.sample(*input_sample)

        acc_lib.chi2(*input_chi2)

    else:
        omp_num_threads = os.environ.get('OMP_NUM_THREADS', 1)

        cycles = 5
        number = 1
        t = timeit.Timer('acc_lib.chi2(*x)'.format(acc_lib),
                         setup=textwrap.dedent("""
                            from __main__ import setup_chi2, acc_lib
                            x = setup_chi2(int({0}), int({1}), "{2}")
                            # from galario import double_cuda as acc_lib
                            # ngpus = acc_lib.ngpus()
                            # acc_lib.use_gpu(0) #max(0, ngpus - 1))
                            # acc_lib.threads_per_block()
                            """.format(size, nsamples, real_type)))

        str_headers = "\t".join(["size", "nsamples", "real", "OMP", "Ttot", "Tavg", "Tstd", "Tmin"])

        filename = "timings_"
        if options.USE_GPU:
            filename += "GPU"
        else:
            filename += "CPU_OMP_NUM_THREADS_{}".format(omp_num_threads)
        filename += "_{}.txt".format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

        t_results = t.repeat(cycles, number)
        str_results = "{}\t{:e}\t{}\t{}\t{:e}\t{:e}\t{:e}\t{:e}".format(size, nsamples, real_type, omp_num_threads, np.sum(t_results),
                                  np.average(t_results), np.std(t_results), np.min(t_results))

        with open(filename, 'w') as f:
            f.write(str_headers + "\n")
            f.write(str_results + "\n")
            f.write(" |--> timings: {}".format(t_results) + "\n")

        print(str_headers)
        print(str_results)
        print(t_results)
        print("Log saved in {}".format(filename))


