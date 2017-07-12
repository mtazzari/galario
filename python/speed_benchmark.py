#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import timeit
import os
import textwrap
import datetime

from utils import generate_random_vis, create_reference_image, create_sampling_points
import galario

import optparse
p = optparse.OptionParser()
p.add_option("--gpu", action="store_true", dest="USE_GPU", default=False,
             help="Use GPU version of galario")
(options, args) = p.parse_args()

if options.USE_GPU:
    if galario.HAVE_CUDA and options.USE_GPU:

        from galario import double_cuda as g_double
        from galario import single_cuda as g_single

        # use last gpu if available. Check `watch -n 0.1 nvidia-smi` to see which gpu is
        # used during test execution.
        ngpus = g_double.ngpus()
        g_double.use_gpu(max(0, ngpus - 1))

        g_double.threads_per_block()
    else:
        print("Option --gpu not valid. galario.HAVE_CUDA is {}. Terminating.".format(galario.HAVE_CUDA))

else:
    from galario import double as g_double
    from galario import single as g_single




def setup_chi2(size, nsamples, real_type):

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

    return ref_image, dRA, dDec, maxuv/size/wle_m, udat/wle_m, vdat/wle_m, x.real.copy(), x.imag.copy(), w


if __name__ == '__main__':


    # fetch environment
    omp_num_threads = os.environ.get('OMP_NUM_THREADS', 1)

    cycles = 5
    number = 1
    double_prec = True

    str_headers = "\t".join(["i", "size", "nsamples", "lib", "real", "Ttot", "Tavg", "Tstd", "Tmin"])
    str_headers += "\n"
    print(str_headers)

    filename = "timings_"

    if options.USE_GPU:
        filename += "GPU"
    else:
        filename += "CPU_OMP_NUM_THREADS_{}".format(omp_num_threads)

    filename += "_{}.txt".format(datetime.datetime.now().isoformat())

    with open(filename, 'w') as f:
        f.write(str_headers)

    # modes: table with basic parameters that we want to change for testing
    #        each entry: [name of the library, real_type, threads_per_block]
    modes = [["g_single", "float32", 32],
             ["g_double", "float64", 32]]

    # NOTE: the option to change threads per block programmatically has to be implemented.

    i_tests = 0

    for mode in modes:
        acc_lib, real_type, threads_per_block = mode

        for size in [256, 512, 1024, 2048, 4096, 8192, 16384]:
            # we may want to switch off this loop over nsamples at the beginning
            for nsamples in [1e3, 1e6]:

                i_tests += 1
                t = timeit.Timer('{}.chi2(*x)'.format(acc_lib),
                                 setup=textwrap.dedent("""
                                 from __main__ import setup_chi2, {3};
                                 x = setup_chi2(int({0}), int({1}), "{2}")
                                 """
                                 .format(
                                     size, nsamples, real_type, acc_lib)))

                # call timeit `cycles` times. timeit returns sum of execution
                t_results = t.repeat(cycles, number)
                str_results = "{}\t{}\t{}\t{}\t{}\t{:e}\t{:e}\t{:e}\t{:e}".format(i_tests, size, nsamples, acc_lib, real_type, np.sum(t_results),
                                          np.average(t_results), np.std(t_results), np.min(t_results))

                with open(filename, 'a') as f:
                    f.write(str_results + "\n")

                print(str_results)
                # these detailed timings are not stored in the log
                print(" |--> timings: {}".format(t_results))
