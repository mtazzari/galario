#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import os
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

        from galario import single_cuda as acc_lib

        # use last gpu if available. Check `watch -n 0.1 nvidia-smi` to see which gpu is
        # used during test execution.
        ngpus = acc_lib.ngpus()
        acc_lib.use_gpu(0) #max(0, ngpus - 1))

        acc_lib.threads_per_block()
    else:
        print("Option --gpu not valid. galario.HAVE_CUDA is {}. Terminating.".format(galario.HAVE_CUDA))

else:
    from galario import single as acc_lib


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
    real_type = "float32"
    input_sample = setup_sample(size, nsamples, real_type)
    input_chi2 = setup_chi2(size, nsamples, real_type)

    acc_lib.sample(*input_sample)

    acc_lib.chi2(*input_chi2)

