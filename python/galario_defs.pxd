###############################################################################
# This file is part of GALARIO:                                               #
# Gpu Accelerated Library for Analysing Radio Interferometer Observations     #
#                                                                             #
# Copyright (C) 2017-2020, Marco Tazzari, Frederik Beaujean, Leonardo Testi.  #
#                                                                             #
# This program is free software: you can redistribute it and/or modify        #
# it under the terms of the Lesser GNU General Public License as published by #
# the Free Software Foundation, either version 3 of the License, or           #
# (at your option) any later version.                                         #
#                                                                             #
# This program is distributed in the hope that it will be useful,             #
# but WITHOUT ANY WARRANTY; without even the implied warranty of              #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                        #
#                                                                             #
# For more details see the LICENSE file.                                      #
# For documentation see https://mtazzari.github.io/galario/                   #
###############################################################################

include "galario_config.pxi"

cdef extern from "galario_py.h" namespace "galario":
    # Main user functions
    void _sample_profile(int nr, void* intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis) except +
    void _sample_image(int nx, int ny, void* image, dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis) except +
    void _sample_unstructured_image(void* x, void* y, int nx, int ny, dreal dxy, int ni, void* image, dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis) except +
    dreal _chi2_profile(int nr, void* intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis_obs_re, void* vis_obs_im, void* vis_obs_w) except +
    dreal _chi2_image(int nx, int ny, void* image, dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis_obs_re, void* vis_obs_im, void* vis_obs_w) except +
    dreal _chi2_unstructured_image(void* x, void* y, int nx, int ny, dreal dxy, int ni, void* data, const dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis_obs_re, void* vis_obs_im, void* weights) except +
    void _sweep(int nr, void* intensity, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal inc, void* image) except +
    void _uv_rotate(dreal PA, dreal dRA, dreal dDec, void* dRArot, void* dDecrot, int nd, void* u, void* v, void* urot, void* vrot) except +

    # Interface for the experts
    void* _copy_input(int nx, int ny, void* realimage) except +
    void* _fft2d(int nx, int ny, void* image) except +
    void _fftshift(int nx, int ny, void* image) except +
    void _fftshift_axis0(int nx, int ny, void* image) except +
    void _interpolate(int nx, int ncol, void* image, dreal v_origin, int nd, void* u, void* v, dreal duv, void* vis) except +
    void _apply_phase_sampled(dreal dRA, dreal dDec, int nd, void* u, void* v, void* vis) except +
    dreal _reduce_chi2(int nd, void* vis_obs_re, void* vis_obs_im, void* vis, void* vis_obs_w) except +

cdef extern from "galario.h" namespace "galario":
    void init() except +
    void cleanup() except +
    int  threads(int num) except +
    void galario_free(void*) except +
    void use_gpu(int device_id) except +
    int  ngpus() except +
