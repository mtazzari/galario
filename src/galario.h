/******************************************************************************
* This file is part of GALARIO:                                               *
* Gpu Accelerated Library for Analysing Radio Interferometer Observations     *
*                                                                             *
* Copyright (C) 2017-2020, Marco Tazzari, Frederik Beaujean, Leonardo Testi.  *
*                                                                             *
* This program is free software: you can redistribute it and/or modify        *
* it under the terms of the Lesser GNU General Public License as published by *
* the Free Software Foundation, either version 3 of the License, or           *
* (at your option) any later version.                                         *
*                                                                             *
* This program is distributed in the hope that it will be useful,             *
* but WITHOUT ANY WARRANTY; without even the implied warranty of              *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                        *
*                                                                             *
* For more details see the LICENSE file.                                      *
* For documentation see https://mtazzari.github.io/galario/                   *
******************************************************************************/

#pragma once

#include "galario_defs.h"

namespace galario {

/* Main user functions */
void sample_profile(int nr, const dreal* intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA,
                    dreal dDec, dreal duv, dreal PA, int nd, const dreal *u, const dreal *v, dcomplex *vis_int);
void sample_image(int nx, int ny, const dreal* image, const dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u, const dreal* v, dcomplex* vis_int);
void sample_unstructured_image(const dreal* x, const dreal *y, int nx, int ny, dreal dxy, int ni, const dreal* image, const dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u, const dreal* v, dcomplex* vis_int);
dreal chi2_profile(int nr, const dreal* intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA,
                   dreal dDec, dreal duv, dreal PA, int nd, const dreal *u, const dreal *v, const dreal *vis_obs_re,
                   const dreal *vis_obs_im, const dreal *weights);
dreal chi2_image(int nx, int ny, const dreal* image, const dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u, const dreal* v, const dreal* vis_obs_re, const dreal* vis_obs_im, const dreal* weights);
dreal chi2_unstructured_image(const dreal* realx, const dreal* realy, int nx, int ny, dreal dxy, int ni, const dreal* realdata, const dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u, const dreal* v, const dreal* vis_obs_re, const dreal* vis_obs_im, const dreal* weights);
void sweep(int nr, const dreal* intensity, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal inc, dcomplex *image);
void uv_rotate(dreal PA, dreal dRA, dreal dDec, dreal* dRArot, dreal* dDecrot, int nd, const dreal* u, const dreal* v, dreal* urot, dreal* vrot);

/* Interface for the experts */
dcomplex* copy_input(int nx, int ny, const dreal* image);
void galario_free(void*);
void fft2d(int nx, int ny, dcomplex* image);
void fftshift(int nx, int ny, dcomplex* image);
void fftshift_axis0(int nx, int ny, dcomplex* matrix);
void interpolate(int nrow, int ncol, const dcomplex *image, const dreal v_origin, int nd,
                 const dreal* u, const dreal* v, const dreal duv,
                 dcomplex* vis_int);
void apply_phase_sampled(dreal dRA, dreal dDec, int nd, const dreal* u,
                         const dreal* v, dcomplex* vis_int);
dreal reduce_chi2(int nd, const dreal* vis_obs_re, const dreal* vis_obs_im,
                 const dcomplex* vis_int, const dreal* weights);

/* Required for multithreading */
void init();
void cleanup();
int threads(int num = 0);

/* GPU related functions */
int ngpus();
void use_gpu(int device_id);
}
