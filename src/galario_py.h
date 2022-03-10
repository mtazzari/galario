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
/* functions for python interface. Need void* to be independent of C
 * type (host vs device) which can't be parsed by cython.
 */

/* Main user functions */
void _sample_profile(int nr, void *intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA, dreal dDec,
                     dreal duv, dreal PA, int nd, void *u, void *v, void *vis_int);
void _sample_image(int nx, int ny, void* data, dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis_int);
void _sample_unstructured_image(void* x, void* y, int nx, int ny, dreal dxy, int ni, void* data, dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis_int);
dreal _chi2_profile(int nr, void *intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA, dreal dDec,
                    dreal duv, dreal PA, int nd, void *u, void *v, void *vis_obs_re, void *vis_obs_im, void *weights);
dreal _chi2_image(int nx, int ny, void* data, dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis_obs_re, void* vis_obs_im, void* weights);
dreal _chi2_unstructured_image(void* realx, void* realy, int nx, int ny, dreal dxy, int ni, void* realdata, const dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis_obs_re, void* vis_obs_im, void* weights);
void _sweep(int nr, void *intensity, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal inc, void *image);
void _uv_rotate(dreal PA, dreal dRA, dreal dDec, void* dRArot, void* dDecrot, int nd, void* u, void* v, void* urot, void* vrot);

/* Interface for the experts */
void* _copy_input(int nx, int ny, void* realdata);
void _fft2d(int nx, int ny, void* data);
void _fftshift(int nx, int ny, void* data);
void _fftshift_axis0(int nx, int ncol, void* data);
void _interpolate(int nx, int ncol, void *data, dreal v_origin, int nd, void *u, void *v, dreal duv, void *vis_int);
void _apply_phase_sampled(dreal dRA, dreal dDec, int nd, void* u, void* v, void* vis_int);
dreal _reduce_chi2(int nd, void* vis_obs_re, void* vis_obs_im, void* vis_int, void* weights);

}
