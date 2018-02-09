#pragma once

#include "galario_defs.h"

namespace galario {
/* functions for python interface. Need void* to be independent of C
 * type (host vs device) which can't be parsed by cython.
 */

/* Main user functions */
void _sample_profile(int nr, void *intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA, dreal dDec,
                     dreal duv, dreal PA, int nd, void *u, void *v, void *fint);
void _sample_image(int nx, int ny, void* data, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* fint);
dreal _chi2_profile(int nr, void *intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA, dreal dDec,
                    dreal duv, dreal PA, int nd, void *u, void *v, void *fobs_re, void *fobs_im, void *weights);
dreal _chi2_image(int nx, int ny, void* data, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* fobs_re, void* fobs_im, void* weights);
void _sweep(int nr, void *intensity, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal inc, void *image);
void _uv_rotate(dreal PA, dreal dRA, dreal dDec, void* dRArot, void* dDecrot, int nd, void* u, void* v, void* urot, void* vrot);

/* Interface for the experts */
void* _copy_input(int nx, int ny, void* realdata);
void _fft2d(int nx, int ny, void* data);
void _fftshift(int nx, int ny, void* data);
void _fftshift_axis0(int nx, int ncol, void* data);
void _interpolate(int nx, int ncol, void *data, int nd, void *u, void *v, dreal duv, void *fint);
void _apply_phase_sampled(dreal dRA, dreal dDec, int nd, void* u, void* v, void* fint);
dreal _reduce_chi2(int nd, void* fobs_re, void* fobs_im, void* fint, void* weights);

}
