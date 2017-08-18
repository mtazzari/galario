#pragma once

#include "galario_defs.h"

/* functions for python interface. Need void* to be independent of C
 * type (host vs device) which can't be parsed by cython.
 */

/* Main user functions */
void _galario_sample_profile(int nr, void* ints, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal dist, dreal inc, dreal dRA, dreal dDec, dreal duv, int nd, void* u, void* v, void* fint);
void _galario_sample_image(int nx, int ny, void* data, dreal dRA, dreal dDec, dreal duv, int nd, void* u, void* v, void* fint);
void _galario_chi2_profile(int nr, void* ints, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal dist, dreal inc,
                           dreal dRA, dreal dDec, dreal duv, int nd, void* u, void* v,
                           void* fobs_re, void* fobs_im, void* weights, dreal* chi2);
void _galario_chi2_image(int nx, int ny, void* data, dreal dRA, dreal dDec, dreal duv, int nd, void* u, void* v, void* fobs_re, void* fobs_im, void* weights, dreal* chi2);
void _galario_sweep(int nr, void* ints, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal dist, dreal inc, void* image);

/* Interface for the experts */
void* _galario_copy_input(int nx, int ny, void* realdata);
void _galario_fft2d(int nx, int ny, void* data);
void _galario_fftshift(int nx, int ny, void* data);
void _galario_fftshift_axis0(int nx, int ncol, void* data);
void _galario_interpolate(int nx, int ncol, void *data, int nd, void *u, void *v, dreal duv, void *fint);
void _galario_apply_phase_sampled(dreal dRA, dreal dDec, int nd, void* u, void* v, void* fint);
void _galario_reduce_chi2(int nd, void* fobs_re, void* fobs_im, void* fint, void* weights, dreal* chi2);
