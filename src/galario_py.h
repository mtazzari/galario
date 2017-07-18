#pragma once

#include "galario_defs.h"

/* functions for python interface. Need void* to be independent of C
 * type (host vs device) which can't be parsed by cython.
 */

/* Main user functions */
void _galario_sample(int nx, void* data, dreal dRA, dreal dDec, dreal du, int nd, void* u, void* v, void* fint);
void _galario_chi2(int nx, void* data, dreal dRA, dreal dDec, dreal du, int nd, void* u, void* v, void* fobs_re, void* fobs_im, void* weights, dreal* chi2);

/* Interface for the experts */
void _galario_fft2d(int nx, void* data);
void _galario_fftshift(int nx, void* data);
void _galario_fftshift_fft2d_fftshift(int nx, void* data);
void _galario_interpolate(int nx, void* data, int nd, void* u, void* v, void* fint);
void _galario_apply_phase_2d(int nx, void* data, dreal dRA, dreal dDec);
void _galario_apply_phase_sampled(dreal dRA, dreal dDec, int nd, void* u, void* v, void* fint);
void _galario_get_uv_idx(int nx, dreal du, int nd, void* u, void* v, void* indu, void* indv);
void _galario_reduce_chi2(int nd, void* fobs_re, void* fobs_im, void* fint, void* weights, dreal* chi2);
