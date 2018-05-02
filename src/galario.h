#pragma once

#include "galario_defs.h"

namespace galario {

/* Main user functions */
void sample_profile(int nr, dreal *const intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA,
                    dreal dDec, dreal duv, dreal PA, int nd, const dreal *u, const dreal *v, dcomplex *fint);
void sample_image(int nx, int ny, const dreal* image, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u, const dreal* v, dcomplex* fint);
dreal chi2_profile(int nr, dreal *const intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA,
                   dreal dDec, dreal duv, dreal PA, int nd, const dreal *u, const dreal *v, const dreal *fobs_re,
                   const dreal *fobs_im, const dreal *weights);
dreal chi2_image(int nx, int ny, const dreal* image, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u, const dreal* v, const dreal* fobs_re, const dreal* fobs_im, const dreal* weights);
void sweep(int nr, dreal *const intensity, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal inc, dcomplex *image);
void uv_rotate(dreal PA, dreal dRA, dreal dDec, dreal* dRArot, dreal* dDecrot, int nd, const dreal* u, const dreal* v, dreal* urot, dreal* vrot);

/* Interface for the experts */
dcomplex* copy_input(int nx, int ny, const dreal* image);
void galario_free(void*);
void fft2d(int nx, int ny, dcomplex* image);
void fftshift(int nx, int ny, dcomplex* image);
void fftshift_axis0(int nx, int ny, dcomplex* matrix);
void interpolate(int nrow, int ncol, const dcomplex *image, int nd,
                 const dreal *u, const dreal *v, const dreal duv,
                 dcomplex *fint);
void apply_phase_sampled(dreal dRA, dreal dDec, int nd, const dreal* u,
                         const dreal* v, dcomplex* fint);
void reduce_chi2(int nd, const dreal* fobs_re, const dreal* fobs_im,
                 const dcomplex* fint, const dreal* weights, dreal* chi2);

/* Required for multithreading */
void init();
void cleanup();
int threads(int num = 0);

/* GPU related functions */
int ngpus();
void use_gpu(int device_id);
}
