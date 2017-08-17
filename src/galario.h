#pragma once

#include "galario_defs.h"

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */
    /* Main user functions */
    void galario_sample_profile(int nr, const dreal* const ints, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal dist, dreal inc,
                                dreal dRA, dreal dDec, dreal duv, int nd, const dreal *u, const dreal *v, dcomplex *fint);
    void galario_sample_image(int nx, int ny, const dreal* data, dreal dRA, dreal dDec, dreal duv, int nd, const dreal* u, const dreal* v, dcomplex* fint);
    void galario_chi2_profile(int nr, const dreal* const ints, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal dist, dreal inc,
                              dreal dRA, dreal dDec, dreal duv, int nd, const dreal *u, const dreal *v,
                              const dreal* fobs_re, const dreal* fobs_im, const dreal* weights, dreal* chi2);
    void galario_chi2_image(int nx, int ny, const dreal* data, dreal dRA, dreal dDec, dreal duv, int nd, const dreal* u, const dreal* v, const dreal* fobs_re, const dreal* fobs_im, const dreal* weights, dreal* chi2);
    void galario_sweep(int nr, const dreal* ints, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal inc, dcomplex* image);

    /* Interface for the experts */
    dcomplex* galario_copy_input(int nx, int ny, const dreal* realdata);
    void galario_free(void*);
    void galario_fft2d(int nx, int ny, dcomplex* data);
    void galario_fftshift(int nx, int ny, dcomplex* data);
    void galario_fftshift_axis0(int nx, int ny, dcomplex* matrix);
    void galario_interpolate(int nrow, int ncol, const dcomplex *data, int nd, const dreal *u, const dreal *v,
                             const dreal duv, dcomplex *fint);
    void galario_apply_phase_sampled(dreal dRA, dreal dDec, int nd, const dreal* u, const dreal* v, dcomplex* fint);
    void galario_reduce_chi2(int nd, const dreal* fobs_re, const dreal* fobs_im, const dcomplex* fint, const dreal* weights, dreal* chi2);

    /* Required for multithreading */
    void galario_init();
    void galario_cleanup();

    /* GPU related functions */
    int galario_threads_per_block(int num
#ifdef __cplusplus
                                          = 0
#endif
                                             );
    int galario_ngpus();
    void galario_use_gpu(int device_id);
#ifdef __cplusplus
}  /* extern "C" */
#endif /* __cplusplus */
