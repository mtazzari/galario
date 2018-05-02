#pragma once

#include "galario_defs.h"

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */
    /* Main user functions */
    void galario_sample_profile(int nr, dreal *const intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA,
                                dreal dDec, dreal duv, dreal PA, int nd, const dreal *u, const dreal *v, dcomplex *fint);
    void galario_sample_image(int nx, int ny, const dreal* image, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u, const dreal* v, dcomplex* fint);
    dreal galario_chi2_profile(int nr, dreal *const intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA,
                                  dreal dDec, dreal duv, dreal PA, int nd, const dreal *u, const dreal *v, const dreal *fobs_re,
                                  const dreal *fobs_im, const dreal *weights);
    dreal galario_chi2_image(int nx, int ny, const dreal* image, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u, const dreal* v, const dreal* fobs_re, const dreal* fobs_im, const dreal* weights);
    void galario_sweep(int nr, dreal *const intensity, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal inc, dcomplex *image);
    void galario_uv_rotate(dreal PA, dreal dRA, dreal dDec, dreal* dRArot, dreal* dDecrot, int nd, const dreal* u, const dreal* v, dreal* urot, dreal* vrot);

    /* Interface for the experts */
    dcomplex* galario_copy_input(int nx, int ny, const dreal* image);
    void galario_free(void*);
    void galario_fft2d(int nx, int ny, dcomplex* image);
    void galario_fftshift(int nx, int ny, dcomplex* image);
    void galario_fftshift_axis0(int nx, int ny, dcomplex* matrix);
    void galario_interpolate(int nrow, int ncol, const dcomplex *image, int nd, const dreal *u, const dreal *v,
                             const dreal duv, dcomplex *fint);
    void galario_apply_phase_sampled(dreal dRA, dreal dDec, int nd, const dreal* u, const dreal* v, dcomplex* fint);
    dreal galario_reduce_chi2(int nd, const dreal* fobs_re, const dreal* fobs_im, const dcomplex* fint, const dreal* weights);

    /* Required for multithreading */
    void galario_init();
    void galario_cleanup();
    int galario_threads(int num
#ifdef __cplusplus
                                = 0
#endif
                                   );

    /* GPU related functions */
    int galario_ngpus();
    void galario_use_gpu(int device_id);
#ifdef __cplusplus
}  /* extern "C" */
#endif /* __cplusplus */
