#pragma once

#include "galario_defs.h"

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */
    /* Main user functions */
    void galario_sample(int nx, int ny, dreal* data, dreal dRA, dreal dDec, dreal du, int nd, const dreal* u, const dreal* v, dcomplex* fint);
    void galario_chi2(int nx, int ny, dreal* data, dreal dRA, dreal dDec, dreal du, int nd, const dreal* u, const dreal* v, const dreal* fobs_re, const dreal* fobs_im, const dreal* weights, dreal* chi2);

    /* Interface for the experts */
    dcomplex* galario_fft2d(int nx, int ny, const dreal* data);
    void galario_fftshift(int nx, int ny, dreal* data);
    void galario_fftshift_axis0(int nx, int ny, dcomplex* matrix);
    void galario_interpolate(int nx, int ny, const dcomplex* data, int nd, const dreal* u, const dreal* v, dcomplex* fint);
    void galario_apply_phase_2d(int nx, int ny, dcomplex* data, dreal dRA, dreal dDec);
    void galario_apply_phase_sampled(dreal dRA, dreal dDec, int nd, const dreal* u, const dreal* v, dcomplex* fint);
    void galario_get_uv_idx(int nx, int ny, dreal du, int nd, const dreal* u, const dreal* v, const dreal* indu, dreal* indv);
    void galario_get_uv_idx_R2C(int nx, int ny, dreal du, int nd, const dreal* u, const dreal* v, dreal* indu, dreal* indv);
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
