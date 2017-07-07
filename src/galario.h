#pragma once

#include "galario_defs.h"

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */
    /* Main user functions */
    void galario_sample(int nx, dreal* data, dreal dRA, dreal dDec, dreal du, int nd, dreal* u, dreal* v, dcomplex* fint);
    void galario_chi2(int nx, dreal* data, dreal dRA, dreal dDec, dreal du, int nd, dreal* u, dreal* v, dreal* fobs_re, dreal* fobs_im, dreal* weights, dreal* chi2);

    /* Interface for the experts */
    void galario_fft2d(int nx, dcomplex* data);
    void galario_fftshift(int nx, dcomplex* data);
    void galario_fftshift_fft2d_fftshift(int nx, dcomplex* data);
    void galario_interpolate(int nx, dcomplex* data, int nd, dreal* u, dreal* v, dcomplex* fint);
    void galario_apply_phase_2d(int nx, dcomplex* data, dreal dRA, dreal dDec);
    void galario_apply_phase_sampled(dreal dRA, dreal dDec, int nd, dreal* u, dreal* v, dcomplex* fint);
    void galario_acc_rotix(int nx, dreal du, int nd, dreal* u, dreal* v, dreal* indu, dreal* indv);
    void galario_reduce_chi2(int nd, dreal* fobs_re, dreal* fobs_im, dcomplex* fint, dreal* weights, dreal* chi2);

    /* Required for multithreading */
    void galario_acc_init();
    void galario_acc_cleanup();

    /* GPU related functions */
    int galario_threads_per_block(int num=0);
    int galario_ngpus();
    void galario_use_gpu(int device_id);
#ifdef __cplusplus
}  /* extern "C" */
#endif /* __cplusplus */
