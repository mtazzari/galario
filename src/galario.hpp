#pragma once

#ifdef __CUDACC__
    #include <cufft.h>
#endif

#ifdef DOUBLE_PRECISION

    typedef double dreal;

    #ifdef __CUDACC__
        typedef cufftDoubleComplex dcomplex;
    #else
        #include <complex>
        typedef std::complex<double> dcomplex;
    #endif // end __CUDACC__

#else // single precision

    typedef float dreal;

    #ifdef __CUDACC__
        typedef cufftComplex dcomplex;
    #else
        #include <complex>
        typedef std::complex<float> dcomplex;
    #endif // end __CUDACC__

#endif // end DOUBLE_PRECISION

//# TODO consistent notation dreal -> real_t, dcomplex -> complex_t
//# to avoid confusion with host vs. device

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */
    int nthreads(int x=0);
    void C_fft2d(int nx, void* data);
    void C_fftshift(int nx, void* data);
    void C_fftshift_fft2d_fftshift(int nx, void* data);
    void C_interpolate(int nx, void* data, int nd, void* u, void* v, void* fint);
    void C_apply_phase_2d(int nx, void* data, dreal x0, dreal y0);
    void C_acc_rotix(int nx, void* pixel_centers, int nd, void* u, void* v, void* indu, void* indv);
    void C_reduce_chi2(int nd, void* fobs_re, void* fobs_im, void* fint, void* weights, dreal* chi2);
    void C_chi2(int nx, void* data, dreal x0, dreal y0, void* vpixel_centers, int nd, void* u, void* v, void* fobs_re, void* fobs_im, void* weights, dreal* chi2);
    void C_acc_init();
    void C_acc_cleanup();

    int C_ngpus();
    void C_use_gpu(int device_id);
#ifdef __cplusplus
}  /* extern "C" */
#endif /* __cplusplus */
