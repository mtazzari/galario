#pragma once

#ifdef __CUDACC__
    #include <cufft.h>
#elif __cplusplus
    #include <complex>
#else
    #include <complex.h>
#endif

#ifdef DOUBLE_PRECISION

    typedef double dreal;

    #ifdef __CUDACC__
        typedef cufftDoubleComplex dcomplex;
    #elif __cplusplus
        typedef std::complex<dreal> dcomplex;
    #else
        typedef double complex dcomplex;
    #endif
#else

    typedef float dreal;

    #ifdef __CUDACC__
        typedef cufftComplex dcomplex;
    #elif __cplusplus
        typedef std::complex<float> dcomplex;
    #else
        typedef float complex dcomplex;
    #endif
#endif
