#pragma once

#ifdef __CUDACC__
    #include <cufft.h>
#else
    #include <complex>
#endif

#ifdef DOUBLE_PRECISION

    typedef double dreal;

    #ifdef __CUDACC__
        typedef cufftDoubleComplex dcomplex;
    #else
        typedef std::complex<dreal> dcomplex;
    #endif
#else

    typedef float dreal;

    #ifdef __CUDACC__
        typedef cufftComplex dcomplex;
    #else
        typedef std::complex<float> dcomplex;
    #endif
#endif
