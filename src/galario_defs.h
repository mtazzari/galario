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
