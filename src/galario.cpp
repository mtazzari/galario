/******************************************************************************
* This file is part of GALARIO:                                               *
* Gpu Accelerated Library for Analysing Radio Interferometer Observations     *
*                                                                             *
* Copyright (C) 2017-2018, Marco Tazzari, Frederik Beaujean, Leonardo Testi.  *
*                                                                             *
* This program is free software: you can redistribute it and/or modify        *
* it under the terms of the Lesser GNU General Public License as published by *
* the Free Software Foundation, either version 3 of the License, or           *
* (at your option) any later version.                                         *
*                                                                             *
* This program is distributed in the hope that it will be useful,             *
* but WITHOUT ANY WARRANTY; without even the implied warranty of              *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                        *
*                                                                             *
* For more details see the LICENSE file.                                      *
* For documentation see https://mtazzari.github.io/galario/                   *
******************************************************************************/

#include "galario.h"
#include "galario_py.h"

// full function makes code hard to read
#define tpb galario::threads()

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cstring>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <sstream>

using std::to_string;

#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>

#include <cublas_v2.h>
#include <cufft.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>

#else // CPU
// general min function already available in cuda
// math_functions.hpp. Need `using` so the right implementation of
// `min` is chosen for the kernels that are both on gpu and cpu
#include <algorithm>
using std::min;
using std::max;

#include <fftw3.h>

#endif

// Stuff needed for GPU and CPU but should not be visible any other translation unit so we can use very common names.
namespace {
    /**
     * Provide a string buffer to avoid overhead from calling std::cout repeatedly.
     */
    std::ostringstream& out(bool reset=false) {
        static std::ostringstream my_stream;

        // insert a newline only if my_stream is empty
        if (!my_stream.tellp())
            my_stream.put('\n');

        if (reset) {
            my_stream.str("\n");
            my_stream.clear();
        }
        return my_stream;
    }

    void flush_timing() {
        // if nothing but the initial newline in there, nothing to show
        if (out().tellp() > 1)
            std::cout << out().str() << std::flush;
        // empty the stream
        out(true);
    }

    template <class T = std::runtime_error>
    void throw_exception(const char *file, const int line, const char* source, const std::string& msg) {
        std::stringstream ss;
        ss << file << ":" << line << ":\n";
        ss << "Error in " << source << ": " << msg;

        throw T(ss.str());
    }

   /**
    * Macros to check input image lengths.
    */
    #define CHECK_INPUT(nx) \
    do { \
        if (nx < 2) { throw_exception<std::invalid_argument>(__FILE__, __LINE__, "check input image", "x dimension = " + to_string(nx) + " is less than 2"); } \
        if (nx % 2 != 0) { throw_exception<std::invalid_argument>(__FILE__, __LINE__, "check input image", "x dimension = " + to_string(nx) + " is odd"); } \
    } while (0)

    #define CHECK_INPUTXY(nx, ny) \
    do { \
        if (nx != ny) { throw_exception<std::invalid_argument>(__FILE__, __LINE__, "check input image", "Expect a square image but got shape (" + to_string(nx) + ", " + to_string(ny) + ")"); } \
        CHECK_INPUT(nx); \
    } while (0)

#define CHECK_CENTRAL_PIXEL(dxy, Rmin, dR) \
    do { \
    const dreal ratio_central_pixel = (dxy / 2. - Rmin) / dR; \
    if (ratio_central_pixel < 5) { \
        throw_exception<std::invalid_argument>(__FILE__, __LINE__, "create image", \
                                               "Expect (dxy/2-Rmin)/dR > 5, i.e. dR small enough to ensure reliable interpolation the inside central pixel but got (dxy/2-Rmin)/dR = " \
                                               + to_string(ratio_central_pixel) + ". Try reducing dR."); \
    } \
    } while (0)

#if defined(_OPENMP) && defined(GALARIO_TIMING)
    struct CPUTimer {
        double start;

        CPUTimer() {
            Start();
        }

        void Start() {
            start = omp_get_wtime();
        }

        void Elapsed(const std::string& msg) {
            const double elapsed = 1000 * (omp_get_wtime() - start);
            ::out() << "[CPU] " << msg << ": " << elapsed << " ms\n";
            // reset the timer for the next use
            Start();
        }
    };

    #define OPENMPTIME(body, msg)                                     \
    do {                                                              \
        CPUTimer t;                                                   \
        body;                                                         \
        t.Elapsed(msg);                                               \
    } while (false)
#else
    #define OPENMPTIME(body, msg) body
    struct CPUTimer {
        void Elapsed(const std::string&) {}
    };
#endif // _OPENMP && TIMING

#ifdef __CUDACC__

    void throw_exception(const char *file, const int line, const char* source, const int err) {
        std::stringstream ss;
        ss << "Failed with error code " << err;
        throw_exception(file, line, source, ss.str());
    }

    #define CCheck(err) __cudaSafeCall((err), __FILE__, __LINE__)
    inline void __cudaSafeCall(cudaError err, const char *file, const int line)  {
        if (err == cudaErrorInitializationError) {
            throw_exception(file, line, "cuda", "Could not initialize cuda. Is a CUDA GPU available at all?");
        }
        if (err == cudaErrorMemoryAllocation) {
            throw std::bad_alloc();
        }
        if (cudaSuccess != err) {
            throw_exception(file, line, "cuda", cudaGetErrorString(err));
        }
    }

    #define CBlasCheck(err) __cublasSafeCall((err), __FILE__, __LINE__)
    inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line) {
        if (err == CUBLAS_STATUS_NOT_INITIALIZED) {
            throw_exception(file, line, "cublas", "Could not initialize cublas. Is a cuda GPU available at all? Or is it ouf memory?");
        }
        if (err == CUBLAS_STATUS_ALLOC_FAILED) {
            throw std::bad_alloc();
        }
        if (CUBLAS_STATUS_SUCCESS != err) {
            throw_exception(file, line, "cublas", err);
        }
    }

    #define CUFFTCheck(err) __cufftwSafeCall((err), __FILE__, __LINE__)
    inline void __cufftwSafeCall(cufftResult_t err, const char *file, const int line) {
        if (err == CUFFT_ALLOC_FAILED) {
            throw std::bad_alloc();
        }
       if (CUFFT_SUCCESS != err) {
           throw_exception(file, line, "cufftw", err);
       }
    }

    cublasHandle_t cublasHandle = nullptr;
    std::mutex cublasHandle_mutex;

    bool cublas_initialized() {
        return cublasHandle != nullptr;
    }

    void cublas_init() {
        // lock to prevent data race
        std::lock_guard<std::mutex> lock(cublasHandle_mutex);

        // check if handle initialized to avoid 2nd thread in race condition to initialize again
        if (cublas_initialized()) {
            return;
        }

        // actually init
        CBlasCheck(cublasCreate(&cublasHandle));
    }

    cublasHandle_t& cublas_handle() {
        if (!cublas_initialized()) {
            cublas_init();
        }
        return cublasHandle;
    }

    /**
     * A simple RAII wrapper around cuda memory for exception safety
     */
    template <typename T>
    struct CudaMemory {
        CudaMemory(size_t n) : nbytes(sizeof(T) * n) {
            const auto error = cudaMalloc(&ptr, nbytes);
            if (error != cudaSuccess) {
                // If this fails, it hides the first error from allocation
                CCheck(cudaFree(ptr));

                // safe to throw an error now, no memory dangling
                CCheck(error);
            }
        }

        /**
         * Allocate and copy `n` elements of type `T` from `source` to device
         */
        CudaMemory(size_t n, const T* source) : CudaMemory(n) {
            CCheck(cudaMemcpy(ptr, source, nbytes, cudaMemcpyHostToDevice));
        }

        // forbid copy operations to avoid double ownership
        CudaMemory(const CudaMemory&) = delete;
        CudaMemory& operator=(const CudaMemory&) = delete;

        // move operations transfer ownership
        CudaMemory(CudaMemory&&) = default;
        CudaMemory& operator=(CudaMemory&&) = default;

        // Should not throw an exception inside destructor, so we don't `CCheck`
        ~CudaMemory() {
            cudaFree(ptr);
        }

        /// Copy back from device to host destination
        void Retrieve(T* destination) {
            CCheck(cudaMemcpy(destination, ptr, nbytes, cudaMemcpyDeviceToHost));
        }

        /// The device pointer
        T* ptr;

        /// The size of the memory allocation
        const size_t nbytes;
    };

    #ifdef GALARIO_TIMING
        struct GPUTimer
        {
            cudaEvent_t start;
            cudaEvent_t stop;

            GPUTimer() {
                CCheck(cudaEventCreate(&start));
                CCheck(cudaEventCreate(&stop));
                Start();
            }

            ~GPUTimer() {
                CCheck(cudaEventDestroy(start));
                CCheck(cudaEventDestroy(stop));
            }

            void Start() {
                CCheck(cudaEventRecord(start, 0));
            }

            void Elapsed(const std::string& msg) {
                CCheck(cudaEventRecord(stop, 0));
                CCheck(cudaEventSynchronize(stop));
                float elapsed;
                CCheck(cudaEventElapsedTime(&elapsed, start, stop));
                ::out() << "[GPU] " << msg << ": " << elapsed << " ms\n";
                Start();
            }
        };
    #else
        struct GPUTimer
        {
            GPUTimer() {
                // call empty Start() just to avoid warning about unused function
                Start();
            }
            void Start() {}
            void Elapsed(const std::string& msg) {}
        };
    #endif // TIMING

    #ifdef DOUBLE_PRECISION
        #define CUFFTEXEC cufftExecD2Z
        #define CUFFTTYPE CUFFT_D2Z
        #define CMPLX(a, b) (make_cuDoubleComplex(a,b))
        #define CMPLXSUB cuCsub
        #define CMPLXADD cuCadd
        #define CMPLXMUL cuCmul
        #define CMPLXCONJ cuConj
        #define CUBLASNRM2 cublasDznrm2

    #else
        #define CUFFTEXEC cufftExecR2C
        #define CUFFTTYPE CUFFT_R2C
        #define CMPLX(a, b) (make_cuFloatComplex(a,b))
        #define CMPLXSUB cuCsubf
        #define CMPLXADD cuCaddf
        #define CMPLXMUL cuCmulf
        #define CMPLXCONJ cuConjf
        #define CUBLASNRM2 cublasScnrm2
    #endif  // DOUBLE_PRECISION

#else // CPU

    #define CMPLXSUB(a, b) ((a) - (b))
    #define CMPLXADD(a, b) ((a) + (b))
    #define CMPLXMUL(a, b) ((a) * (b))
    #define CMPLXCONJ conj
#endif // __CUDACC__

#ifdef DOUBLE_PRECISION
    #define SQRT sqrt
    #define FFTW(name) fftw_ ## name
#else
    #define SQRT sqrtf
    #define FFTW(name) fftwf_ ## name
#endif

} // anonymous namespace

namespace galario {
int threads(int num) {
#ifdef __CUDACC__
    // mynthreads^2 is used per block
    static int mynthreads = 16;
    // `num^2`: number of threads per block for 2D operations
    if (num > 0)
        mynthreads = num;
#else
    #if defined(_OPENMP)
        /* fix the number of openmp threads. disabling dynamic to respect the user's
           wish as much as possible */
        static int mynthreads = omp_get_max_threads();
        if (num > 0) {
            mynthreads = num;
            omp_set_dynamic(0);
            omp_set_num_threads(num);
        }
    #else
        // no threads, `num` ignored
        static int mynthreads = 1;
    #endif
#endif
    return mynthreads;
}

void init() {
#ifdef __CUDACC__
    // Avoid initializing cublas unconditionally. It takes a lot of memory and
    // fails if cuda is not available. Let the initialization be done only if
    // cublas is actually needed.
    // cublas_handle();
#else
    #ifdef _OPENMP
    const int status = FFTW(init_threads)();
    if (status == 0) {
        throw_exception(__FILE__, __LINE__, "fftw", "fftw_init_threads() failed");
    }
    #endif
#endif
}

void cleanup() {
#ifdef __CUDACC__
    if (cublas_initialized()) {
        CBlasCheck(cublasDestroy(cublas_handle()));
    }
#else
    #ifdef _OPENMP
    FFTW(cleanup_threads)();
    #endif
    FFTW(cleanup)();
#endif
}

void galario_free(void* data) {
#ifdef __CUDACC__
    free(data);
#else
    fftw_free(data);
#endif
}
}

#ifdef __CUDACC__
/**
 * Return complex image on the device made from real image with array size `nx*ny` on the host.
 *
 */
CudaMemory<dcomplex> copy_input_d(int nx, int ny, const dreal* realdata) {
    GPUTimer t;
    auto const ncol = ny/2+1;
    auto const rowsize_real = sizeof(dreal)*ny;
    auto const rowsize_complex = sizeof(dcomplex)*ncol;

    // create destination array
    CudaMemory<dcomplex> data_d(nx * ncol);

    // set the padding by defining different sizes of a row in bytes
    CCheck(cudaMemcpy2D(data_d.ptr, rowsize_complex, realdata, rowsize_real, rowsize_real, nx, cudaMemcpyHostToDevice));
    t.Elapsed("copy_input_H->D");
    return data_d;
}
#endif

namespace galario {
/**
 * Copy an (nx, ny) square image into a complex buffer for real-to-complex FFTW.
 *
 * Buffer ownership transferred to caller, use `galario_free(buffer)`.
 *
 * If turns out to be slow have a look here:
 *   https://stackoverflow.com/questions/19601696/what-is-the-fastest-do-array-padding-of-the-image-array
 */
dcomplex* copy_input(int nx, int ny, const dreal* realdata) {
    CHECK_INPUTXY(nx, ny);
    // in r2c, the last dimension only has ~half the size
    auto const ncol = ny/2 + 1;

#ifdef __CUDACC__
    auto buffer = static_cast<dcomplex*>(malloc(sizeof(dcomplex)*nx*ncol));
#else
    // fftw_alloc for aligned memory to use SIMD acceleration
    auto buffer = reinterpret_cast<dcomplex*>(FFTW(alloc_complex)(nx*ncol));
#endif

    // copy and respect padding in last dimension. Treating the complex output
    // buffer as a sequence of real entries, the last (nx odd) or last two
    // columns (nx even) have to be skipped when copying in the input
    auto real_buffer = reinterpret_cast<dreal*>(buffer);

    // #reals = 2*#complex
    auto const rowsize = 2*ncol;

    // copy over entire input rows to output array
    auto const nbytes = sizeof(dreal)*ny;
#pragma omp parallel for shared(real_buffer, realdata)
    for (int i = 0; i < nx; ++i) {
       std::memcpy(&real_buffer[i*rowsize], &realdata[i*ny], nbytes);
    }
    return buffer;
}

void* _copy_input(int nx, int ny, void* realdata) {
    return copy_input(nx, ny, static_cast<dreal*>(realdata));
}
}

#ifdef __CUDACC__
void fft_d(int nx, int ny, dcomplex* data_d) {
     cufftHandle plan;

     /* Create a 2D FFT plan and execute it. */
     // TODO: find a way to store the plan
     CUFFTCheck(cufftPlan2d(&plan, nx, ny, CUFFTTYPE));
     CUFFTCheck(CUFFTEXEC(plan, reinterpret_cast<dreal*>(data_d), data_d));

     // cufft calls are asynchronous but in default stream
     CCheck(cudaDeviceSynchronize());
     CUFFTCheck(cufftDestroy(plan));
}
#else
/**
 * Requires `data` to be large enough to hold the complex output after an
 * in-place transform, and the real input has to in the right memory locations
 * respecting the padding in the last dimension; see
 * http://fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html
 */
void fft_h(int nx, int ny, dcomplex* data) {
    dreal* input = reinterpret_cast<dreal*>(data);
    FFTW(complex)* output = reinterpret_cast<FFTW(complex)*>(data);
#ifdef _OPENMP
    fftw_plan_with_nthreads(galario::threads());
#endif
    FFTW(plan) p = FFTW(plan_dft_r2c_2d)(nx, ny, input, output, FFTW_ESTIMATE);
    FFTW(execute)(p);

    // TODO: find a way to store the plan (maybe homogeneously with the cuFFTPlan
    FFTW(destroy_plan)(p);
}
#endif

namespace galario {
/**
 * `data`: nx * nx matrix
 * output: a buffer in the format described at http://fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data. It needs to be freed by `fftw_free`, not the ordinary `free`!
 */
void fft2d(int nx, int ny, dcomplex* data) {
    CHECK_INPUTXY(nx, ny);
#ifdef __CUDACC__
    CudaMemory<dcomplex> data_d(nx*(ny/2 + 1), data);
    fft_d(nx, ny, data_d.ptr);
    data_d.Retrieve(data);
#else
    fft_h(nx, ny, data);
#endif
}

void _fft2d(int nx, int ny, void* data) {
    fft2d(nx, ny, static_cast<dcomplex*>(data));
}
}

/**
 * Shift quadrants of the square image. Swap the upper-left quadrant with the
 * lower-right quadrant and the upper-right with the lower-left quadrant.
 *
 * We work on an array of real numbers stored inside a complex array, FFTW in-place format.
 *
 * To avoid if statements, we do two swaps.
 *
 * For cache efficiency, may have to do loop tiling; i.e., the source and target
 * should fit into the cache. If the image is too large, only part of a row may
 * fit. This is a responsibility of the caller.
 **/
// `a` is a matrix (size: nx*ny)
#ifdef __CUDACC__
__host__ __device__
#endif
inline void shift_core(int const idx_x, int const idx_y, int const nx, int const ny, dreal* const __restrict__ a) {
    /* row-wise access */

    // number of real elements in the complex row of length ny/2+1
    auto const rowsize = 2*(ny/2+1);

    /* from upper left to lower right and from upper right to lower left */
    auto const src_ul = idx_x * rowsize + idx_y;
    auto const src_ur = src_ul + ny/2;

    // half the rows down
    auto const tgt_ul = src_ur + nx/2 * rowsize;

    // half a column to the left
    auto const tgt_ur = tgt_ul - ny/2;

    /* swap the values */
    auto tmp = a[src_ul];
    a[src_ul] = a[tgt_ul];
    a[tgt_ul] = tmp;

    tmp = a[src_ur];
    a[src_ur] = a[tgt_ur];
    a[tgt_ur] = tmp;
}

/**
 * grid stride loop
 */
#ifdef __CUDACC__
__global__ void shift_d(const int nx, const int ny, dcomplex* const __restrict__ data) {

    dreal* a = reinterpret_cast<dreal*>(data);

    // indices
    int const x0 = blockDim.x * blockIdx.x + threadIdx.x;
    int const y0 = blockDim.y * blockIdx.y + threadIdx.y;

    // stride
    int const sx = blockDim.x * gridDim.x;
    int const sy = blockDim.y * gridDim.y;

    for (auto x = x0; x < nx/2; x += sx) {
        for (auto y = y0; y < ny/2; y += sy) {
            shift_core(x, y, nx, ny, a);
        }
    }
}
#else

void shift_h(int const nx, int const ny, dcomplex* const __restrict__ data) {
   dreal* a = reinterpret_cast<dreal*>(data);
#pragma omp parallel for
    for (auto x = 0; x < nx/2; ++x) {
        for (auto y = 0; y < ny/2; ++y) {
            shift_core(x, y, nx, ny, a);
        }
    }
}
#endif

namespace galario {
void fftshift(int nx, int ny, dcomplex* data) {
    CHECK_INPUTXY(nx, ny);
#ifdef __CUDACC__
    CudaMemory<dcomplex> data_d(nx*(ny/2+1), data);

    shift_d<<<dim3(nx/2/tpb+1, ny/2/tpb+1), dim3(tpb, tpb)>>>(nx, ny, data_d.ptr);

    CCheck(cudaDeviceSynchronize());
    data_d.Retrieve(data);
#else
    shift_h(nx, ny, data);
#endif
}

void _fftshift(int nx, int ny, void* data) {
    fftshift(nx, ny, static_cast<dcomplex*>(data));
}
}

/**
 * Shift quadrants of a rectangular matrix of size (nrow, ncol).
 * Swap the upper quadrant with the lower quadrant.
 *
 * For cache efficiency, may have to do loop tiling; i.e., the source and target
 * should fit into the cache. If the image is too large, only part of a row may
 * fit. This is a responsibility of the caller.
 **/
#ifdef __CUDACC__
__host__ __device__
#endif
inline void shift_axis0_core(int const idx_x, int const idx_y, int const nrow, int const ncol, dcomplex* const __restrict__ matrix) {
    /* row-wise access */

    // from top-half to bottom-half
    auto const src_u = idx_x*ncol + idx_y;
    auto const tgt_u = src_u + nrow/2*ncol;

    // swap the values
    auto tmp = matrix[src_u];
    matrix[src_u] = matrix[tgt_u];
    matrix[tgt_u] = tmp;
}

/**
 * grid stride loop
 */
#ifdef __CUDACC__
__global__ void shift_axis0_d(int const nrow, int const ncol, dcomplex* const __restrict__ matrix) {
    // indices
    int const x0 = blockDim.x * blockIdx.x + threadIdx.x;
    int const y0 = blockDim.y * blockIdx.y + threadIdx.y;

    // stride
    int const sx = blockDim.x * gridDim.x;
    int const sy = blockDim.y * gridDim.y;

    for (auto x = x0; x < nrow/2; x += sx) {
        for (auto y = y0; y < ncol; y += sy) {
            shift_axis0_core(x, y, nrow, ncol, matrix);
        }
    }
}
#else

void shift_axis0_h(int const nrow, int const ncol, dcomplex* const __restrict__ matrix) {
#pragma omp parallel for
    for (auto x = 0; x < nrow/2; ++x) {
        for (auto y = 0; y < ncol; ++y) {
            shift_axis0_core(x, y, nrow, ncol, matrix);
        }
    }
}
#endif

namespace galario {
void fftshift_axis0(int nrow, int ncol, dcomplex* matrix) {
    CHECK_INPUT(nrow);
#ifdef __CUDACC__
    CudaMemory<dcomplex> matrix_d(nrow * ncol, matrix);

    shift_axis0_d<<<dim3(nrow/2/tpb+1, ncol/tpb+1), dim3(tpb, tpb)>>>(nrow, ncol, matrix_d.ptr);

    CCheck(cudaDeviceSynchronize());
    matrix_d.Retrieve(matrix);
#else
    shift_axis0_h(nrow, ncol, matrix);
#endif
}

void _fftshift_axis0(int nrow, int ncol, void* matrix) {
    fftshift_axis0(nrow, ncol, static_cast<dcomplex*>(matrix));
}
}

/**
 * Bilinear interpolation in 2D according to Numerical Recipes.
 *
 * Interpolation of a matrix `data` in the generic point (u, v).
 *
 *     fint(u, v) = (1-t)(1-q)y0 + t(1-q)y1 + t*q*y2 + (1-t)*q*y3
 *                = t*q*(y0-y1+y2-y3) + t(-y0+y1) + q(-y0 + y3) + y0
 *
 * `y0` is bottom-left grid point, `y1` the bottom-right etc. forming a
 * a grid square around (u, v), ordered counter-clockwise.
 * `q` and `t` are the fractions of the desired location from left (bottom)
 * to right (upper) grid point.
 *
 * @param nrow, ncol : shape of the matrix `data`
 * @param data : complex 2D matrix containing the Real to Complex transform of an input image
 * @param u : x-axis coordinate of the point where `data` has to be interpolated
 * @param v : y-axis coordinate of the point where `data` has to be interpolated
 * @param duv : pixel size in the Fourier space, assumed to be uniform and the same in u and v direction
 * @returns: the interpolated point.
 *
 * Notes
 * The u and v coordinate axes follow the convention for radio interferometry for which
 * an input image has Right Ascension (x axis) increasing from Right to Left, and Declination
 * (y axis) increasing from Bottom to Top. u and v are parallel to Right Ascension and Declination, respectively.
 */
#ifdef __CUDACC__
__host__ __device__
#endif
inline dcomplex interpolate_core(int const nrow, int const ncol, const dcomplex *const data, dreal const v_origin,
                                 const dreal u, const dreal v, const dreal duv) {

    const int half_nrow = nrow/2;

    // compute indices
    dreal const indu = fabs(u)/duv;
    dreal indv;  // also indv is const

    // could be shorter: half_nx + sign(u)*v/duv;
    if (u < 0.) {
        indv = half_nrow - v_origin * v/duv;
    }
    else {
        indv = half_nrow + v_origin * v/duv;
    }

    // notations as in (3.6.5) of Numerical Recipes. They put the origin in the
    // lower-left.
    int const fl_u = floor(indu);
    int const fl_v = floor(indv);
    dcomplex const t = {indv - fl_v, 0.0};
    dcomplex const q = {indu - fl_u, 0.0};

    // linear index of y0
    int const base = fl_u + fl_v * ncol;

    /* the four grid points around the target point */
    const dcomplex& y0 = data[base];
    const dcomplex& y1 = data[base + ncol];
    const dcomplex& y2 = data[base + ncol + 1];
    const dcomplex& y3 = data[base + 1];

    /* ~ t*q */
    dcomplex const add1 = CMPLXADD(y0, y2);
    dcomplex const add2 = CMPLXADD(y1, y3);
    dcomplex const df1 = CMPLXSUB(add1, add2);
    dcomplex const mul1 = CMPLXMUL(q, df1);
    dcomplex const term1 = CMPLXMUL(t, mul1);

    /* ~ t */
    dcomplex const term2_sub = CMPLXSUB(y1, y0);
    dcomplex const term2 = CMPLXMUL(t, term2_sub);

    /* ~ q */
    dcomplex const term3_sub = CMPLXSUB(y3, y0);
    dcomplex const term3 = CMPLXMUL(q, term3_sub);

    /* add up all 4 terms */
    dcomplex const final_add2 = CMPLXADD(term2, term3);
    dcomplex const final_add1 = CMPLXADD(term1, final_add2);

    dcomplex phaseInterp = CMPLXADD(final_add1, y0);

    if (u < 0.) {
        phaseInterp = CMPLXCONJ(phaseInterp);
    }

    dreal angleInterp = arg(phaseInterp);

    dreal const tr = indv - fl_v;
    dreal const qr = indu - fl_u;

    dreal const y0r = abs(y0);
    dreal const y1r = abs(y1);
    dreal const y2r = abs(y2);
    dreal const y3r = abs(y3);

    dreal ampInterp = y0r;
    ampInterp += (y3r-y0r)*qr;
    ampInterp += (y1r-y0r)*tr;
    ampInterp += (y0r-y1r+y2r-y3r)*tr*qr;

    dcomplex interpolated = dcomplex{ampInterp*dreal(cos(angleInterp)), ampInterp*dreal(sin(angleInterp))};

    return interpolated;
    
}

#ifdef __CUDACC__
__global__ void interpolate_d(int const nrow, int const ncol, const dcomplex* const __restrict__ data, dreal const v_origin, int const nd, const dreal* const u, const dreal* const v, dreal const duv, dcomplex* const __restrict__ fint)
{
    //index
    int const idx_0 = blockDim.x * blockIdx.x + threadIdx.x;

    // stride
    int const sx = blockDim.x * gridDim.x;

    for (auto idx = idx_0; idx < nd; idx += sx) {
        fint[idx] = interpolate_core(nrow, ncol, data, v_origin, u[idx], v[idx], duv);
    }
}
#else

void interpolate_h(int const nrow, int const ncol, const dcomplex* const data, dreal const v_origin, int const nd, const dreal* const u, const dreal* const v, dreal const duv, dcomplex* fint) {

#pragma omp parallel for
    for (auto idx = 0; idx < nd; ++idx) {
        fint[idx] = interpolate_core(nrow, ncol, data, v_origin, u[idx], v[idx], duv);
    }
}
#endif

namespace galario {
void interpolate(int nrow, int ncol, const dcomplex *data, dreal const v_origin, int nd, const dreal *u, const dreal *v,
                         const dreal duv, dcomplex *fint) {

#ifdef __CUDACC__
    // copy the image data
    CudaMemory<dcomplex> data_d(nrow * ncol, data);

    // copy u,v and reserve memory for the interpolated values
    CudaMemory<dreal> u_d(nd, u);
    CudaMemory<dreal> v_d(nd, v);
    CudaMemory<dcomplex> fint_d(nd);

    // oversubscribe blocks because we don't know if #(data points) divisible by nthreads
    auto const nthreads = tpb * tpb;
    interpolate_d<<<nd / nthreads + 1, nthreads>>>(nrow, ncol, data_d.ptr, v_origin, nd, u_d.ptr, v_d.ptr, duv, fint_d.ptr);

    CCheck(cudaDeviceSynchronize());

    // retrieve interpolated values
    fint_d.Retrieve(fint);
#else
    interpolate_h(nrow, ncol, data, v_origin, nd, u, v, duv, fint);
#endif
}

void _interpolate(int nrow, int ncol, void *data, dreal v_origin, int nd, void *u, void *v, dreal duv, void *fint) {
    interpolate(nrow, ncol, static_cast<dcomplex*>(data), v_origin, nd, static_cast<dreal*>(u),
                        static_cast<dreal*>(v), duv, static_cast<dcomplex*>(fint));
}
}

// APPLY_PHASE TO SAMPLED POINTS //
#ifdef __CUDACC__
__host__ __device__
#endif
inline void apply_phase_sampled_core(int const idx_x, const dreal* const u, const dreal* const v, dcomplex* const __restrict__ fint, dreal const dRA, dreal const dDec) {

    dreal const angle = u[idx_x]*dRA + v[idx_x]*dDec;

    dcomplex const phase = dcomplex{dreal(cos(angle)), dreal(sin(angle))};

    fint[idx_x] = CMPLXMUL(fint[idx_x], phase);
}

#ifdef __CUDACC__
__global__ void apply_phase_sampled_d(dreal dRA, dreal dDec, int const nd, const dreal* const u, const dreal* const v, dcomplex* __restrict__ fint) {

    if ((dRA==0.) && (dDec==0.)) {
        return;
    }

    dRA *= 2.*(dreal)M_PI;
    dDec *= 2.*(dreal)M_PI;

    //index
    int const idx_x0 = blockDim.x * blockIdx.x + threadIdx.x;

    // stride
    int const sx = blockDim.x * gridDim.x;

    for (auto x = idx_x0; x < nd; x += sx) {
        apply_phase_sampled_core(x, u, v, fint, dRA, dDec);
    }
}
#else

void apply_phase_sampled_h(dreal dRA, dreal dDec, int const nd, const dreal* const u, const dreal* const v, dcomplex* const __restrict__ fint) {

    if ((dRA==0.) && (dDec==0.)) {
        return;
    }

    dRA *= 2.*(dreal)M_PI;
    dDec *= 2.*(dreal)M_PI;

#pragma omp parallel for shared(dRA, dDec) schedule(static)
    for (auto x = 0; x < nd; ++x) {
        apply_phase_sampled_core(x, u, v, fint, dRA, dDec);
    }
}
#endif

namespace galario {
void apply_phase_sampled(dreal dRA, dreal dDec, int const nd, const dreal* const u, const dreal* const v, dcomplex* const __restrict__ fint) {
#ifdef __CUDACC__

     CudaMemory<dreal> u_d(nd, u);
     CudaMemory<dreal> v_d(nd, v);
     CudaMemory<dcomplex> fint_d(nd, fint);

     auto const nthreads = tpb * tpb;
     apply_phase_sampled_d<<<nd/nthreads+1, nthreads>>>(dRA, dDec, nd, u_d.ptr, v_d.ptr, fint_d.ptr);

     CCheck(cudaDeviceSynchronize());
     fint_d.Retrieve(fint);
#else
    apply_phase_sampled_h(dRA, dDec, nd, u, v, fint);
#endif
}

void _apply_phase_sampled(dreal dRA, dreal dDec, int nd, void* const u,
                                  void* const v, void* __restrict__ fint) {
    apply_phase_sampled(dRA, dDec, nd, static_cast<dreal*>(u),
                                static_cast<dreal*>(v), static_cast<dcomplex*>(fint));
}
}

/**
 * Rotates the RA, Dec offsets and the u and v coordinates by Position Angle PA
 */
#ifdef __CUDACC__
__host__ __device__
#endif
inline void uv_rotate_core(dreal cos_PA, dreal sin_PA, const dreal u, const dreal v, dreal& urot, dreal& vrot) {

    urot = u * cos_PA - v * sin_PA;
    vrot = u * sin_PA + v * cos_PA;

}

#ifdef __CUDACC__
__global__ void uv_rotate_d(dreal cos_PA, dreal sin_PA, int const nd, const dreal* const u, const dreal* const v, dreal* const urot, dreal* vrot) {
    //index
    int const idx_x0 = blockDim.x * blockIdx.x + threadIdx.x;

    // stride
    int const sx = blockDim.x * gridDim.x;

    for (auto i = idx_x0; i < nd; i += sx) {
        uv_rotate_core(cos_PA, sin_PA, u[i], v[i], urot[i], vrot[i]);
    }
}
#else

void uv_rotate_h(dreal PA, dreal dRA, dreal dDec, dreal* dRArot, dreal* dDecrot, int const nd, const dreal* const u, const dreal* const v,
                 dreal* const urot, dreal* vrot) {
    CPUTimer t;

    if (PA==0.) {
        *dRArot = dRA;
        *dDecrot = dDec;
        memcpy(urot, u, sizeof(dreal)*nd);
        memcpy(vrot, v, sizeof(dreal)*nd);
        return;
    }

    const dreal cos_PA = cos(PA);
    const dreal sin_PA = sin(PA);

#pragma omp parallel for
    for (auto i = 0; i < nd; ++i) {
        uv_rotate_core(cos_PA, sin_PA, u[i], v[i], urot[i], vrot[i]);
    }

    uv_rotate_core(cos_PA, sin_PA, dRA, dDec, *dRArot, *dDecrot);

    t.Elapsed("uv_rotate_h");
}
#endif

namespace galario {

void uv_rotate(dreal PA, dreal dRA, dreal dDec, dreal* dRArot, dreal* dDecrot, int const nd, const dreal* const u, const dreal* const v,
                       dreal* const urot, dreal* const vrot) {
#ifdef __CUDACC__
     CudaMemory<dreal> u_d(nd, u);
     CudaMemory<dreal> v_d(nd, v);

     CudaMemory<dreal> urot_d(nd);
     CudaMemory<dreal> vrot_d(nd);

     if (PA==0.) {
        *dRArot = dRA;
        *dDecrot = dDec;
        cudaMemcpy(urot_d.ptr, u_d.ptr, u_d.nbytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(vrot_d.ptr, v_d.ptr, v_d.nbytes, cudaMemcpyDeviceToDevice);
     } else {
        const dreal cos_PA = cos(PA);
        const dreal sin_PA = sin(PA);

        auto const nthreads = tpb * tpb;
        uv_rotate_d<<<nd/nthreads +1, nthreads>>>(cos_PA, sin_PA, nd, u_d.ptr, v_d.ptr, urot_d.ptr, vrot_d.ptr);
        uv_rotate_core(cos_PA, sin_PA, dRA, dDec, *dRArot, *dDecrot);
     }
     CCheck(cudaDeviceSynchronize());
     urot_d.Retrieve(urot);
     vrot_d.Retrieve(vrot);
#else
    uv_rotate_h(PA, dRA, dDec, dRArot, dDecrot, nd, u, v, urot, vrot);
#endif
}

void _uv_rotate(dreal PA, dreal dRA, dreal dDec, void* dRArot, void* dDecrot, int nd, void* const u,
                                  void* const v, void* const urot, void* const vrot) {
    uv_rotate(PA, dRA, dDec, static_cast<dreal*>(dRArot), static_cast<dreal*>(dDecrot), nd, static_cast<dreal*>(u),
                                static_cast<dreal*>(v), static_cast<dreal*>(urot), static_cast<dreal*>(vrot));
}
}

/**
 * Sweep.
 * TODO avoid rmax 3 definitions. pass rmax (perhaps also base as argument of sweep_core.
 **/
#ifdef __CUDACC__
__host__ __device__
#endif
inline void sweep_core(int const i, int const j, int const nr, const dreal* const intensity,
                       dreal const Rmin, dreal const dR, const int rmax, int const nxy, int const rowsize,
                       dreal const dxy, dreal const cos_inc, dreal const sr_to_px,  dreal* const __restrict__ image) {

    dreal const x = (rmax - j) * dxy;
    dreal const y = (rmax - i) * dxy;

    dreal const r = sqrt(pow(x/cos_inc, 2) + pow(y, 2));

    // TODO Require Rmin < dxy, else iR could be negative and we get a segfault
    // interpolate 1D.
    int const iR = max(int(floor((r-Rmin) / dR)), 0);

    int const row_offset = nxy / 2 - rmax;
    int const col_offset = nxy / 2 - rmax;
    auto const base = (i+row_offset)*rowsize + j+col_offset;

    // TODO Can we remove if clause by loop < 2*rmax?
    if (iR > nr-2) {
        image[base] = 0.0;
    } else {
        image[base] = sr_to_px * (intensity[iR] + (r - iR * dR - Rmin) * (intensity[iR + 1] - intensity[iR]) / dR);
    }
}

#ifdef __CUDACC__

__global__ void central_pixel_d(const int nxy, dcomplex* const __restrict__ image, const dreal value) {
    auto real_image = reinterpret_cast<dreal*>(image);
    auto const rowsize = 2*(nxy/2+1);
    real_image[nxy/2*rowsize+nxy/2] = value;
}

__global__ void sweep_d(int const nr, const dreal* const intensity, dreal const Rmin, dreal const dR,
                        const dreal rmax,
                        int const nxy, dreal const dxy, dreal const inc, dreal const sr_to_px,
                        dcomplex* const __restrict__ image) {

    dreal const cos_inc = cos(inc);

    auto real_image = reinterpret_cast<dreal*>(image);
    auto const rowsize = 2*(nxy/2+1);

    // indices
    int const x0 = blockDim.x * blockIdx.x + threadIdx.x;
    int const y0 = blockDim.y * blockIdx.y + threadIdx.y;

    // stride
    int const sx = blockDim.x * gridDim.x;
    int const sy = blockDim.y * gridDim.y;

    for (auto i = x0; i < 2*rmax; i += sx) {
        for (auto j = y0; j < 2*rmax; j += sy) {
            sweep_core(i, j, nr, intensity, Rmin, dR, rmax, nxy, rowsize, dxy, cos_inc, sr_to_px, real_image);
        }
    }

}

/**
 * Create image on device from `intensity`.
 */
CudaMemory<dcomplex> create_image_d(int nr, const dreal* const intensity, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal inc) {
    GPUTimer t, t_start;

    CHECK_CENTRAL_PIXEL(dxy, Rmin, dR);

    // start with a zero image
    CudaMemory<dcomplex> image_d(nxy * (nxy / 2 + 1)); t.Elapsed("create_image_d::malloc_image");
    CCheck(cudaMemset(image_d.ptr, 0, image_d.nbytes)); t.Elapsed("create_image_d::memset");

    // transfer intensities
    CudaMemory<dreal> intensity_d(nr, intensity); t.Elapsed("create_image_d::malloc_copy_intensity_H->D");

    // Convert intensities from Jy/steradians to Jy/pixels.
    // The intensity profile in input are in Jy/sr, while the sweeped image should be in Jy/px.
    const dreal sr_to_px = dxy * dxy;

    // most of the image will stay 0, we only need the kernel on a few pixels near the center
    auto const rmax = min((int)ceil((Rmin+nr*dR)/dxy), nxy/2);

    auto const nblocks = (2*rmax) / tpb + 1;

    sweep_d<<<dim3(nblocks, nblocks), dim3(tpb, tpb)>>>(nr, intensity_d.ptr, Rmin, dR, rmax, nxy, dxy, inc, sr_to_px, image_d.ptr);
    CCheck(cudaDeviceSynchronize());
    t.Elapsed("create_image_d::sweep");

    // central pixel
    auto const iIN = int(floor((dxy / 2 - Rmin) / dR));
    dreal flux = 0.;
    for (auto i = 1; i < iIN; ++i) {
        flux += (Rmin + dR * i) * intensity[i];
    };

    flux *= 2.;
    flux += Rmin * intensity[0] + (Rmin + iIN * dR) * intensity[iIN];
    flux *= dR;

    // add flux in the radial cell fraction between Rmin+iIN*dR and dxy/2 by linear interpolation
    // note that dxy/2 falls between Rmin+iIN*dR and Rmin+(iIN+1)*dR
    dreal I_interp = (intensity[iIN + 1] - intensity[iIN]) / (dR) * (dxy / 2. - (Rmin + dR * (iIN))) + intensity[iIN];
    flux += ((Rmin + iIN * dR) * intensity[iIN] + dxy / 2. * I_interp) * (dxy / 2. - (Rmin + iIN * dR));

    dreal area = pow(dxy/2., 2) - pow(Rmin, 2);

    auto const value = sr_to_px * flux / area;
    central_pixel_d<<<1,1>>>(nxy, image_d.ptr, value);
    CCheck(cudaDeviceSynchronize());
    t.Elapsed("create_image_d::central_pixel");

    t_start.Elapsed("create_image");

    return image_d;
}

#else


/**
 * Create 2D image from intensity profile.
 * For the central pixel, we compute the average intensity inside the pixel, i.e.:
 *
 *                 / dxy/2
 *                 |
 *                 | 2 pi I(R) R dR
 *                 |
 *                 / Rmin
 * central_pixel =  ---------------------
 *                 / dxy/2
 *                 |
 *                 | 2 pi R dR
 *                 |
 *                 / Rmin
 *
 * This formulation is based on the condition that the flux is conserved inside the central pixel.
 * The top integral is solved with the trapezoidal rule.
 *
 */
void create_image_h(int const nr, const dreal *const intensity, dreal const Rmin, dreal const dR, int const nxy, dreal const dxy,
                    dreal const inc, dcomplex *const image) {

    CPUTimer t;

    CHECK_CENTRAL_PIXEL(dxy, Rmin, dR);

    // start with zero image
    auto const ncol = nxy/2+1;
    auto const nbytes = sizeof(dcomplex)*nxy*ncol;
    memset(image, 0, nbytes);

    // now sweep
    dreal const cos_inc = cos(inc);
    int const rmax = min((int)ceil((Rmin+nr*dR)/dxy), nxy/2);

    // change units
    dreal const sr_to_px = dxy*dxy;

    auto real_image = reinterpret_cast<dreal*>(image);
    auto const rowsize = 2*ncol;
#pragma omp parallel for
    for (auto i = 0; i < 2*rmax; ++i) {
        for (auto j = 0; j < 2*rmax; ++j) {
            sweep_core(i, j, nr, intensity, Rmin, dR, rmax, nxy, rowsize, dxy, cos_inc, sr_to_px, real_image);
        }
    }

    // central pixel
    auto const iIN = int(floor((dxy / 2 - Rmin) / dR));
    dreal flux = 0.;
    for (auto i = 1; i < iIN; ++i) {
        flux += (Rmin + dR * i) * intensity[i];
    };

    flux *= 2.;
    flux += Rmin * intensity[0] + (Rmin + iIN * dR) * intensity[iIN];
    flux *= dR;

    // add flux in the radial cell fraction between Rmin+iIN*dR and dxy/2 by linear interpolation
    // note that dxy/2 falls between Rmin+iIN*dR and Rmin+(iIN+1)*dR
    dreal I_interp = (intensity[iIN + 1] - intensity[iIN]) / (dR) * (dxy / 2. - (Rmin + dR * (iIN))) + intensity[iIN];
    flux += ((Rmin + iIN * dR) * intensity[iIN] + dxy / 2. * I_interp) * (dxy / 2. - (Rmin + iIN * dR));

    dreal area = pow(dxy/2., 2) - pow(Rmin, 2);

    real_image[nxy/2*rowsize + nxy/2] = sr_to_px * flux / area;

    t.Elapsed("create_image");
}
#endif


namespace galario {
void sweep(int nr, const dreal* intensity, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal inc, dcomplex* image) {
    CHECK_INPUT(nxy);

#ifdef __CUDACC__
    // image allocated inside sweep
    CudaMemory<dcomplex> image_d = create_image_d(nr, intensity, Rmin, dR, nxy, dxy, inc);
    image_d.Retrieve(image);
#else
    create_image_h(nr, intensity, Rmin, dR, nxy, dxy, inc, image);
#endif
}

void _sweep(int nr, void *intensity, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal inc, void *image) {
    sweep(nr, static_cast<dreal *>(intensity), Rmin, dR, nxy, dxy, inc, static_cast<dcomplex *>(image));
}
}

#ifdef __CUDACC__
inline void sample_d(int nx, int ny, dcomplex* data_d, const dreal v_origin, dreal dRA, dreal dDec, int nd, dreal duv, const dreal PA, const dreal* u, const dreal* v, dcomplex* fint_d)
{
    GPUTimer t_start;

    int const ncol = ny/2+1;

    // ################################
    // ### ALLOCATION, INITIALIZATION ###
    // ################################

    /* async memory copy:, see issue https://github.com/mtazzari/galario/issues/40
       TODO copy memory asynchronously or create streams to define dependencies
       use nonzero cudaStream_t
       kernel<<< blocks, threads, bytes=0, stream =! 0>>>();

       all cufft calls are asynchronous, can specify the stream explicitly (cf. doc)
       same for cublas
       draw dependencies on paper: first thing is to do fft while other data is transferred
    */

    GPUTimer t;
    CudaMemory<dreal> u_d(nd, u);
    CudaMemory<dreal> v_d(nd, v);

    CudaMemory<dreal> urot_d(nd);
    CudaMemory<dreal> vrot_d(nd);
    t.Elapsed("sample::copy_uv_H->D");

    auto const nthreads = tpb * tpb;
    dreal dRArot = 0.;
    dreal dDecrot = 0.;

    // ################################
    // ########### KERNELS ############
    // ################################
    // rotate uv points
     if (PA==0.) {
        dRArot = dRA;
        dDecrot = dDec;
        cudaMemcpy(urot_d.ptr, u_d.ptr, u_d.nbytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(vrot_d.ptr, v_d.ptr, u_d.nbytes, cudaMemcpyDeviceToDevice);
        t.Elapsed("sample::copy_uvrot_D->D");
     } else {
        const dreal cos_PA = cos(PA);
        const dreal sin_PA = sin(PA);

        uv_rotate_d<<<nd/nthreads +1, nthreads>>>(cos_PA, sin_PA, nd, u_d.ptr, v_d.ptr, urot_d.ptr, vrot_d.ptr);
        uv_rotate_core(cos_PA, sin_PA, dRA, dDec, dRArot, dDecrot);
        t.Elapsed("sample::uv_rotate");
     }

    // Kernel for shift --> FFT --> shift
    shift_d<<<dim3(nx/2/tpb+1, ny/2/tpb+1), dim3(tpb, tpb)>>>(nx, ny, data_d); t.Elapsed("sample::1st_shift");
    fft_d(nx, ny, (dcomplex*) data_d); t.Elapsed("sample::FFT");
    shift_axis0_d<<<dim3(nx/2/tpb+1, ncol/2/tpb+1), dim3(tpb, tpb)>>>(nx, ncol, data_d); t.Elapsed("sample::2nd_shift");

    // oversubscribe blocks because we don't know if #(data points) divisible by nthreads
    interpolate_d<<<nd / nthreads + 1, nthreads>>>(nx, ncol, data_d, v_origin, nd, urot_d.ptr, vrot_d.ptr, duv, fint_d); t.Elapsed("sample::interpolate");

    // apply phase to the sampled points
    apply_phase_sampled_d<<<nd / nthreads + 1, nthreads>>>(dRArot, dDecrot, nd, urot_d.ptr, vrot_d.ptr, fint_d); t.Elapsed("sample::apply_phase_sampled");

    t_start.Elapsed("sample_tot");
}
#else

void sample_h(int nx, int ny, dcomplex* data, const dreal v_origin, dreal dRA, dreal dDec, int nd, dreal duv, const dreal PA, const dreal* u, const dreal* v, dcomplex* fint) {
    CPUTimer t_start;

    int const ncol = ny/2+1;

    OPENMPTIME(shift_h(nx, ny, data), "sample::1st_shift");

    OPENMPTIME(fft_h(nx, ny, data), "sample::FFT");

    OPENMPTIME(shift_axis0_h(nx, ncol, data), "sample::2nd_shift");

    auto urot = reinterpret_cast<dreal*>(FFTW(alloc_real)(nd));
    auto vrot = reinterpret_cast<dreal*>(FFTW(alloc_real)(nd));
    dreal dRArot;
    dreal dDecrot;
    uv_rotate_h(PA, dRA, dDec, &dRArot, &dDecrot, nd, u, v, urot, vrot);

    // interpolate
    OPENMPTIME(interpolate_h(nx, ncol, data, v_origin, nd, urot, vrot, duv, fint), "sample::interpolate");

    // apply phase to the sampled points
    OPENMPTIME(apply_phase_sampled_h(dRArot, dDecrot, nd, urot, vrot, fint), "sample::apply_phase_sampled");

    galario::galario_free(urot);
    galario::galario_free(vrot);
    t_start.Elapsed("sample_tot");
}

#endif


namespace galario {
/**
 * return result in `fint`
 */
void sample_image(int nx, int ny, const dreal* realdata, dreal v_origin, dreal dRA, dreal dDec, dreal duv,
                          const dreal PA, int nd, const dreal* u, const dreal* v, dcomplex* fint) {
    CPUTimer t_start;

    // Initialization for uv_idx and interpolate
    CHECK_INPUT(nx);

#ifdef __CUDACC__
    GPUTimer t_total;
    CudaMemory<dcomplex> fint_d(nd);

    auto data_d = copy_input_d(nx, ny, realdata);

    // do the actual computation
    sample_d(nx, ny, data_d.ptr, v_origin, dRA, dDec, nd, duv, PA, u, v, fint_d.ptr);

    // retrieve interpolated values
    CCheck(cudaDeviceSynchronize());

    GPUTimer t;
    fint_d.Retrieve(fint);
    t.Elapsed("sample_image::fint_ D->H");

    t_total.Elapsed("sample_image_tot");
#else
    CPUTimer t;

    auto data = copy_input(nx, ny, realdata); t.Elapsed("sample_image::copy_input");

    sample_h(nx, ny, data, v_origin, dRA, dDec, nd, duv, PA, u, v, fint);

    t = CPUTimer(); galario_free(data); t.Elapsed("sample_image::free_data");
#endif
    t_start.Elapsed("sample_image_tot");
}

void _sample_image(int nx, int ny, void* data, dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* fint) {
    sample_image(nx, ny, static_cast<dreal*>(data), v_origin, dRA, dDec, duv, PA, nd, static_cast<dreal*>(u), static_cast<dreal*>(v), static_cast<dcomplex*>(fint));
}


/**
 * return result in `fint`
 *
 */
void sample_profile(int nr, const dreal* intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA,
                    dreal dDec, dreal duv, dreal PA, int nd, const dreal *u, const dreal *v, dcomplex *fint) {
    CPUTimer t_start;

    CHECK_INPUT(nxy);

    // set origin 'upper' for images produced by sweep
    auto const v_origin = 1.;

#ifdef __CUDACC__
    CudaMemory<dcomplex> image_d = create_image_d(nr, intensity, Rmin, dR, nxy, dxy, inc);
    CudaMemory<dcomplex> fint_d(nd);

    // do the actual computation
    sample_d(nxy, nxy, image_d.ptr, v_origin, dRA, dDec, nd, duv, PA, u, v, fint_d.ptr);

    // retrieve interpolated values
    CCheck(cudaDeviceSynchronize());
    fint_d.Retrieve(fint);
#else
    int const ncol = nxy/2+1;

    // fftw_alloc for aligned memory to use SIMD acceleration
    auto data = reinterpret_cast<dcomplex*>(FFTW(alloc_complex)(nxy*ncol));

    // ensure data is initialized with zeroes
    create_image_h(nr, intensity, Rmin, dR, nxy, dxy, inc, data);

    sample_h(nxy, nxy, data, v_origin, dRA, dDec, nd, duv, PA, u, v, fint);
    galario_free(data);
#endif
    t_start.Elapsed("sample_profile_tot");
}


void _sample_profile(int nr, void *intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA, dreal dDec,
                             dreal duv, dreal PA, int nd, void *u, void *v, void *fint) {
    sample_profile(nr, static_cast<dreal *>(intensity), Rmin, dR, dxy, nxy, inc, dRA, dDec, duv, PA, nd,
                           static_cast<dreal *>(u), static_cast<dreal *>(v), static_cast<dcomplex *>(fint));
}
}


/**
 * Compute weighted difference between observations (`fobs_re` and `fobs_im`) and model predictions `fint`, write to `fint`
 */
#ifdef __CUDACC__
__host__ __device__
#endif
inline void diff_weighted_core(int const idx_x, int const nd, const dreal* const __restrict__ fobs_re,
                               const dreal * const __restrict__ fobs_im, const dcomplex* const __restrict__ fint,
                               const dreal* const __restrict__ weights, dcomplex& res)
{
    dcomplex const fobs_cmplx = dcomplex { fobs_re[idx_x], fobs_im[idx_x] };
    dcomplex const sqrt_w_cmplx = dcomplex { SQRT(weights[idx_x]), 0.0 } ;
    res = CMPLXSUB(fint[idx_x], fobs_cmplx);
    res = CMPLXMUL(res, sqrt_w_cmplx);
}

#ifdef __CUDACC__
__global__ void diff_weighted_d
(int const nd, const dreal* const __restrict__ fobs_re, const dreal* const __restrict__ fobs_im,  dcomplex* const __restrict__ fint, const dreal* const __restrict__ weights)
{
    //index
    int const idx_x0 = blockDim.x * blockIdx.x + threadIdx.x;

    // stride
    int const sidx_x = blockDim.x * gridDim.x;

    for (auto idx_x = idx_x0; idx_x < nd; idx_x += sidx_x) {
        // fint copied before, so it is ok to overwrite inside diff_weighted_core
        diff_weighted_core(idx_x, nd, fobs_re, fobs_im, fint, weights, fint[idx_x]);
    }
}
#endif

#ifdef __CUDACC__
dreal reduce_chi2_d
(int nd, const dreal* const __restrict__ fobs_re, const dreal* const __restrict__ fobs_im, dcomplex * const __restrict__ fint, const dreal* const __restrict__ weights)
{
    GPUTimer t_start, t;

    auto const nthreads = tpb * tpb;

    /* compute weighted difference */
    diff_weighted_d<<<nd / nthreads + 1, nthreads>>>(nd, fobs_re, fobs_im, fint, weights);
    t.Elapsed("reduce_chi2::diff_weighted");

    // only device pointers!
    // compute the Euclidean norm
    dreal chi2 = 0;
    CUBLASNRM2(cublas_handle(), nd, fint, 1, &chi2);
    // but we want the square of the norm
    chi2 *= chi2;
    t.Elapsed("reduce_chi2::reduction");
    t_start.Elapsed("reduce_chi2_tot");

    return chi2;
}
#endif

namespace galario {
dreal reduce_chi2(int nd, const dreal* fobs_re, const dreal* fobs_im, const dcomplex* fint, const dreal* weights) {
     CPUTimer t_start;
     dreal chi2 = 0.;
#ifdef __CUDACC__

    /* allocate and copy */
     CudaMemory<dreal> fobs_re_d(nd, fobs_re);
     CudaMemory<dreal> fobs_im_d(nd, fobs_im);
     CudaMemory<dcomplex> fint_d(nd, fint);
     CudaMemory<dreal> weights_d(nd, weights);

     chi2 = reduce_chi2_d(nd, fobs_re_d.ptr, fobs_im_d.ptr, fint_d.ptr, weights_d.ptr);
#else
     // compute chi2 by hand in a single pass over data, avoiding creation of
     // intermediate complex values
#pragma omp parallel for reduction(+:chi2)
     for (auto idx = 0; idx < nd; ++idx) {
         dcomplex chi;
         diff_weighted_core(idx, nd, fobs_re, fobs_im, fint, weights, chi);
         // \chi^2 += a^2 + b^2
         const auto a = real(chi);
         const auto b = imag(chi);
         chi2 += a*a + b*b;
     }
#endif
     t_start.Elapsed("reduce_chi2_tot");

     return chi2;
}

dreal _reduce_chi2(int nd, void* fobs_re, void* fobs_im, void* fint, void* weights) {
    return reduce_chi2(nd, static_cast<dreal*>(fobs_re), static_cast<dreal*>(fobs_im), static_cast<dcomplex*>(fint), static_cast<dreal*>(weights));
}

int ngpus()
{
    int num_devices = 0;
#ifdef __CUDACC__
    CCheck(cudaGetDeviceCount(&num_devices));
#endif
    return num_devices;
}

void use_gpu(int device_id)
{
#ifdef __CUDACC__
    CCheck(cudaSetDevice(device_id));
#endif
}

dreal chi2_image(int nx, int ny, const dreal* realdata, const dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u, const dreal* v, const dreal* fobs_re, const dreal* fobs_im, const dreal* weights) {
    CPUTimer t_start;

    CHECK_INPUTXY(nx, ny);
    dreal chi2 = 0;
#ifdef __CUDACC__
    GPUTimer t;
     // ################################
     // ### ALLOCATION, INITIALIZATION ###
     // ################################

     /* async memory copy:
      TODO copy memory asynchronously or create streams to define dependencies
      use nonzero cudaStream_t
      kernel<<< blocks, threads, bytes=0, stream =! 0>>>();

      all cufft calls are asynchronous, can specify the stream explicitly (cf. doc)
      same for cublas
      draw dependcies on paper: first thing is to do fft while other data is transferred

      While the FFT etc. are calculated, we can copy over the weights and observed values.
     */
    // reserve memory for the interpolated values
    CudaMemory<dcomplex> fint_d(nd);
    t.Elapsed("chi2_image::malloc_fint");

    // Initialization for comparison and chi square computation
    /* allocate and copy observational data */
    CudaMemory<dreal> fobs_re_d(nd, fobs_re);
    CudaMemory<dreal> fobs_im_d(nd, fobs_im);
    CudaMemory<dreal> weights_d(nd, weights);
    t.Elapsed("chi2_image::copy_observations");

    auto data_d = copy_input_d(nx, ny, realdata);

    sample_d(nx, ny, data_d.ptr, v_origin, dRA, dDec, nd, duv, PA, u, v, fint_d.ptr);
    chi2 = reduce_chi2_d(nd, fobs_re_d.ptr, fobs_im_d.ptr, fint_d.ptr, weights_d.ptr);
#else
    CPUTimer t;

    auto fint = reinterpret_cast<dcomplex*>(FFTW(alloc_complex)(nd)); t.Elapsed("chi2_imag::fftw_alloc");
    sample_image(nx, ny, realdata, v_origin, dRA, dDec, duv, PA, nd, u, v, fint);

    chi2 = reduce_chi2(nd, fobs_re, fobs_im, fint, weights);

    t = CPUTimer(); galario_free(fint); t.Elapsed("chi2_imag::free_fint");
#endif
    t_start.Elapsed("chi2_image_tot");
    flush_timing();

    return chi2;
}

dreal _chi2_image(int nx, int ny, void* realdata, const dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* fobs_re, void* fobs_im, void* weights) {
    return chi2_image(nx, ny, static_cast<dreal*>(realdata), v_origin, dRA, dDec, duv, PA, nd, static_cast<dreal*>(u),
                 static_cast<dreal*>(v), static_cast<dreal*>(fobs_re), static_cast<dreal*>(fobs_im),
                 static_cast<dreal*>(weights));
}

dreal chi2_profile(int nr, dreal *const intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA,
                          dreal dDec, dreal duv, dreal PA, int nd, const dreal *u, const dreal *v, const dreal *fobs_re,
                          const dreal *fobs_im, const dreal *weights) {
    CPUTimer t_start;
    CHECK_INPUT(nxy);
    dreal chi2 = 0;

#ifdef __CUDACC__
    GPUTimer t, t_start2;

    // set origin 'upper' for images produced by sweep
    auto const v_origin = 1.;

    CudaMemory<dcomplex> fint_d(nd);
    t.Elapsed("chi2_profile::malloc_fint");

    // Initialization for comparison and chi square computation
    /* allocate and copy observational data */
    CudaMemory<dreal> fobs_re_d(nd, fobs_re);
    CudaMemory<dreal> fobs_im_d(nd, fobs_im);
    CudaMemory<dreal> weights_d(nd, weights);
    t.Elapsed("chi2_profile::copy_observations");

    auto image_d = create_image_d(nr, intensity, Rmin, dR, nxy, dxy, inc);

    sample_d(nxy, nxy, image_d.ptr, v_origin, dRA, dDec, nd, duv, PA, u, v, fint_d.ptr);
    chi2 = reduce_chi2_d(nd, fobs_re_d.ptr, fobs_im_d.ptr, fint_d.ptr, weights_d.ptr);

    t_start2.Elapsed("chi2_profile_tot_gputimer");
#else
    CPUTimer t;

    dcomplex* fint = (dcomplex*) malloc(sizeof(dcomplex)*nd); t.Elapsed("chi2_profile::malloc_fint");
    sample_profile(nr, intensity, Rmin, dR, dxy, nxy, inc, dRA, dDec, duv, PA, nd, u, v, fint);
    chi2 = reduce_chi2(nd, fobs_re, fobs_im, fint, weights);
    t = CPUTimer(); free(fint); t.Elapsed("chi2_profile::free_fint");
#endif
    t_start.Elapsed("chi2_profile_tot");
    flush_timing();

    return chi2;
}

dreal _chi2_profile(int nr, void *intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA, dreal dDec,
                           dreal duv, dreal PA, int nd, void *u, void *v, void *fobs_re, void *fobs_im, void *weights) {
    return chi2_profile(nr, static_cast<dreal *>(intensity), Rmin, dR, dxy, nxy, inc, dRA, dDec, duv, PA, nd,
                         static_cast<dreal *>(u), static_cast<dreal *>(v), static_cast<dreal *>(fobs_re),
                         static_cast<dreal *>(fobs_im), static_cast<dreal *>(weights));
}
}
