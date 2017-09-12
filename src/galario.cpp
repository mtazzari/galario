#include "galario.h"
#include "galario_py.h"

// full function makes code hard to read
#define tpb galario_threads()

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cassert>
#include <cstring>
#include <cmath>
#include <iosfwd>
#include <iostream>
#include <sstream>

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

   /**
    * Macros to check input image lengths.
    */
    #define CHECK_INPUT(nx)   \
    do {                      \
        assert(nx >= 2);      \
        assert(nx % 2 == 0);  \
    } while (0)

    #define CHECK_INPUTXY(nx, ny) \
    do {                      \
        assert(nx >= 2);      \
        assert(ny >= 2);      \
        assert(nx % 2 == 0);  \
        assert(ny % 2 == 0);  \
    } while (0)

}

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
    #include <cuda_runtime_api.h>
    #include <cuda.h>
    #include <cuComplex.h>

    #include <cublas_v2.h>
    #include <cufft.h>

    #include <cstdio>
    #include <cstdlib>

    #define CCheck(err) __cudaSafeCall((err), __FILE__, __LINE__)
    inline void __cudaSafeCall(cudaError err, const char *file, const int line)  {
    #ifndef NDEBUG
        if(cudaSuccess != err) {
            fprintf(stderr, "[ERROR] Cuda call %s: %d\n%s\n", file, line, cudaGetErrorString(err));
            exit(42);
        }
    #endif
    }

    #define CBlasCheck(err) __cublasSafeCall((err), __FILE__, __LINE__)
    inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line) {
    #ifndef NDEBUG
        if(CUBLAS_STATUS_SUCCESS != err) {
           fprintf(stderr, "[ERROR] Cublas call %s: %d failed with code %d\n", file, line, err);
           exit(43);
        }
    #endif
    }

    #define CUFFTCheck(err) __cufftwSafeCall((err), __FILE__, __LINE__)
    inline void __cufftwSafeCall(cufftResult_t err, const char *file, const int line) {
    #ifndef NDEBUG
       if (CUFFT_SUCCESS != err) {
           fprintf(stderr, "[ERROR] Cufftw call %s: %d\n failed with code %d\n", file, line, err);
           exit(44);
       }
    #endif
    }

namespace {
    cublasHandle_t& cublas_handle() {
        static bool initialized = false;
        static cublasHandle_t handle;
        if (!initialized)
            CBlasCheck(cublasCreate(&handle));
        return handle;
    }
}

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

            void Stop() {
                CCheck(cudaEventRecord(stop, 0));
            }

            void Elapsed(const std::string& msg) {
                CCheck(cudaEventRecord(stop, 0));
                CCheck(cudaEventSynchronize(stop));
                float elapsed;
                CCheck(cudaEventElapsedTime(&elapsed, start, stop));
                ::out() << "[GPU] " << msg << ": " <<elapsed << " ms\n";
                Start();
            }
        };
    #else
        struct GPUTimer
        {
            GPUTimer() {}
            void Start() {}
            void Stop() {}
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
    // general min function already available in cuda
    // math_functions.hpp. Need `using` so the right implementation of
    // `min` is chosen for the kernels that are both on gpu and cpu
    #include <algorithm>
    using std::min;
    using std::max;

    #include <fftw3.h>
    #define FFTWCheck(status) __fftwSafeCall((status), __FILE__, __LINE__)

    inline void __fftwSafeCall(int status, const char *file, const int line) {
    #ifndef NDEBUG
        if(status == 0) {
            fprintf(stderr, "[ERROR] FFTW call %s: %d\n", file, line);
            exit(45);
        }
    #endif // NDEBUG
    }

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

int galario_threads(int num) {
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

void galario_init() {
#ifdef __CUDACC__
    cublas_handle();
#else
    #ifdef _OPENMP
    FFTWCheck(fftw_init_threads());
    #endif
#endif
}

void galario_cleanup() {
#ifdef __CUDACC__
    CBlasCheck(cublasDestroy(cublas_handle()));
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

#ifdef __CUDACC__
/**
 * Return device pointer to complex image made from real image with array size `nx*ny` on the host.
 *
 * Caller is responsible for freeing the device memory with `cudaFree()`.
 */
dcomplex* copy_input_d(int nx, int ny, const dreal* realdata) {
    GPUTimer t;
    auto const ncol = ny/2+1;
    auto const rowsize_real = sizeof(dreal)*ny;
    auto const rowsize_complex = sizeof(dcomplex)*ncol;

    // create destination array
    dcomplex *data_d;
    CCheck(cudaMalloc(&data_d, sizeof(dcomplex)*nx*ncol));

    // set the padding by defining different sizes of a row in bytes
    CCheck(cudaMemcpy2D(data_d, rowsize_complex, realdata, rowsize_real, rowsize_real, nx, cudaMemcpyHostToDevice));
    t.Elapsed("copy_input_H->D");
    return data_d;
}
#endif

/**
 * Copy an (nx, ny) square image into a complex buffer for real-to-complex FFTW.
 *
 * Buffer ownership transferred to caller, use `galario_free(buffer)`.
 *
 * If turns out to be slow have a look here:
 *   https://stackoverflow.com/questions/19601696/what-is-the-fastest-do-array-padding-of-the-image-array
 */
dcomplex* galario_copy_input(int nx, int ny, const dreal* realdata) {
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

void* _galario_copy_input(int nx, int ny, void* realdata) {
    return galario_copy_input(nx, ny, static_cast<dreal*>(realdata));
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
    fftw_plan_with_nthreads(galario_threads());
    FFTW(plan) p = FFTW(plan_dft_r2c_2d)(nx, ny, input, output, FFTW_ESTIMATE);
    FFTW(execute)(p);

    // TODO: find a way to store the plan (maybe homogeneously with the cuFFTPlan
    FFTW(destroy_plan)(p);
}
#endif

/**
 * `realdata`: nx * nx matrix
 * output: a buffer in the format described at http://fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data. It needs to be freed by `fftw_free`, not the ordinary `free`!
 */
void galario_fft2d(int nx, int ny, dcomplex* data) {
    CHECK_INPUTXY(nx, ny);
#ifdef __CUDACC__
    dcomplex *data_d;
    size_t nbytes = sizeof(dcomplex)*nx*(ny/2 + 1);
    CCheck(cudaMalloc(&data_d, nbytes));
    CCheck(cudaMemcpy(data_d, data, nbytes, cudaMemcpyHostToDevice));
    fft_d(nx, ny, data_d);

    CCheck(cudaMemcpy(data, data_d, nbytes, cudaMemcpyDeviceToHost));
    CCheck(cudaFree(data_d));
#else
    fft_h(nx, ny, data);
#endif
}

void _galario_fft2d(int nx, int ny, void* data) {
    galario_fft2d(nx, ny, static_cast<dcomplex*>(data));
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

void galario_fftshift(int nx, int ny, dcomplex* data) {
    CHECK_INPUTXY(nx, ny);
#ifdef __CUDACC__
    dcomplex *data_d;
    size_t nbytes = sizeof(dcomplex)*nx*(ny/2+1);
    CCheck(cudaMalloc(&data_d, nbytes));
    CCheck(cudaMemcpy(data_d, data, nbytes, cudaMemcpyHostToDevice));

    shift_d<<<dim3(nx/2/tpb+1, ny/2/tpb+1), dim3(tpb, tpb)>>>(nx, ny, data_d);

    CCheck(cudaDeviceSynchronize());
    CCheck(cudaMemcpy(data, data_d, nbytes, cudaMemcpyDeviceToHost));
    CCheck(cudaFree(data_d));
#else
    shift_h(nx, ny, data);
#endif
}

void _galario_fftshift(int nx, int ny, void* data) {
    galario_fftshift(nx, ny, static_cast<dcomplex*>(data));
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

void galario_fftshift_axis0(int nrow, int ncol, dcomplex* matrix) {
    CHECK_INPUT(nrow);
#ifdef __CUDACC__
    dcomplex *matrix_d;
    size_t nbytes = sizeof(dcomplex)*nrow*ncol;
    CCheck(cudaMalloc(&matrix_d, nbytes));
    CCheck(cudaMemcpy(matrix_d, matrix, nbytes, cudaMemcpyHostToDevice));

    shift_axis0_d<<<dim3(nrow/2/tpb+1, ncol/tpb+1), dim3(tpb, tpb)>>>(nrow, ncol, matrix_d);

    CCheck(cudaDeviceSynchronize());
    CCheck(cudaMemcpy(matrix, matrix_d, nbytes, cudaMemcpyDeviceToHost));
    CCheck(cudaFree(matrix_d));
#else
    shift_axis0_h(nrow, ncol, matrix);
#endif
}

void _galario_fftshift_axis0(int nrow, int ncol, void* matrix) {
    galario_fftshift_axis0(nrow, ncol, static_cast<dcomplex*>(matrix));
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
inline dcomplex interpolate_core(int const nrow, int const ncol, const dcomplex *const data,
                                 const dreal u, const dreal v, const dreal duv) {

    const int half_nrow = nrow/2;

    // compute indices
    dreal const indu = fabs(u)/duv;
    dreal indv;  // also indv is const

    // could be shorter: half_nx + sign(u)*v/duv;
    if (u < 0.) {
        indv = half_nrow - v/duv;
    }
    else {
        indv = half_nrow + v/duv;
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

    dcomplex interpolated = CMPLXADD(final_add1, y0);

    if (u < 0.) {
        interpolated = CMPLXCONJ(interpolated);
    }

    return interpolated;
}

#ifdef __CUDACC__
__global__ void interpolate_d(int const nrow, int const ncol, const dcomplex* const __restrict__ data, int const nd, const dreal* const u, const dreal* const v, dreal const duv, dcomplex* const __restrict__ fint)
{
    //index
    int const idx_0 = blockDim.x * blockIdx.x + threadIdx.x;

    // stride
    int const sx = blockDim.x * gridDim.x;

    for (auto idx = idx_0; idx < nd; idx += sx) {
        fint[idx] = interpolate_core(nrow, ncol, data,  u[idx], v[idx], duv);
    }
}
#else

void interpolate_h(int const nrow, int const ncol, const dcomplex* const data, int const nd, const dreal* const u, const dreal* const v, dreal const duv, dcomplex* fint) {

#pragma omp parallel for
    for (auto idx = 0; idx < nd; ++idx) {
        fint[idx] = interpolate_core(nrow, ncol, data, u[idx], v[idx], duv);
    }
}
#endif

void galario_interpolate(int nrow, int ncol, const dcomplex *data, int nd, const dreal *u, const dreal *v,
                         const dreal duv, dcomplex *fint) {

#ifdef __CUDACC__
    // copy the image data
    dcomplex *data_d;
    size_t nbytes = sizeof(dcomplex)*nrow*ncol;
    CCheck(cudaMalloc(&data_d, nbytes));
    CCheck(cudaMemcpy(data_d, data, nbytes, cudaMemcpyHostToDevice));

    // copy u,v and reserve memory for the interpolated values
    dreal *u_d, *v_d;
    dcomplex *fint_d;
    size_t nbytes_nd = sizeof(dreal)*nd;

    CCheck(cudaMalloc(&u_d, nbytes_nd));
    CCheck(cudaMemcpy(u_d, u, nbytes_nd, cudaMemcpyHostToDevice));

    CCheck(cudaMalloc(&v_d, nbytes_nd));
    CCheck(cudaMemcpy(v_d, v, nbytes_nd, cudaMemcpyHostToDevice));

    int nbytes_fint = sizeof(dcomplex) * nd;
    CCheck(cudaMalloc(&fint_d, nbytes_fint));

    // oversubscribe blocks because we don't know if #(data points) divisible by nthreads
    auto const nthreads = tpb * tpb;
    interpolate_d<<<nd / nthreads + 1, nthreads>>>(nrow, ncol, (dcomplex*) data_d, nd, (dreal*)u_d, (dreal*)v_d, duv, (dcomplex*) fint_d);

    CCheck(cudaDeviceSynchronize());

    // retrieve interpolated values
    CCheck(cudaMemcpy(fint, fint_d, nbytes_fint, cudaMemcpyDeviceToHost));

    // free memories
    CCheck(cudaFree(data_d));
    CCheck(cudaFree(u_d));
    CCheck(cudaFree(v_d));
    CCheck(cudaFree(fint_d));
#else
    interpolate_h(nrow, ncol, data, nd, u, v, duv, fint);
#endif
}

void _galario_interpolate(int nrow, int ncol, void *data, int nd, void *u, void *v, dreal duv, void *fint) {
    galario_interpolate(nrow, ncol, static_cast<dcomplex*>(data), nd, static_cast<dreal*>(u),
                        static_cast<dreal*>(v), duv, static_cast<dcomplex*>(fint));
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

void galario_apply_phase_sampled(dreal dRA, dreal dDec, int const nd, const dreal* const u, const dreal* const v, dcomplex* const __restrict__ fint) {
#ifdef __CUDACC__

     size_t nbytes_d_complex = sizeof(dcomplex)*nd;
     size_t nbytes_d_dreal = sizeof(dreal)*nd;

     dreal *u_d, *v_d;
     dcomplex *fint_d;

     CCheck(cudaMalloc(&u_d, nbytes_d_dreal));
     CCheck(cudaMemcpy(u_d, u, nbytes_d_dreal, cudaMemcpyHostToDevice));

     CCheck(cudaMalloc(&v_d, nbytes_d_dreal));
     CCheck(cudaMemcpy(v_d, v, nbytes_d_dreal, cudaMemcpyHostToDevice));

     CCheck(cudaMalloc(&fint_d, nbytes_d_complex));
     CCheck(cudaMemcpy(fint_d, fint, nbytes_d_complex, cudaMemcpyHostToDevice));

     auto const nthreads = tpb * tpb;
     apply_phase_sampled_d<<<nd/nthreads+1, nthreads>>>(dRA, dDec, nd, u_d, v_d, fint_d);

     CCheck(cudaDeviceSynchronize());
     CCheck(cudaMemcpy(fint, fint_d, nbytes_d_complex, cudaMemcpyDeviceToHost));
     CCheck(cudaFree(fint_d));
     CCheck(cudaFree(v_d));
     CCheck(cudaFree(u_d));
#else
    apply_phase_sampled_h(dRA, dDec, nd, u, v, fint);
#endif
}

void _galario_apply_phase_sampled(dreal dRA, dreal dDec, int nd, void* const u,
                                  void* const v, void* __restrict__ fint) {
    galario_apply_phase_sampled(dRA, dDec, nd, static_cast<dreal*>(u),
                                static_cast<dreal*>(v), static_cast<dcomplex*>(fint));
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

void galario_uv_rotate(dreal PA, dreal dRA, dreal dDec, dreal* dRArot, dreal* dDecrot, int const nd, const dreal* const u, const dreal* const v,
                       dreal* const urot, dreal* const vrot) {
#ifdef __CUDACC__
     size_t nbytes_d_dreal = sizeof(dreal)*nd;

     dreal *u_d, *v_d, *urot_d, *vrot_d;

     CCheck(cudaMalloc(&u_d, nbytes_d_dreal));
     CCheck(cudaMemcpy(u_d, u, nbytes_d_dreal, cudaMemcpyHostToDevice));

     CCheck(cudaMalloc(&v_d, nbytes_d_dreal));
     CCheck(cudaMemcpy(v_d, v, nbytes_d_dreal, cudaMemcpyHostToDevice));

     CCheck(cudaMalloc(&urot_d, nbytes_d_dreal));
     CCheck(cudaMalloc(&vrot_d, nbytes_d_dreal));

     if (PA==0.) {
        *dRArot = dRA;
        *dDecrot = dDec;
        cudaMemcpy(urot_d, u_d, nbytes_d_dreal, cudaMemcpyDeviceToDevice);
        cudaMemcpy(vrot_d, v_d, nbytes_d_dreal, cudaMemcpyDeviceToDevice);
     } else {
        const dreal cos_PA = cos(PA);
        const dreal sin_PA = sin(PA);

        auto const nthreads = tpb * tpb;
        uv_rotate_d<<<nd/nthreads +1, nthreads>>>(cos_PA, sin_PA, nd, u_d, v_d, urot_d, vrot_d);
        uv_rotate_core(cos_PA, sin_PA, dRA, dDec, *dRArot, *dDecrot);
     }
     CCheck(cudaDeviceSynchronize());
     CCheck(cudaMemcpy(urot, urot_d, nbytes_d_dreal, cudaMemcpyDeviceToHost));
     CCheck(cudaMemcpy(vrot, vrot_d, nbytes_d_dreal, cudaMemcpyDeviceToHost));
     CCheck(cudaFree(v_d));
     CCheck(cudaFree(u_d));
     CCheck(cudaFree(vrot_d));
     CCheck(cudaFree(urot_d));
#else
    uv_rotate_h(PA, dRA, dDec, dRArot, dDecrot, nd, u, v, urot, vrot);
#endif
}

void _galario_uv_rotate(dreal PA, dreal dRA, dreal dDec, void* dRArot, void* dDecrot, int nd, void* const u,
                                  void* const v, void* const urot, void* const vrot) {
    galario_uv_rotate(PA, dRA, dDec, static_cast<dreal*>(dRArot), static_cast<dreal*>(dDecrot), nd, static_cast<dreal*>(u),
                                static_cast<dreal*>(v), static_cast<dreal*>(urot), static_cast<dreal*>(vrot));
}


/**
 * Sweep.
 * TODO avoid rmax 3 definitions. pass rmax (perhaps also base as argument of sweep_core.
 **/
#ifdef __CUDACC__
__host__ __device__
#endif
inline void sweep_core(int const i, int const j, int const nr, const dreal* const ints,
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
        image[base] = sr_to_px * (ints[iR] + (r - iR * dR - Rmin) * (ints[iR + 1] - ints[iR]) / dR);
    }
}

#ifdef __CUDACC__

__global__ void central_pixel_d(const int nxy, dcomplex* const __restrict__ image, const dreal value) {
    auto real_image = reinterpret_cast<dreal*>(image);
    auto const rowsize = 2*(nxy/2+1);
    real_image[nxy/2*rowsize+nxy/2] = value;
}

__global__ void sweep_d(int const nr, const dreal* const ints, dreal const Rmin, dreal const dR,
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
            sweep_core(i, j, nr, ints, Rmin, dR, rmax, nxy, rowsize, dxy, cos_inc, sr_to_px, real_image);
        }
    }

}

/**
 * Allocate memory on device for `ints` and `image`. `addr_*` is the address of the pointer to the beginning of that memory.
 */
void create_image_d(int nr, const dreal* const ints, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal inc, dcomplex** addr_image_d) {
    GPUTimer t, t_start;
    auto const ncol = nxy/2+1;
    auto const nbytes = sizeof(dcomplex)*nxy*ncol;

    // start with a zero image
    CCheck(cudaMalloc(addr_image_d, nbytes)); t.Elapsed("create_image_d::malloc_image");
    CCheck(cudaMemset(*addr_image_d, 0, nbytes)); t.Elapsed("create_image_d::memset");

    // transfer intensities
    dreal* ints_d;
    auto const nbytes_ints = sizeof(dreal)*nr;
    CCheck(cudaMalloc(&ints_d, nbytes_ints)); t.Elapsed("create_image_d::malloc_ints");
    CCheck(cudaMemcpy(ints_d, ints, nbytes_ints, cudaMemcpyHostToDevice));
    t.Elapsed("create_image_d::copy_ints_H->D");

    // Convert intensities from Jy/steradians to Jy/pixels.
    // The intensity profile in input are in Jy/sr, while the sweeped image should be in Jy/px.
    dreal const sr_to_px = dxy*dxy;

    // most of the image will stay 0, we only need the kernel on a few pixels near the center
    auto const rmax = min((int)ceil((Rmin+nr*dR)/dxy), nxy/2);

    auto const nblocks = (2*rmax) / tpb + 1;
    sweep_d<<<dim3(nblocks, nblocks), dim3(tpb, tpb)>>>(nr, ints_d, Rmin, dR, rmax, nxy, dxy, inc, sr_to_px, *addr_image_d);
    CCheck(cudaDeviceSynchronize());
    t.Elapsed("create_image_d::sweep");

    // central pixel needs special treatment
    auto const value = sr_to_px * (ints[0] + Rmin * (ints[0] - ints[1]) / dR);
    central_pixel_d<<<1,1>>>(nxy, *addr_image_d, value);
    t.Elapsed("create_image_d::central_pixel");

    CCheck(cudaFree(ints_d)); t.Elapsed("create_image_d::free_ints");
    t_start.Elapsed("create_image");
}

#else

void create_image_h(int const nr, const dreal *const ints, dreal const Rmin, dreal const dR, int const nxy, dreal const dxy,
                    dreal const inc, dcomplex *const image) {
    CPUTimer t;

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
            sweep_core(i, j, nr, ints, Rmin, dR, rmax, nxy, rowsize, dxy, cos_inc, sr_to_px, real_image);
        }
    }

    // central pixel
    if (Rmin != 0.)
        real_image[nxy/2*rowsize+nxy/2] = sr_to_px * (ints[0] + Rmin * (ints[0] - ints[1]) / dR);


    t.Elapsed("create_image");
}
#endif


void galario_sweep(int nr, dreal *const ints, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal inc, dcomplex *image) {
    CHECK_INPUT(nxy);

#ifdef __CUDACC__
    // image allocated inside sweep
    dcomplex *image_d;
    create_image_d(nr, ints, Rmin, dR, nxy, dxy, inc, &image_d);

    auto const nbytes = sizeof(dcomplex)*nxy*(nxy/2+1);
    CCheck(cudaMemcpy(image, image_d, nbytes, cudaMemcpyDeviceToHost));
    CCheck(cudaFree(image_d));
#else
    create_image_h(nr, ints, Rmin, dR, nxy, dxy, inc, image);
#endif
}

void _galario_sweep(int nr, void *ints, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal inc, void *image) {
    galario_sweep(nr, static_cast<dreal *>(ints), Rmin, dR, nxy, dxy, inc, static_cast<dcomplex *>(image));
}

#ifdef __CUDACC__
inline void sample_d(int nx, int ny, dcomplex* data_d, dreal dRA, dreal dDec, int nd, dreal duv, const dreal PA, const dreal* u, const dreal* v, dcomplex* fint_d)
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

    dreal *u_d, *v_d, *urot_d, *vrot_d;
    size_t nbytes_ndat = sizeof(dreal)*nd;

    GPUTimer t;
    CCheck(cudaMalloc(&u_d, nbytes_ndat));
    CCheck(cudaMemcpy(u_d, u, nbytes_ndat, cudaMemcpyHostToDevice));
    CCheck(cudaMalloc(&v_d, nbytes_ndat));
    CCheck(cudaMemcpy(v_d, v, nbytes_ndat, cudaMemcpyHostToDevice));
    CCheck(cudaMalloc(&urot_d, nbytes_ndat));
    CCheck(cudaMalloc(&vrot_d, nbytes_ndat));
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
        cudaMemcpy(urot_d, u_d, nbytes_ndat, cudaMemcpyDeviceToDevice);
        cudaMemcpy(vrot_d, v_d, nbytes_ndat, cudaMemcpyDeviceToDevice);
        t.Elapsed("sample::copy_uvrot_D->D");
     } else {
        const dreal cos_PA = cos(PA);
        const dreal sin_PA = sin(PA);

        uv_rotate_d<<<nd/nthreads +1, nthreads>>>(cos_PA, sin_PA, nd, u_d, v_d, urot_d, vrot_d);
        uv_rotate_core(cos_PA, sin_PA, dRA, dDec, dRArot, dDecrot);
        t.Elapsed("sample::uv_rotate");
     }

    // Kernel for shift --> FFT --> shift
    shift_d<<<dim3(nx/2/tpb+1, ny/2/tpb+1), dim3(tpb, tpb)>>>(nx, ny, data_d); t.Elapsed("sample::1st_shift");
    fft_d(nx, ny, (dcomplex*) data_d); t.Elapsed("sample::FFT");
    shift_axis0_d<<<dim3(nx/2/tpb+1, ncol/2/tpb+1), dim3(tpb, tpb)>>>(nx, ncol, data_d); t.Elapsed("sample::2nd_shift");

    // oversubscribe blocks because we don't know if #(data points) divisible by nthreads
    interpolate_d<<<nd / nthreads + 1, nthreads>>>(nx, ncol, data_d, nd, urot_d, vrot_d, duv, fint_d); t.Elapsed("sample::interpolate");

    // apply phase to the sampled points
    apply_phase_sampled_d<<<nd / nthreads + 1, nthreads>>>(dRArot, dDecrot, nd, urot_d, vrot_d, fint_d); t.Elapsed("sample::apply_phase_sampled");

    // ################################
    // ########### CLEANUP ############
    // ################################
    CCheck(cudaFree(u_d));
    CCheck(cudaFree(v_d));
    CCheck(cudaFree(vrot_d));
    CCheck(cudaFree(urot_d));

    t_start.Elapsed("sample_tot");
}
#else

void sample_h(int nx, int ny, dcomplex* data, dreal dRA, dreal dDec, int nd, dreal duv, const dreal PA, const dreal* u, const dreal* v, dcomplex* fint) {
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
    OPENMPTIME(interpolate_h(nx, ncol, data, nd, urot, vrot, duv, fint), "sample::interpolate");

    // apply phase to the sampled points
    OPENMPTIME(apply_phase_sampled_h(dRArot, dDecrot, nd, urot, vrot, fint), "sample::apply_phase_sampled");

    galario_free(urot);
    galario_free(vrot);
    t_start.Elapsed("sample_tot");
}

#endif


/**
 * return result in `fint`
 */
void galario_sample_image(int nx, int ny, const dreal* realdata, dreal dRA, dreal dDec, dreal duv,
                          const dreal PA, int nd, const dreal* u, const dreal* v, dcomplex* fint) {
    CPUTimer t_start;

    // Initialization for uv_idx and interpolate
    CHECK_INPUT(nx);

#ifdef __CUDACC__
    GPUTimer t_total;
    dcomplex *fint_d;
    int nbytes_fint = sizeof(dcomplex) * nd;
    CCheck(cudaMalloc(&fint_d, nbytes_fint));

    dcomplex* data_d = copy_input_d(nx, ny, realdata);

    // do the actual computation
    sample_d(nx, ny, data_d, dRA, dDec, nd, duv, PA, u, v, fint_d);

    // retrieve interpolated values
    CCheck(cudaDeviceSynchronize());

    GPUTimer t;
    CCheck(cudaMemcpy(fint, fint_d, nbytes_fint, cudaMemcpyDeviceToHost)); t.Elapsed("sample_image::fint_ D->H");

    CCheck(cudaFree(fint_d)); t.Elapsed("sample_image::fint_cudaFree");
    CCheck(cudaFree(data_d)); t.Elapsed("sample_image::data_cudaFree");
    t_total.Elapsed("sample_image_tot");
#else
    CPUTimer t;

    auto data = galario_copy_input(nx, ny, realdata); t.Elapsed("sample_image::copy_input");

    sample_h(nx, ny, data, dRA, dDec, nd, duv, PA, u, v, fint);

    t = CPUTimer(); galario_free(data); t.Elapsed("sample_image::free_data");
#endif
    t_start.Elapsed("sample_image_tot");
}

void _galario_sample_image(int nx, int ny, void* data, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* fint) {
    galario_sample_image(nx, ny, static_cast<dreal*>(data), dRA, dDec, duv, PA, nd, static_cast<dreal*>(u), static_cast<dreal*>(v), static_cast<dcomplex*>(fint));
}


/**
 * return result in `fint`
 *
 */
void galario_sample_profile(int nr, dreal *const ints, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA,
                            dreal dDec, dreal duv, dreal PA, int nd, const dreal *u, const dreal *v, dcomplex *fint) {
    CPUTimer t_start;

    CHECK_INPUT(nxy);

#ifdef __CUDACC__
    dcomplex *image_d;

    create_image_d(nr, ints, Rmin, dR, nxy, dxy, inc, &image_d);

    dcomplex *fint_d;
    int nbytes_fint = sizeof(dcomplex) * nd;
    CCheck(cudaMalloc(&fint_d, nbytes_fint));

    // do the actual computation
    sample_d(nxy, nxy, image_d, dRA, dDec, nd, duv, PA, u, v, fint_d);

    // retrieve interpolated values
    CCheck(cudaDeviceSynchronize());
    CCheck(cudaMemcpy(fint, fint_d, nbytes_fint, cudaMemcpyDeviceToHost));

    CCheck(cudaFree(fint_d));
    CCheck(cudaFree(image_d));
#else
    int const ncol = nxy/2+1;

    // fftw_alloc for aligned memory to use SIMD acceleration
    auto data = reinterpret_cast<dcomplex*>(FFTW(alloc_complex)(nxy*ncol));

    // ensure data is initialized with zeroes
    create_image_h(nr, ints, Rmin, dR, nxy, dxy, inc, data);

    sample_h(nxy, nxy, data, dRA, dDec, nd, duv, PA, u, v, fint);
    galario_free(data);
#endif
    t_start.Elapsed("sample_profile_tot");
}


void _galario_sample_profile(int nr, void *ints, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA, dreal dDec,
                             dreal duv, dreal PA, int nd, void *u, void *v, void *fint) {
    galario_sample_profile(nr, static_cast<dreal *>(ints), Rmin, dR, dxy, nxy, inc, dRA, dDec, duv, PA, nd,
                           static_cast<dreal *>(u), static_cast<dreal *>(v), static_cast<dcomplex *>(fint));
}


/**
 * Compute weighted difference between observations (`fobs_re` and `fobs_im`) and model predictions `fint`, write to `fint`
 */
#ifdef __CUDACC__
__host__ __device__
#endif
inline void diff_weighted_core(int const idx_x, int const nd, const dreal* const __restrict__ fobs_re,
                               const dreal * const __restrict__ fobs_im, dcomplex* const __restrict__ fint,
                               const dreal* const __restrict__ weights)
{
    dcomplex const fobs_cmplx = dcomplex { fobs_re[idx_x], fobs_im[idx_x] };
    dcomplex const sqrt_w_cmplx = dcomplex { SQRT(weights[idx_x]), 0.0 } ;
    fint[idx_x] = CMPLXSUB(fint[idx_x], fobs_cmplx);
    fint[idx_x] = CMPLXMUL(fint[idx_x], sqrt_w_cmplx);
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
        diff_weighted_core(idx_x, nd, fobs_re, fobs_im, fint, weights);
    }
}
#else

void diff_weighted_h
        (int const nd, const dreal* const fobs_re, const dreal* const fobs_im, dcomplex* const fint, const dreal* const weights)
{
    CPUTimer t;
#pragma omp parallel for
    for (auto idx = 0; idx < nd; ++idx) {
        diff_weighted_core(idx, nd, fobs_re, fobs_im, fint, weights);
    }
    t.Elapsed("reduce_chi2::diff_weighted");
}
#endif

#ifdef __CUDACC__
void reduce_chi2_d
(int nd, const dreal* const __restrict__ fobs_re, const dreal* const __restrict__ fobs_im, dcomplex * const __restrict__ fint, const dreal* const __restrict__ weights, dreal* const __restrict__ chi2)
{
    GPUTimer t_start, t;

    auto const nthreads = tpb * tpb;

    /* compute weighted difference */
    diff_weighted_d<<<nd / nthreads + 1, nthreads>>>(nd, fobs_re, fobs_im, fint, weights);
    t.Elapsed("reduce_chi2::diff_weighted");

    // only device pointers! maybe not ... check with jiri
    // compute the Euclidean norm
    CUBLASNRM2(cublas_handle(), nd, fint, 1, chi2);
    // but we want the square of the norm
    *chi2 *= *chi2;
    t.Elapsed("reduce_chi2::reduction");
    t_start.Elapsed("reduce_chi2_tot");
}
#endif

void galario_reduce_chi2(int nd, const dreal* fobs_re, const dreal* fobs_im, dcomplex* fint, const dreal* weights, dreal* chi2) {
     CPUTimer t_start;
#ifdef __CUDACC__

    /* allocate and copy */
     dreal *fobs_re_d, *fobs_im_d, *weights_d;
     size_t nbytes_nd = sizeof(dreal)*nd;

     CCheck(cudaMalloc(&fobs_re_d, nbytes_nd));
     CCheck(cudaMemcpy(fobs_re_d, fobs_re, nbytes_nd, cudaMemcpyHostToDevice));

     CCheck(cudaMalloc(&fobs_im_d, nbytes_nd));
     CCheck(cudaMemcpy(fobs_im_d, fobs_im, nbytes_nd, cudaMemcpyHostToDevice));

     CCheck(cudaMalloc(&weights_d, nbytes_nd));
     CCheck(cudaMemcpy(weights_d, weights, nbytes_nd, cudaMemcpyHostToDevice));

     dreal *chi2_d;
     size_t nbytes_chi2 = sizeof(dreal);
     CCheck(cudaMalloc(&chi2_d, nbytes_chi2));

     dcomplex* fint_d;
     size_t nbytes_fint = sizeof(dcomplex) * nd;
     CCheck(cudaMalloc(&fint_d, nbytes_fint));
     CCheck(cudaMemcpy(fint_d, fint, nbytes_fint, cudaMemcpyHostToDevice));

     reduce_chi2_d(nd, fobs_re_d, fobs_im_d, fint_d, weights_d, chi2);

     // CCheck(cudaMemcpy(fint, fint_d, nbytes_fint, cudaMemcpyDeviceToHost));

     /* free */
     CCheck(cudaFree(fobs_re_d));
     CCheck(cudaFree(fobs_im_d));
     CCheck(cudaFree(weights_d));
     CCheck(cudaFree(chi2_d));
     CCheck(cudaFree(fint_d));

#else
     diff_weighted_h(nd, fobs_re, fobs_im, fint, weights);

     CPUTimer t;
     // TODO: if available, use BLAS (mkl?) functions cblas_scnrm2 or cblas_dznrm2 for float/double complex
     // compute the Euclidean norm
     dreal y = 0.;
#pragma omp parallel for reduction(+:y)
     for (auto i = 0; i < nd; ++i) {
         dcomplex const x = fint[i];
         dcomplex const x_conj = CMPLXCONJ(fint[i]);
         y += real(CMPLXMUL(x, x_conj));
     }
     *chi2 = y;
     t.Elapsed("reduce_chi2::reduction");
#endif
     t_start.Elapsed("reduce_chi2_tot");
}

void _galario_reduce_chi2(int nd, void* fobs_re, void* fobs_im, void* fint, void* weights, dreal* chi2) {
    galario_reduce_chi2(nd, static_cast<dreal*>(fobs_re), static_cast<dreal*>(fobs_im), static_cast<dcomplex*>(fint), static_cast<dreal*>(weights), chi2);
}

int galario_ngpus()
{
    int num_devices = 0;
#ifdef __CUDACC__
    CCheck(cudaGetDeviceCount(&num_devices));
#endif
    return num_devices;
}

void galario_use_gpu(int device_id)
{
#ifdef __CUDACC__
    CCheck(cudaSetDevice(device_id));
#endif
}

#ifdef __CUDACC__
void copy_observations_d(int nd, const dreal* x, dreal** addr_x_d) {
    size_t nbytes_ndat = sizeof(dreal)*nd;
    CCheck(cudaMalloc(addr_x_d, nbytes_ndat));
    CCheck(cudaMemcpy(*addr_x_d, x, nbytes_ndat, cudaMemcpyHostToDevice));
}
#endif

void galario_chi2_image(int nx, int ny, const dreal* realdata, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u, const dreal* v, const dreal* fobs_re, const dreal* fobs_im, const dreal* weights, dreal* chi2) {
    CPUTimer t_start;

    CHECK_INPUTXY(nx, ny);
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
    dcomplex *fint_d;
    int nbytes_fint = sizeof(dcomplex) * nd;
    CCheck(cudaMalloc(&fint_d, nbytes_fint)); t.Elapsed("chi2_image::malloc_fint");

    // Initialization for comparison and chi square computation
    /* allocate and copy observational data */
    dreal *fobs_re_d, *fobs_im_d, *weights_d;
    copy_observations_d(nd, fobs_re, &fobs_re_d);
    copy_observations_d(nd, fobs_im, &fobs_im_d);
    copy_observations_d(nd, weights, &weights_d);
    t.Elapsed("chi2_image::copy_observations");

    dcomplex* data_d = copy_input_d(nx, ny, realdata);

    sample_d(nx, ny, data_d, dRA, dDec, nd, duv, PA, u, v, fint_d);
    reduce_chi2_d(nd, fobs_re_d, fobs_im_d, fint_d, weights_d, chi2);

    // ################################
    // ########### CLEANUP ############
    // ################################
    t.Start();
    CCheck(cudaFree(fint_d));
    CCheck(cudaFree(fobs_re_d));
    CCheck(cudaFree(fobs_im_d));
    CCheck(cudaFree(weights_d));
    CCheck(cudaFree(data_d));
    t.Elapsed("chi2_image::free_all");
#else
    CPUTimer t;

    auto fint = reinterpret_cast<dcomplex*>(FFTW(alloc_complex)(nd)); t.Elapsed("chi2_imag::fftw_alloc");
    galario_sample_image(nx, ny, realdata, dRA, dDec, duv, PA, nd, u, v, fint);

    galario_reduce_chi2(nd, fobs_re, fobs_im, fint, weights, chi2);

    t = CPUTimer(); galario_free(fint); t.Elapsed("chi2_imag::free_fint");
#endif
    t_start.Elapsed("chi2_image_tot");
    flush_timing();
}

void _galario_chi2_image(int nx, int ny, void* realdata, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* fobs_re, void* fobs_im, void* weights, dreal* chi2) {
    galario_chi2_image(nx, ny, static_cast<dreal*>(realdata), dRA, dDec, duv, PA, nd, static_cast<dreal*>(u),
                 static_cast<dreal*>(v), static_cast<dreal*>(fobs_re), static_cast<dreal*>(fobs_im),
                 static_cast<dreal*>(weights), chi2);
}

void galario_chi2_profile(int nr, dreal *const ints, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA,
                          dreal dDec, dreal duv, dreal PA, int nd, const dreal *u, const dreal *v, const dreal *fobs_re,
                          const dreal *fobs_im, const dreal *weights, dreal *chi2) {
    CPUTimer t_start;
    CHECK_INPUT(nxy);
#ifdef __CUDACC__
    GPUTimer t, t_start2;
    dcomplex *fint_d;
    int nbytes_fint = sizeof(dcomplex) * nd;
    CCheck(cudaMalloc(&fint_d, nbytes_fint)); t.Elapsed("chi2_profile::malloc_fint");

    // Initialization for comparison and chi square computation
    /* allocate and copy observational data */
    dreal *fobs_re_d, *fobs_im_d, *weights_d;
    copy_observations_d(nd, fobs_re, &fobs_re_d);
    copy_observations_d(nd, fobs_im, &fobs_im_d);
    copy_observations_d(nd, weights, &weights_d);
    t.Elapsed("chi2_profile::copy_observations");

    dcomplex *image_d;
    create_image_d(nr, ints, Rmin, dR, nxy, dxy, inc, &image_d);

    sample_d(nxy, nxy, image_d, dRA, dDec, nd, duv, PA, u, v, fint_d);
    reduce_chi2_d(nd, fobs_re_d, fobs_im_d, fint_d, weights_d, chi2);

    t.Start();
    CCheck(cudaFree(fint_d));
    CCheck(cudaFree(fobs_re_d));
    CCheck(cudaFree(fobs_im_d));
    CCheck(cudaFree(weights_d));
    CCheck(cudaFree(image_d));
    t.Elapsed("chi2_profile::free_all");
    t_start2.Elapsed("chi2_profile_tot_gputimer");
#else
    CPUTimer t;

    dcomplex* fint = (dcomplex*) malloc(sizeof(dcomplex)*nd); t.Elapsed("chi2_profile::malloc_fint");
    galario_sample_profile(nr, ints, Rmin, dR, dxy, nxy, inc, dRA, dDec, duv, PA, nd, u, v, fint);
    galario_reduce_chi2(nd, fobs_re, fobs_im, fint, weights, chi2);
    t = CPUTimer(); free(fint); t.Elapsed("chi2_profile::free_fint");
#endif
    t_start.Elapsed("chi2_profile_tot");
    flush_timing();
}

void _galario_chi2_profile(int nr, void *ints, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA, dreal dDec,
                           dreal duv, dreal PA, int nd, void *u, void *v, void *fobs_re, void *fobs_im, void *weights,
                           dreal *chi2) {
    galario_chi2_profile(nr, static_cast<dreal *>(ints), Rmin, dR, dxy, nxy, inc, dRA, dDec, duv, PA, nd,
                         static_cast<dreal *>(u), static_cast<dreal *>(v), static_cast<dreal *>(fobs_re),
                         static_cast<dreal *>(fobs_im), static_cast<dreal *>(weights), chi2);
}
