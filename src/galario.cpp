#include "galario.h"
#include "galario_py.h"

#ifdef __CUDACC__
    #include <cuda_runtime_api.h>
    #include <cuda.h>
    #include <cuComplex.h>

    #include <cublas_v2.h>

    #include <cstdio>
    #include <cstdlib>

    #define CCheck(err) __cudaSafeCall((err), __FILE__, __LINE__)

    inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
    #ifndef NDEBUG
        if(cudaSuccess != err) {
            fprintf(stderr, "[ERROR] Cuda call %s: %d\n%s\n", file, line, cudaGetErrorString(err));
            exit(42);
        }
    }
    #endif

    // TODO do for cufft

    #define CBlasCheck(err) __cublasSafeCall((err), __FILE__, __LINE__)

    // TODO output error code
    inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line) {
    #ifndef NDEBUG
        if(CUBLAS_STATUS_SUCCESS != err) {
            fprintf(stderr, "[ERROR] Cublas call %s: %d\n", file, line);
            exit(43);
        }
    #endif
    }

    #ifdef DOUBLE_PRECISION
        #define CUFFTEXEC cufftExecZ2Z
        #define CUFFTTYPE CUFFT_Z2Z
        #define CMPLX(a, b) (make_cuDoubleComplex(a,b))
        #define CMPLXSUB cuCsub
        #define CMPLXADD cuCadd
        #define CMPLXMUL cuCmul
        #define CUBLASNRM2 cublasDznrm2

    #else
        #define CUFFTEXEC cufftExecC2C
        #define CUFFTTYPE CUFFT_C2C
        #define CMPLX(a, b) (make_cuFloatComplex(a,b))
        #define CMPLXSUB cuCsubf
        #define CMPLXADD cuCaddf
        #define CMPLXMUL cuCmulf
        #define CUBLASNRM2 cublasScnrm2
    #endif  // DOUBLE_PRECISION
#else
    #define CMPLXSUB(a, b) ((a) - (b))
    #define CMPLXADD(a, b) ((a) + (b))
    #define CMPLXMUL(a, b) ((a) * (b))
#ifdef _OPENMP
    #include <omp.h>
#endif
    #include <fftw3.h>

#define FFTWCheck(status) __fftwSafeCall((status), __FILE__, __LINE__)

inline void __fftwSafeCall(int status, const char *file, const int line) {
#ifndef NDEBUG
    if(status == 0) {
        fprintf(stderr, "[ERROR] FFTW call %s: %d\n", file, line);
        exit(44);
    }
#endif // NDEBUG
}

#endif // __CUDACC__

#ifdef DOUBLE_PRECISION
    #define SQRT sqrt
    #define FFTW(name) fftw_ ## name
#else
    #define SQRT sqrtf
    #define FFTW(name) fftwf_ ## name
#endif

#include <cassert>
#include <cmath>

constexpr int NRANK = 2;
constexpr int BATCH = 1;

int galario_threads_per_block(int x)
{
    static int mynthreads = 32;
    if (x > 0)
        mynthreads = x;
    return mynthreads;
}

#ifdef __CUDACC__
void galario_init() {}
void galario_cleanup() {}

#else
void galario_init() {
#ifdef _OPENMP
    FFTWCheck(fftw_init_threads());
    fftw_plan_with_nthreads(omp_get_max_threads());
#endif
}

void galario_cleanup() {
#ifdef _OPENMP
  FFTW(cleanup_threads)();
#endif
  FFTW(cleanup)();
}
#endif // __CUDACC__


#ifdef __CUDACC__
void fft_d(int nx, int ny, dcomplex* data_d) {
     cufftHandle plan;
     int n[NRANK] = {nx, ny};

     /* Create a 2D FFT plan. */
     // TODO: find a way to store the plan
     if (cufftPlanMany(&plan, NRANK, n,
                       NULL, 1, 0,
                       NULL, 1, 0,
                       CUFFTTYPE,BATCH) != CUFFT_SUCCESS){
          fprintf(stderr, "CUFFT Error: Unable to create plan\n");
          return;
     }

     if (CUFFTEXEC(plan, data_d, data_d, CUFFT_FORWARD) != CUFFT_SUCCESS){
          fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
          return;
     }

     // cufft calls are asynchronous
     CCheck(cudaDeviceSynchronize());
     cufftDestroy(plan);
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
    FFTW(plan) p = FFTW(plan_dft_r2c_2d)(nx, ny, input, output, FFTW_ESTIMATE);
    FFTW(execute)(p);

    // TODO: find a way to store the plan (maybe homogeneously with the cuFFTPlan
    FFTW(destroy_plan)(p);
}

#endif

/**
 * Copy an (nx, ny) square image into a complex buffer for real-to-complex FFTW.
 *
 * Operates only on the host.
 */
dcomplex* copy_real_to_buffer(int nx, int ny, const dreal* realdata) {
    // in r2c, the last dimension only has ~half the size
    auto const complex_columns = ny/2 + 1;

    // fftw_alloc for aligned memory to use SIMD acceleration
    auto buffer = reinterpret_cast<dcomplex*>(FFTW(alloc_complex)(nx*complex_columns));

    // copy and respect padding in last dimension. Treating the complex output
    // buffer as a sequence of real entries, the last (nx odd) or last two
    // columns (nx even) have to be skipped when copying in the input
    auto real_buffer = reinterpret_cast<dreal*>(buffer);

    // copy over input to output array
    auto const rowsize = 2*complex_columns;
#pragma omp parallel for shared(real_buffer, realdata)
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            real_buffer[i*rowsize + j] = realdata[i*ny + j];
        }
    }
    return buffer;
}

/**
 * `realdata`: nx * nx matrix
 * output: a buffer in the format described at http://fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data. It needs to be freed by `fftw_free`, not the ordinary `free`!
 */
dcomplex* galario_fft2d(int nx, int ny, dreal* realdata) {
    // TODO update as for CPU
#ifdef __CUDACC__
    dcomplex *data_d;
    size_t nbytes = sizeof(dcomplex)*nx*(ny/2 + 1);
    CCheck(cudaMalloc((void**)&data_d, nbytes));

    // TODO copy data on device to respect padding for in-place trafo
    CCheck(cudaMemcpy(data_d, data, nbytes, cudaMemcpyHostToDevice));
    FIXME_copy_over();
    fft_d(nx, ny, (dcomplex*) data_d);

    CCheck(cudaMemcpy(data, data_d, nbytes, cudaMemcpyDeviceToHost));
    CCheck(cudaFree(data_d));
#else
    auto buffer = copy_real_to_buffer(nx, ny, realdata);
    fft_h(nx, ny, buffer);
    return buffer;
#endif
}

void* _galario_fft2d(int nx, int ny, void* data) {
    return galario_fft2d(nx, ny, static_cast<dreal*>(data));
}

/**
 * Shift quadrants of the square image. Swap the upper-left quadrant with the
 * lower-right quadrant and the upper-right with the lower-left quadrant.
 *
 * To avoid if statements, we do two swaps.
 *
 * For cache efficiency, may have to do loop tiling; i.e., the source and target
 * should fit into the cache. If the image is too large, only part of a row may
 * fit. This is a responsibility of the caller.
 **/
// `a` is a matrix (size: nx^2)
#ifdef __CUDACC__
__host__ __device__
#endif
inline void shift_core(int const idx_x, int const idx_y, int const nx, int const ny, dreal* const __restrict__ a) {
    /* row-wise access */

    // from upper left to lower right
    auto const src_ul = idx_x*ny + idx_y;
    auto const tgt_ul = src_ul + nx*(ny+1)/2;

    // from upper right to lower left
    auto const src_ur = src_ul + ny/2;
    auto const tgt_ur = tgt_ul - ny/2;

    // swap the values
    auto const tmp_ul = a[src_ul];
    a[src_ul] = a[tgt_ul];
    a[tgt_ul] = tmp_ul;

    auto const tmp_ur = a[src_ur];
    a[src_ur] = a[tgt_ur];
    a[tgt_ur] = tmp_ur;
}

void shift_h(int const nx, int const ny, dreal* const __restrict__ a) {
#pragma omp parallel for
    for (auto x = 0; x < nx/2; ++x) {
        for (auto y = 0; y < ny/2; ++y) {
            shift_core(x, y, nx, ny, a);
        }
    }
}

// TODO make shift_d and shift_h the same function, with ifdef __CUDACC__ inside.
/**
 * grid stride loop
 */
#ifdef __CUDACC__
__global__ void shift_d(int const nx, dreal* const __restrict__ a) {
  // indices
  int const x0 = blockDim.x * blockIdx.x + threadIdx.x;
  int const y0 = blockDim.y * blockIdx.y + threadIdx.y;

  // stride
  int const sx = blockDim.x * gridDim.x;
  int const sy = blockDim.y * gridDim.y;

  for (auto x = x0; x < nx/2; x += sx) {
    for (auto y = y0; y < ny/2; y += sy) {
      shift_core(x, y, nx, a);
    }
  }
}
#endif

void galario_fftshift(int nx, int ny, dreal* data) {
#ifdef __CUDACC__
    dreal *data_d;
    size_t nbytes = sizeof(dreal)*nx*ny;
    CCheck(cudaMalloc((void**)&data_d, nbytes));
    CCheck(cudaMemcpy(data_d, data, nbytes, cudaMemcpyHostToDevice));

    shift_d<<<dim3(nx/2/galario_threads_per_block()+1, nx/2/galario_threads_per_block()+1), dim3(galario_threads_per_block(), galario_threads_per_block())>>>(nx, ny, (dreal*) data_d);

    CCheck(cudaDeviceSynchronize());
    CCheck(cudaMemcpy(data, data_d, nbytes, cudaMemcpyDeviceToHost));
    CCheck(cudaFree(data_d));
#else
    shift_h(nx, ny, data);
#endif
}

void _galario_fftshift(int nx, int ny, void* data) {
    galario_fftshift(nx, ny, static_cast<dreal*>(data));
}

/**
 * Shift quadrants of a rectangular matrix of size (nx, ny).
 * Swap the upper quadrant with the lower quadrant.
 *
 * For cache efficiency, may have to do loop tiling; i.e., the source and target
 * should fit into the cache. If the image is too large, only part of a row may
 * fit. This is a responsibility of the caller.
 **/
#ifdef __CUDACC__
__host__ __device__
#endif
inline void shift_axis0_core(int const idx_x, int const idx_y, int const nx, int const ny, dcomplex* const __restrict__ a) {
    /* row-wise access */

    // from top-half to bottom-half
    auto const src_u = idx_x*ny + idx_y;
    auto const tgt_u = src_u + nx/2*ny;

    // swap the values
    auto tmp = a[src_u];
    a[src_u] = a[tgt_u];
    a[tgt_u] = tmp;
}

void shift_axis0_h(int const nx, int const ny, dcomplex *const a) {

#pragma omp parallel for
    for (auto x = 0; x < nx/2; ++x) {
        for (auto y = 0; y < ny; ++y) {
            shift_axis0_core(x, y, nx, ny, a);
        }
    }
}

// TODO make shift_d and shift_h the same function, with ifdef __CUDACC__ inside.
/**
 * grid stride loop
 */
#ifdef __CUDACC__
__global__ void shift_axis0_d(int const nx, int const ny, dcomplex* const __restrict__ a) {
  // indices
  int const x0 = blockDim.x * blockIdx.x + threadIdx.x;
  int const y0 = blockDim.y * blockIdx.y + threadIdx.y;

  // stride
  int const sx = blockDim.x * gridDim.x;
  int const sy = blockDim.y * gridDim.y;

  for (auto x = x0; x < nx/2; x += sx) {
    for (auto y = y0; y < ny; y += sy) {
      shift_axis0_core(x, y, nx, ny, a);
    }
  }
}
#endif

void galario_fftshift_axis0(int nx, int ny, dcomplex* data) {
#ifdef __CUDACC__
    dcomplex *data_d;
    size_t nbytes = sizeof(dcomplex)*nx*(ny/2+1);
    CCheck(cudaMalloc((void**)&data_d, nbytes));
    CCheck(cudaMemcpy(data_d, data, nbytes, cudaMemcpyHostToDevice));

    shift_axis0_d<<<dim3(nx/2/galario_threads_per_block()+1, ny/galario_threads_per_block()+1), dim3(galario_threads_per_block(), galario_threads_per_block())>>>(nx, ny, (dcomplex*) data_d);

    CCheck(cudaDeviceSynchronize());
    CCheck(cudaMemcpy(data, data_d, nbytes, cudaMemcpyDeviceToHost));
    CCheck(cudaFree(data_d));
#else
    shift_axis0_h(nx, ny, data);
#endif
}

void _galario_fftshift_axis0(int nx, int ny, void* data) {
    galario_fftshift_axis0(nx, ny, static_cast<dcomplex*>(data));
}


/**
 * Bilinear interpolation in 2D according to Numerical Recipes.
 *
 * fint(indu, indv) = (1-t)(1-u)y0 + t(1-u)y1 + t*u*y2 + (1-t)*u*y3
 *
 * where `y0` is bottom-left grid point, `y1` the bottom-right etc. and `t` and
 * `u` are the fractions of the desired location from left (bottom) to right
 * (upper) grid point.
 *
 * @param ny number of columns
 * @param data Fourier transform of the image [nx * nx].
 * @param nd number of data points.
 * @param indu int(indu[i]) is the closest index into data smaller than the x value of the data point. The offset to int(indu[i]) gives the position in the pixel [nd]
 * @param indv same as indu but for the v direction [nd].
 * @param fint The image values obtained with bilinear interpolation at the data point values [nd].
 */
//    we need to re-define CUCSUB, CUCADD, CUCMUL if __CUDACC__ not defined.
//    suggestion: change CUCSUB -> CSUB ... that, CSUB=CUCSUB ifdef __CUDACC__, else CSUB: subtract between two complex numbers
#ifdef __CUDACC__
__host__ __device__
#endif
inline void interpolate_core(int const idx_x, int const ny, dcomplex* const __restrict__ data, int const nd, dreal* const __restrict__ indv, dreal* const __restrict__ indu,  dcomplex* __restrict__ fint) {

    // notations as in (3.6.5) of Numerical Recipes. They put the origin in the
    // lower-left.
    int const fl_u = floor(indu[idx_x]);
    int const fl_v = floor(indv[idx_x]);
    dcomplex const t = {indu[idx_x] - fl_u, 0.0};
    dcomplex const u = {indv[idx_x] - fl_v, 0.0};

    // linear index of y0
    int const base = fl_v + fl_u * ny;

    // y0 + y2
    dcomplex const add1 = CMPLXADD(data[base], data[base+ny+1]);
    // y3 + y1
    dcomplex const add2 = CMPLXADD(data[base+ny], data[base+1]);
    // y0+y2-y3-y1
    dcomplex const df1 = CMPLXSUB(add1, add2);
    //
    dcomplex const mul1 = CMPLXMUL(u, df1);
    dcomplex const term1 = CMPLXMUL(t, mul1);
    dcomplex const term2_sub = CMPLXSUB(data[base+ny], data[base]);
    dcomplex const term2 = CMPLXMUL(t, term2_sub);
    // u*y3
    dcomplex const term3_sub = CMPLXSUB(data[base+1], data[base]);
    dcomplex const term3 = CMPLXMUL(u, term3_sub);

    dcomplex const final_add2 = CMPLXADD(term2, term3);
    dcomplex const final_add1 = CMPLXADD(term1, final_add2);

    fint[idx_x] = CMPLXADD(data[base], final_add1);
}

#ifdef __CUDACC__
__global__ void interpolate_d(int const ny, dcomplex* const __restrict__ data, int const nd, dreal* const indu, dreal* const indv, dcomplex* __restrict__ fint)
{
    //index
    int const idx_x0 = blockDim.x * blockIdx.x + threadIdx.x;

    // stride
    int const sx = blockDim.x * gridDim.x;

    for (auto idx_x = idx_x0; idx_x < nd; idx_x += sx)
    {
        interpolate_core(idx_x, ny, data, nd, indu, indv, fint);
     }
}
#endif

void interpolate_h(int const ny, dcomplex *const data, int const nd, dreal *const indu, dreal *const indv,
                   dcomplex *fint) {
#pragma omp parallel for
    for (auto idx = 0; idx < nd; ++idx)
    {
        interpolate_core(idx, ny, data, nd, indu, indv, fint);
    }
}

void galario_interpolate(int nx, int ny, dcomplex *data, int nd, dreal *u, dreal *v, dcomplex *fint) {
#ifdef __CUDACC__
    // copy the image data
    dcomplex *data_d;
    size_t nbytes = sizeof(dcomplex)*nx*ny;
    CCheck(cudaMalloc((void**)&data_d, nbytes));
    CCheck(cudaMemcpy(data_d, data, nbytes, cudaMemcpyHostToDevice));

    // copy u,v and reserve memory for the interpolated values
    dreal *u_d, *v_d;
    dcomplex *fint_d;
    size_t nbytes_nd = sizeof(dreal)*nd;

    CCheck(cudaMalloc((void**)&u_d, nbytes_nd));
    CCheck(cudaMemcpy(u_d, u, nbytes_nd, cudaMemcpyHostToDevice));

    CCheck(cudaMalloc((void**)&v_d, nbytes_nd));
    CCheck(cudaMemcpy(v_d, v, nbytes_nd, cudaMemcpyHostToDevice));

    int nbytes_fint = sizeof(dcomplex) * nd;
    CCheck(cudaMalloc((void**)&fint_d, nbytes_fint));

    // oversubscribe blocks because we don't know if #(data points) divisible by nthreads
    interpolate_d<<<nd / galario_threads_per_block() + 1, galario_threads_per_block()>>>(ny, (dcomplex*) data_d, nd, (dreal*)u_d, (dreal*)v_d, (dcomplex*) fint_d);

    CCheck(cudaDeviceSynchronize());

    // retrieve interpolated values
    CCheck(cudaMemcpy(fint, fint_d, nbytes_fint, cudaMemcpyDeviceToHost));

    // free memories
    CCheck(cudaFree(data_d));
    CCheck(cudaFree(u_d));
    CCheck(cudaFree(v_d));
    CCheck(cudaFree(fint_d));
#else
    interpolate_h(ny, data, nd, u, v, fint);
#endif
}

void _galario_interpolate(int nx, int ny, void *data, int nd, void *u, void *v, void *fint) {
    galario_interpolate(nx, ny, static_cast<dcomplex*>(data), nd, static_cast<dreal*>(u),
                        static_cast<dreal*>(v), static_cast<dcomplex*>(fint));
}

// APPLY_PHASE TO SAMPLED POINTS //
#ifdef __CUDACC__
__host__ __device__
#endif
inline void apply_phase_sampled_core(int const idx_x, dreal* const u, dreal* const v, dcomplex* const __restrict__ fint, dreal const dRA, dreal const dDec) {

    dreal const angle = u[idx_x]*dRA + v[idx_x]*dDec;

    dcomplex const phase = dcomplex{dreal(cos(angle)), dreal(sin(angle))};

    fint[idx_x] = CMPLXMUL(fint[idx_x], phase);
}


#ifdef __CUDACC__
__global__ void apply_phase_sampled_d(dreal dRA, dreal dDec, int const nd, dreal* const u, dreal* const v, dcomplex* __restrict__ fint) {

    if ((dRA==0) || (dDec==0)) {
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
#endif

void apply_phase_sampled_h(dreal dRA, dreal dDec, int const nd, dreal* const u, dreal* const v, dcomplex* __restrict__ fint) {

    if ((dRA==0) || (dDec==0)) {
        return;
    }

    dRA *= 2.*(dreal)M_PI;
    dDec *= 2.*(dreal)M_PI;

#pragma omp parallel for shared(dRA, dDec) schedule(static)
    for (auto x = 0; x < nd; ++x) {
        apply_phase_sampled_core(x, u, v, fint, dRA, dDec);
    }
}

void galario_apply_phase_sampled(dreal dRA, dreal dDec, int const nd, dreal* const u, dreal* const v, dcomplex* __restrict__ fint) {
#ifdef __CUDACC__

     size_t nbytes_d_complex = sizeof(dcomplex)*nd;
     size_t nbytes_d_dreal = sizeof(dreal)*nd;

     dreal *u_d, *v_d;
     dcomplex *fint_d;

     CCheck(cudaMalloc((void**)&u_d, nbytes_d_dreal));
     CCheck(cudaMemcpy(u_d, u, nbytes_d_dreal, cudaMemcpyHostToDevice));

     CCheck(cudaMalloc((void**)&v_d, nbytes_d_dreal));
     CCheck(cudaMemcpy(v_d, v, nbytes_d_dreal, cudaMemcpyHostToDevice));

     CCheck(cudaMalloc((void**)&fint_d, nbytes_d_complex));
     CCheck(cudaMemcpy(fint_d, fint, nbytes_d_complex, cudaMemcpyHostToDevice));

     apply_phase_sampled_d<<<nd/galario_threads_per_block()+1, nd/galario_threads_per_block()+1>>>(dRA, dDec, nd, u_d, v_d, fint_d);

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

// APPLY_PHASE 2D //
#ifdef __CUDACC__
__host__ __device__
#endif
inline void apply_phase_core(int const idx_x, int const idx_y, int const nx, int const ny, dcomplex* const __restrict__ data, dreal const dRA, dreal const dDec) {

    dreal const u = idx_x/(dreal)nx - 0.5;
    dreal const v = idx_y/(dreal)ny - 0.5;
    dreal const angle = u*dRA + v*dDec;
    // TODO should nx -> ny? Looks like column-wise access, we don't want that
    auto const idx = idx_x + idx_y*nx;

    dcomplex const phase = dcomplex{dreal(cos(angle)), dreal(sin(angle))};
    data[idx] = CMPLXMUL(data[idx], phase);
}

#ifdef __CUDACC__
__global__ void apply_phase_d(int const nx, int const ny, dcomplex* const __restrict__ data, dreal dRA, dreal dDec) {

    if ((dRA==0) || (dDec==0)) {
        return;
    }

    dRA *= 2.*(dreal)M_PI;
    dDec *= 2.*(dreal)M_PI;

    // indices
    int const idx_x0 = blockDim.x * blockIdx.x + threadIdx.x;
    int const idx_y0 = blockDim.y * blockIdx.y + threadIdx.y;

    // stride
    int const sx = blockDim.x * gridDim.x;
    int const sy = blockDim.y * gridDim.y;

    for (auto x = idx_x0; x < nx; x += sx) {
        for (auto y = idx_y0; y < ny; y += sy) {
            apply_phase_core(x, y, nx, ny, data, dRA, dDec);
        }
    }
}
#endif

void apply_phase_h(int const nx, int const ny, dcomplex* const __restrict__ data, dreal dRA, dreal dDec) {

    if ((dRA==0) || (dDec==0)) {
        return;
    }

    dRA *= 2.*(dreal)M_PI;
    dDec *= 2.*(dreal)M_PI;

#pragma omp parallel for shared(dRA, dDec) schedule(static)
    for (auto x = 0; x < nx; ++x) {
        for (auto y = 0; y < ny; ++y) {
            apply_phase_core(x, y, nx, ny, data, dRA, dDec);
        }
    }
}

void galario_apply_phase_2d(int nx, int ny, dcomplex* data, dreal dRA, dreal dDec) {
#ifdef __CUDACC__
    dcomplex *data_d;

    size_t nbytes = sizeof(dcomplex)*nx*nx;

    CCheck(cudaMalloc((void**)&data_d, nbytes));
    CCheck(cudaMemcpy(data_d, data, nbytes, cudaMemcpyHostToDevice));

    apply_phase_d<<<dim3(nx/galario_threads_per_block()+1, nx/galario_threads_per_block()+1), dim3(galario_threads_per_block(), galario_threads_per_block())>>>(nx, ny, (dcomplex*) data_d, dRA, dDec);

    CCheck(cudaDeviceSynchronize());
    CCheck(cudaMemcpy(data, data_d, nbytes, cudaMemcpyDeviceToHost));
    CCheck(cudaFree(data_d));
#else
    apply_phase_h(nx, ny, data, dRA, dDec);
#endif
}

void _galario_apply_phase_2d(int nx, int ny, void* data, dreal dRA, dreal dDec) {
    galario_apply_phase_2d(nx, ny, static_cast<dcomplex*>(data), dRA, dDec);
}

/**
 * Indices of data points of (rotated) data points
 * into the Fourier transform of the image.
 *
 * The integer part is the index into the transformed image and identifies a point i,
 * the remainder is the fraction to move from point i towards point (i+1) to arrive at the actual data point.
 *
 * Assumptions:
 * 1. input arrays are assumed contiguous.
 * 2. the pixel size is uniform and the same in u and v direction
 * 3. the extent of the pixel_centers is the same in u and v direction but need not be square around origin
 *
 * @param nx number of pixels
 * @param u part of data points in u direction
 * @param v part of data points in v direction
 * @param indu Index in u direction [output]
 * @param indv Index in v direction [output]
*
 */
#ifdef __CUDACC__
__host__ __device__ inline void uv_idx_core
#else
inline void uv_idx_core
#endif
        (int const i, int const half_nx, dreal const du, dreal const* const u, dreal const* const v, dreal* const __restrict__ indu, dreal*  const __restrict__ indv) {
    indu[i] = half_nx + u[i]/du;
    indv[i] = half_nx + v[i]/du;
}


#ifdef __CUDACC__
__global__ void uv_idx_d(const int nx, int ny, dreal du, int nd, dreal const* u, dreal const* v, dreal* const __restrict__ indu, dreal* const __restrict__ indv)
    {
        // index
        int const i0 = blockDim.x * blockIdx.x + threadIdx.x;

        // stride
        int const si = blockDim.x * gridDim.x;

        int const half_nx = nx/2;

        for (auto i = i0; i < nd; i += si) {
            uv_idx_core(i, half_nx, du, u, v, indu, indv);
        }
    }
#endif

void uv_idx_h(const int nx, int ny, dreal du, int nd, dreal const *u, dreal const *v, dreal *const __restrict__ indu,
              dreal *const __restrict__ indv) {

    int const half_nx = nx/2;

#pragma omp parallel for
    for (auto i = 0; i < nd; ++i) {
        uv_idx_core(i, half_nx, du, u, v, indu, indv);
    }
}


void galario_get_uv_idx(int nx, int ny, dreal du, int nd, dreal* u, dreal* v, dreal* indu, dreal* indv) {
    assert(nx >= 2);

#ifdef __CUDACC__
    dreal *u_d, *v_d;
    size_t nbytes_nd = sizeof(dreal)*nd;

    CCheck(cudaMalloc((void**)&u_d, nbytes_nd));
    CCheck(cudaMemcpy(u_d, u, nbytes_nd, cudaMemcpyHostToDevice));
    CCheck(cudaMalloc((void**)&v_d, nbytes_nd));
    CCheck(cudaMemcpy(v_d, v, nbytes_nd, cudaMemcpyHostToDevice));

    dreal *indu_d, *indv_d;
    CCheck(cudaMalloc((void**)&indu_d, nbytes_nd));
    CCheck(cudaMalloc((void**)&indv_d, nbytes_nd));

    uv_idx_d<<<nd / galario_threads_per_block() + 1, galario_threads_per_block()>>>(nx, ny, du, nd, u_d, v_d, indu_d, indv_d);

    CCheck(cudaDeviceSynchronize());

    // retrieve indices
    CCheck(cudaMemcpy(indu, indu_d, nbytes_nd, cudaMemcpyDeviceToHost));
    CCheck(cudaMemcpy(indv, indv_d, nbytes_nd, cudaMemcpyDeviceToHost));

    // free memories
    CCheck(cudaFree(u_d));
    CCheck(cudaFree(v_d));
    CCheck(cudaFree(indu_d));
    CCheck(cudaFree(indv_d));
#else
    uv_idx_h(nx, ny, du, nd, u, v, indu, indv);
#endif
}

void _galario_get_uv_idx(int nx, int ny, dreal du, int nd, void* u, void* v, void* indu, void* indv) {
    galario_get_uv_idx(nx, ny, du, nd, static_cast<dreal*>(u), static_cast<dreal*>(v), static_cast<dreal*>(indu), static_cast<dreal*>(indv));
}


#ifdef __CUDACC__
__host__ __device__
#endif
inline void uv_idx_R2C_core(int const i, int const half_nx, dreal const du, dreal const* const u, dreal const* const v, dreal* const __restrict__ indu, dreal*  const __restrict__ indv) {
    indu[i] = fabs(u[i])/du;
    indv[i] = half_nx + v[i]/du;

    if (u[i] < 0.) indv[i] *= -1.;
}


#ifdef __CUDACC__
__global__ void uv_idx_R2C_d(const int nx, int ny, dreal du, int nd, dreal const* u, dreal const* v, dreal* const __restrict__ indu, dreal* const __restrict__ indv)
    {
        // index
        int const i0 = blockDim.x * blockIdx.x + threadIdx.x;

        // stride
        int const si = blockDim.x * gridDim.x;

        int const half_nx = nx/2;

        for (auto i = i0; i < nd; i += si) {
            uv_idx_R2C_core(i, half_nx, du, u, v, indu, indv);
        }
    }
#endif

void uv_idx_R2C_h(const int nx, int ny, dreal du, int nd, dreal const *u, dreal const *v, dreal *const __restrict__ indu,
              dreal *const __restrict__ indv) {

    int const half_nx = nx/2;

#pragma omp parallel for
    for (auto i = 0; i < nd; ++i) {
        uv_idx_R2C_core(i, half_nx, du, u, v, indu, indv);
    }
}

void galario_get_uv_idx_R2C(int nx, int ny, dreal du, int nd, dreal* u, dreal* v, dreal* indu, dreal* indv) {
    assert(nx >= 2);

#ifdef __CUDACC__
    dreal *u_d, *v_d;
    size_t nbytes_nd = sizeof(dreal)*nd;

    CCheck(cudaMalloc((void**)&u_d, nbytes_nd));
    CCheck(cudaMemcpy(u_d, u, nbytes_nd, cudaMemcpyHostToDevice));
    CCheck(cudaMalloc((void**)&v_d, nbytes_nd));
    CCheck(cudaMemcpy(v_d, v, nbytes_nd, cudaMemcpyHostToDevice));

    dreal *indu_d, *indv_d;
    CCheck(cudaMalloc((void**)&indu_d, nbytes_nd));
    CCheck(cudaMalloc((void**)&indv_d, nbytes_nd));

    uv_idx_R2C_d<<<nd / galario_threads_per_block() + 1, galario_threads_per_block()>>>(nx, ny, du, nd, u_d, v_d, indu_d, indv_d);

    CCheck(cudaDeviceSynchronize());

    // retrieve indices
    CCheck(cudaMemcpy(indu, indu_d, nbytes_nd, cudaMemcpyDeviceToHost));
    CCheck(cudaMemcpy(indv, indv_d, nbytes_nd, cudaMemcpyDeviceToHost));

    // free memories
    CCheck(cudaFree(u_d));
    CCheck(cudaFree(v_d));
    CCheck(cudaFree(indu_d));
    CCheck(cudaFree(indv_d));
#else
    uv_idx_R2C_h(nx, ny, du, nd, u, v, indu, indv);
#endif
}

void _galario_get_uv_idx_R2C(int nx, int ny, dreal du, int nd, void* u, void* v, void* indu, void* indv) {
    galario_get_uv_idx_R2C(nx, ny, du, nd, static_cast<dreal*>(u), static_cast<dreal*>(v), static_cast<dreal*>(indu), static_cast<dreal*>(indv));
}

#ifdef __CUDACC__
inline void sample_d(int nx, int ny, dcomplex* data_d, dreal dRA, dreal dDec, int nd, dreal du, dreal* u_d, dreal* v_d, dreal* indu_d, dreal* indv_d, dcomplex* fint_d)
{

    const dreal arcsec_to_rad = (dreal)M_PI / 3600. / 180.;
    dRA *= arcsec_to_rad;
    dDec *= arcsec_to_rad;

     // ################################
     // ########### KERNELS ############
     // ################################
     // Kernel for shift --> FFT --> shift
     shift_d<<<dim3(nx/2/galario_threads_per_block()+1, nx/2/galario_threads_per_block()+1), dim3(galario_threads_per_block(), galario_threads_per_block())>>>(nx, ny, data_d);
     fft_d(nx, ny, (dcomplex*) data_d);
     shift_d<<<dim3(nx/2/galario_threads_per_block()+1, nx/2/galario_threads_per_block()+1), dim3(galario_threads_per_block(), galario_threads_per_block())>>>(nx, ny, data_d);
     CCheck(cudaDeviceSynchronize());

     // Kernel for uv_idx and interpolate
     uv_idx_d<<<nd / galario_threads_per_block() + 1, galario_threads_per_block()>>>(nx, ny, du, nd, u_d, v_d, indu_d, indv_d);

     // oversubscribe blocks because we don't know if #(data points) divisible by nthreads
     interpolate_d<<<nd / galario_threads_per_block() + 1, galario_threads_per_block()>>>(nx, ny, data_d, nd, indu_d, indv_d, fint_d);

     // apply phase to the sampled points
     apply_phase_sampled_d<<<nd / galario_threads_per_block() + 1, galario_threads_per_block()>>>(dRA, dDec, nd, u_d, v_d, fint_d);
}

/**
 * Return device pointer to complex image made from real image on the host.
 *
 * Caller is responsible for freeing the device memory.
 */
dcomplex* copy_real_to_device(int nx, int ny, const dreal* realdata) {
    dcomplex *data_d;
    CCheck(cudaMalloc((void**)&data_d, nx*nbytesrow));

    // TODO hide latency with asynchronous copies
    /*  copy rows individually to skip the padding elements in the
        destination array */
    auto const rowsize = (ny/2+1) << 1;
    auto const nbytesrow = sizeof(dreal)*ny;
    for (auto i=0; i < nx; ++i) {
        CCheck(cudaMemcpy(data_d + i*rowsize, realdata + i*ny, nbytesrow, cudaMemcpyHostToDevice));
    }

    return data_d;
}
#endif

/**
 * return result in `fint`
 */
void galario_sample(int nx, dreal* realdata, dreal dRA, dreal dDec, dreal du, int nd, dreal* u, dreal* v, dcomplex* fint) {
    // Initialization for uv_idx and interpolate
    assert(nx >= 2);

#ifdef __CUDACC__
    // ################################
    // ### ALLOCATION, INITIALIZATION ###
    // ################################

    dcomplex* data_d = copy_real_to_device(nx, ny, realdata);

    /* async memory copy:, see issue https://github.com/mtazzari/galario/issues/40
       TODO copy memory asynchronously or create streams to define dependencies
       use nonzero cudaStream_t
       kernel<<< blocks, threads, bytes=0, stream =! 0>>>();

       all cufft calls are asynchronous, can specify the stream explicitly (cf. doc)
       same for cublas
       draw dependencies on paper: first thing is to do fft while other data is transferred
    */

    dreal *u_d, *v_d;
    size_t nbytes_ndat = sizeof(dreal)*nd;
    dreal *indu_d, *indv_d;
    CCheck(cudaMalloc((void**)&indu_d, nbytes_ndat));
    CCheck(cudaMalloc((void**)&indv_d, nbytes_ndat));

    CCheck(cudaMalloc((void**)&u_d, nbytes_ndat));
    CCheck(cudaMemcpy(u_d, u, nbytes_ndat, cudaMemcpyHostToDevice));
    CCheck(cudaMalloc((void**)&v_d, nbytes_ndat));
    CCheck(cudaMemcpy(v_d, v, nbytes_ndat, cudaMemcpyHostToDevice));

    // take indu_d and indv_d as u and v (no need to copy them) and reserve memory for the interpolated values
    dcomplex *fint_d;
    int nbytes_fint = sizeof(dcomplex) * nd;
    CCheck(cudaMalloc((void**)&fint_d, nbytes_fint));

    // do the work on the gpu
    sample_d(nx, data_d, dRA, dDec, nd, du, u_d, v_d, indu_d, indv_d, fint_d);

    // ################################
    // ########### TRANSFER DATA ######
    // ################################
    CCheck(cudaDeviceSynchronize());
    CCheck(cudaMemcpy(fint, fint_d, nbytes_fint, cudaMemcpyDeviceToHost));

    // ################################
    // ########### CLEANUP ############
    // ################################
    CCheck(cudaFree(data_d));
    CCheck(cudaFree(u_d));
    CCheck(cudaFree(v_d));
    CCheck(cudaFree(indu_d));
    CCheck(cudaFree(indv_d));
    CCheck(cudaFree(fint_d));
#else

    const dreal arcsec_to_rad = (dreal)M_PI / 3600. / 180.;
    dRA *= arcsec_to_rad;
    dDec *= arcsec_to_rad;

    // TODO real shift
    // TODO first copy, then shift, so input remains untouched
    shift_h(nx, ny, realdata);

    auto data = copy_real_to_buffer(nx, ny, realdata);

    fft_h(nx, ny, data);

    shift_axis0_h(nx, ny/2+1, data);

    // uv_idx_h
    auto indu = (dreal*) malloc(sizeof(dreal)*nd);
    auto indv = (dreal*) malloc(sizeof(dreal)*nd);

    uv_idx_h(nx, ny, du, nd, u, v, indu, indv);

    // interpolate
    interpolate_h(ny, data, nd, indu, indv, fint);

    // apply phase to the sampled points
    apply_phase_sampled_h(dRA, dDec, nd, u, v, fint);

    free(indv);
    free(indu);
    fftw_free(data);
#endif
}

void _galario_sample(int nx, void* data, dreal dRA, dreal dDec, dreal du, int nd, void* u, void* v, void* fint) {
    galario_sample(nx, static_cast<dreal*>(data), dRA, dDec, du, nd, static_cast<dreal*>(u), static_cast<dreal*>(v), static_cast<dcomplex*>(fint));
}

/**
 * Compute weighted difference between observations (`fobs_re` and `fobs_im`) and model predictions `fint`, write to `fint`
 */
#ifdef __CUDACC__
__host__ __device__
#endif
inline void diff_weighted_core(int const idx_x, int const nd, dreal const* const __restrict__ fobs_re,
                               dreal const* const __restrict__ fobs_im, dcomplex* const __restrict__ fint,
                               dreal const* const __restrict__ weights)
{
    dcomplex const fobs_cmplx = dcomplex { fobs_re[idx_x], fobs_im[idx_x] };
    dcomplex const sqrt_w_cmplx = dcomplex { SQRT(weights[idx_x]), 0.0 } ;
    fint[idx_x] = CMPLXSUB(fint[idx_x], fobs_cmplx);
    fint[idx_x] = CMPLXMUL(fint[idx_x], sqrt_w_cmplx);
}

#ifdef __CUDACC__
__global__ void diff_weighted_d
(int const nd, dreal const* const __restrict__ fobs_re, dreal const* const __restrict__ fobs_im, dcomplex* const __restrict__ fint, dreal const* const __restrict__ weights)
{
    //index
    int const idx_x0 = blockDim.x * blockIdx.x + threadIdx.x;

    // stride
    int const sidx_x = blockDim.x * gridDim.x;

    for (auto idx_x = idx_x0; idx_x < nd; idx_x += sidx_x) {
        diff_weighted_core(idx_x, nd, fobs_re, fobs_im, fint, weights);
    }
}
#endif

void diff_weighted_h
        (int const nd, dreal const* const fobs_re, dreal const* const fobs_im, dcomplex* const fint, dreal const* const weights)
{
#pragma omp parallel for
    for (auto idx = 0; idx < nd; ++idx) {
        diff_weighted_core(idx, nd, fobs_re, fobs_im, fint, weights);
    }
}

#ifdef __CUDACC__
void reduce_chi2_d
(int nd, dreal const* const __restrict__ fobs_re, dreal const* const __restrict__ fobs_im, dcomplex * const __restrict__ fint, dreal const* const __restrict__ weights, dreal* chi2)
{
    cublasHandle_t handle;
    CBlasCheck(cublasCreate(&handle));

    /* compute weighted difference */
    diff_weighted_d<<<nd / galario_threads_per_block() + 1, galario_threads_per_block()>>>(nd, fobs_re, fobs_im, fint, weights);

    // only device pointers! maybe not ... check with jiri
    // compute the Euclidean norm
    CUBLASNRM2(handle, nd, fint, 1, chi2);
    // but we want the square of the norm
    *chi2 *= *chi2;

    CBlasCheck(cublasDestroy(handle));
}
#endif

void galario_reduce_chi2(int nd, dreal* fobs_re, dreal* fobs_im, dcomplex* fint, dreal* weights, dreal* chi2) {
#ifdef __CUDACC__

    /* allocate and copy */
     dreal *fobs_re_d, *fobs_im_d, *weights_d;
     size_t nbytes_nd = sizeof(dreal)*nd;

     CCheck(cudaMalloc((void**)&fobs_re_d, nbytes_nd));
     CCheck(cudaMemcpy(fobs_re_d, fobs_re, nbytes_nd, cudaMemcpyHostToDevice));

     CCheck(cudaMalloc((void**)&fobs_im_d, nbytes_nd));
     CCheck(cudaMemcpy(fobs_im_d, fobs_im, nbytes_nd, cudaMemcpyHostToDevice));

     CCheck(cudaMalloc((void**)&weights_d, nbytes_nd));
     CCheck(cudaMemcpy(weights_d, weights, nbytes_nd, cudaMemcpyHostToDevice));

     dreal *chi2_d;
     size_t nbytes_chi2 = sizeof(dreal);
     CCheck(cudaMalloc((void**)&chi2_d, nbytes_chi2));

     dcomplex* fint_d;
     size_t nbytes_fint = sizeof(dcomplex) * nd;
     CCheck(cudaMalloc((void**)&fint_d, nbytes_fint));
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

    // TODO: if available, use BLAS (mkl?) functions cblas_scnrm2 or cblas_dznrm2 for float/double complex
    // compute the Euclidean norm
    dreal y = 0.;
#pragma omp parallel for reduction(+:y)
    for (auto i = 0; i < nd; ++i) {
        dcomplex const x = fint[i];
        y += real(CMPLXMUL(x, conj(x)));
    }
    *chi2 = y;

#endif
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

void galario_chi2(int nx, dreal* realdata, dreal dRA, dreal dDec, dreal du, int nd, dreal* u, dreal* v, dreal* fobs_re, dreal* fobs_im, dreal* weights, dreal* chi2) {

    // dcomplex* data_cmplx = (dcomplex*) data;  // casting all the times or only once?
    // Initilization for uv_idx and interpolate
    assert(nx >= 2);

#ifdef __CUDACC__
     // ################################
     // ### ALLOCATION, INITIALIZATION ###
     // ################################

    dcomplex *data_d = copy_real_to_device(nx, realdata);

     /* async memory copy:
      TODO copy memory asynchronously or create streams to define dependencies
      use nonzero cudaStream_t
      kernel<<< blocks, threads, bytes=0, stream =! 0>>>();

      all cufft calls are asynchronous, can specify the stream explicitly (cf. doc)
      same for cublas
      draw dependcies on paper: first thing is to do fft while other data is transferred
     */

     dreal *u_d, *v_d;
     size_t nbytes_ndat = sizeof(dreal)*nd;
     dreal *indu_d, *indv_d;
     CCheck(cudaMalloc((void**)&indu_d, nbytes_ndat));
     CCheck(cudaMalloc((void**)&indv_d, nbytes_ndat));

     CCheck(cudaMalloc((void**)&u_d, nbytes_ndat));
     CCheck(cudaMemcpy(u_d, u, nbytes_ndat, cudaMemcpyHostToDevice));
     CCheck(cudaMalloc((void**)&v_d, nbytes_ndat));
     CCheck(cudaMemcpy(v_d, v, nbytes_ndat, cudaMemcpyHostToDevice));

     // take indu_d and indv_d as u and v (no need to copy them) and reserve memory for the interpolated values
     dcomplex *fint_d;
     int nbytes_fint = sizeof(dcomplex) * nd;
     CCheck(cudaMalloc((void**)&fint_d, nbytes_fint));

     // Initialization for comparison and chi square computation
     /* allocate and copy observational data */
     dreal *fobs_re_d, *fobs_im_d, *weights_d;

     CCheck(cudaMalloc((void**)&fobs_re_d, nbytes_ndat));
     CCheck(cudaMemcpy(fobs_re_d, fobs_re, nbytes_ndat, cudaMemcpyHostToDevice));

     CCheck(cudaMalloc((void**)&fobs_im_d, nbytes_ndat));
     CCheck(cudaMemcpy(fobs_im_d, fobs_im, nbytes_ndat, cudaMemcpyHostToDevice));

     CCheck(cudaMalloc((void**)&weights_d, nbytes_ndat));
     CCheck(cudaMemcpy(weights_d, weights, nbytes_ndat, cudaMemcpyHostToDevice));

     sample_d(nx, data_d, dRA, dDec, nd, du, u_d, v_d, indu_d, indv_d, fint_d);
     reduce_chi2_d(nd, fobs_re_d, fobs_im_d, fint_d, weights_d, chi2);
     // ################################
     // ########### CLEANUP ############
     // ################################
     /*float elapsed=0;
     cudaEvent_t start, stop;
     CCheck(cudaEventCreate(&start));
     CCheck(cudaEventCreate(&stop));
     CCheck(cudaEventRecord(start, 0));*/

     CCheck(cudaFree(data_d));
     CCheck(cudaFree(u_d));
     CCheck(cudaFree(v_d));
     CCheck(cudaFree(indu_d));
     CCheck(cudaFree(indv_d));
     CCheck(cudaFree(fint_d));
     CCheck(cudaFree(fobs_re_d));
     CCheck(cudaFree(fobs_im_d));
     CCheck(cudaFree(weights_d));

     /*CCheck(cudaEventRecord(stop, 0));
     CCheck(cudaEventSynchronize(stop));
     CCheck(cudaEventElapsedTime(&elapsed, start, stop) );
     CCheck(cudaEventDestroy(start));
     CCheck(cudaEventDestroy(stop));
     printf("The total time to free memory in chi2 is %.3f ms", elapsed);
     */
#else

     dcomplex* fint = (dcomplex*) malloc(sizeof(dcomplex)*nd);
     galario_sample(nx, realdata, dRA, dDec, du, nd, u, v, fint);

     galario_reduce_chi2(nd, fobs_re, fobs_im, fint, weights, chi2);

     free(fint);

#endif

}

void _galario_chi2(int nx, void* realdata, dreal dRA, dreal dDec, dreal du, int nd, void* u, void* v, void* fobs_re, void* fobs_im, void* weights, dreal* chi2) {
    galario_chi2(nx, static_cast<dreal*>(realdata), dRA, dDec, du, nd, static_cast<dreal*>(u), static_cast<dreal*>(v), static_cast<dreal*>(fobs_re), static_cast<dreal*>(fobs_im), static_cast<dreal*>(weights), chi2);
}
