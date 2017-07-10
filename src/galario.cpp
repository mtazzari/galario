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
#include <vector>

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
void galario_acc_init() {}
void galario_acc_cleanup() {}

#else
void galario_acc_init() {
#ifdef _OPENMP
    FFTWCheck(fftw_init_threads());
    fftw_plan_with_nthreads(omp_get_max_threads());
#endif
}

// TODO: define macro FFTW as fftw or fftwf
void galario_acc_cleanup() {
#ifdef _OPENMP
  FFTW(cleanup_threads)();
#endif
  FFTW(cleanup)();
}
#endif // __CUDACC__


#ifdef __CUDACC__
void fft_d(int nx, dcomplex* data_d) {
     cufftHandle plan;
     int n[NRANK] = {nx, nx};

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

void fft_h(int nx, dcomplex* data) {
    // FFTW replacement
    FFTW(complex)* fftw_data = reinterpret_cast<FFTW(complex)*>(data);
    // TODO: should ascertain that data has already been aligned

    // TODO: find a way to store the plan (maybe homogeneously with the cuFFTPlan
    FFTW(plan) p = FFTW(plan_dft_2d)(nx, nx, fftw_data, fftw_data, FFTW_FORWARD, FFTW_ESTIMATE);
    FFTW(execute)(p);

    FFTW(destroy_plan)(p);
}

#endif

void galario_fft2d(int nx, dcomplex* data) {
#ifdef __CUDACC__
    dcomplex *data_d;
    size_t nbytes = sizeof(dcomplex)*nx*nx;
    CCheck(cudaMalloc((void**)&data_d, nbytes));
    CCheck(cudaMemcpy(data_d, data, nbytes, cudaMemcpyHostToDevice));

    fft_d(nx, (dcomplex*) data_d);

    CCheck(cudaMemcpy(data, data_d, nbytes, cudaMemcpyDeviceToHost));
    CCheck(cudaFree(data_d));
#else
    fft_h(nx, data);
#endif
}

void _galario_fft2d(int nx, void* data) {
    galario_fft2d(nx, static_cast<dcomplex*>(data));
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
__host__ __device__ inline void shift_core
#else
inline void shift_core
#endif
        (int const idx_x, int const idx_y, int const nx, dcomplex* const __restrict__ a) {
#if 0
    /* column-wise access */

    auto const src_ul = idx_x + idx_y*nx;
    auto const src_ll = idx_x + idx_y*nx + nx*nx/2;
    auto const tgt_ul = src_ul + nx/2 + nx*nx/2;
    auto const tgt_ll = src_ll + nx/2 - nx*nx/2 ;

    auto const temp_ul = a[src_ul] ;
    a[src_ul] = a[tgt_ul] ;
    a[tgt_ul] = temp_ul ;

    auto const temp_ll = a[src_ll] ;
    a[src_ll] = a[tgt_ll];
    a[tgt_ll] = temp_ll;
#endif
    /* row-wise access */

    // from upper left to lower right
    auto const src_ul = idx_x*nx + idx_y;
    auto const tgt_ul = src_ul + nx*(nx+1)/2;

    // from upper right to lower left
    auto const src_ur = src_ul + nx/2;
    auto const tgt_ur = tgt_ul - nx/2;

    // swap the values
    auto const tmp_ul = a[src_ul];
    a[src_ul] = a[tgt_ul];
    a[tgt_ul] = tmp_ul;

    auto const tmp_ur = a[src_ur];
    a[src_ur] = a[tgt_ur];
    a[tgt_ur] = tmp_ur;
}

void shift_h(int const nx, dcomplex* const __restrict__ a) {
#pragma omp parallel for
    for (auto x = 0; x < nx/2; ++x) {
        for (auto y = 0; y < nx/2; ++y) {
            shift_core(x, y, nx, a);
        }
    }
}

// TODO nx -> size
// TODO make shift_d and shift_h the same function, with ifdef __CUDACC__ inside.
/**
 * grid stride loop
 */
#ifdef __CUDACC__
__global__ void shift_d(int const nx, dcomplex* const __restrict__ a) {
  // indices
  int const x0 = blockDim.x * blockIdx.x + threadIdx.x;
  int const y0 = blockDim.y * blockIdx.y + threadIdx.y;

  // stride
  int const sx = blockDim.x * gridDim.x;
  int const sy = blockDim.y * gridDim.y;

  for (auto x = x0; x < nx/2; x += sx) {
    for (auto y = y0; y < nx/2; y += sy) {
      shift_core(x, y, nx, a);
    }
  }
}
#endif

void galario_fftshift(int nx, dcomplex* data) {
#ifdef __CUDACC__
    dcomplex *data_d;
    size_t nbytes = sizeof(dcomplex)*nx*nx;
    CCheck(cudaMalloc((void**)&data_d, nbytes));
    CCheck(cudaMemcpy(data_d, data, nbytes, cudaMemcpyHostToDevice));

    shift_d<<<dim3(nx/2/galario_threads_per_block()+1, nx/2/galario_threads_per_block()+1), dim3(galario_threads_per_block(), galario_threads_per_block())>>>(nx, (dcomplex*) data_d);

    CCheck(cudaDeviceSynchronize());
    CCheck(cudaMemcpy(data, data_d, nbytes, cudaMemcpyDeviceToHost));
    CCheck(cudaFree(data_d));
#else
    shift_h(nx, data);
#endif
}

void _galario_fftshift(int nx, void* data) {
    galario_fftshift(nx, static_cast<dcomplex*>(data));
}

void galario_fftshift_fft2d_fftshift(int nx, dcomplex* data) {
#ifdef __CUDACC__
    dcomplex *data_d;
     size_t nbytes = sizeof(dcomplex)*nx*nx;
     CCheck(cudaMalloc((void**)&data_d, nbytes));
     CCheck(cudaMemcpy(data_d, data, nbytes, cudaMemcpyHostToDevice));

     shift_d<<<dim3(nx/2/galario_threads_per_block()+1, nx/2/galario_threads_per_block()+1), dim3(galario_threads_per_block(), galario_threads_per_block())>>>(nx, (dcomplex*) data_d);
     fft_d(nx, (dcomplex*) data_d);
     shift_d<<<dim3(nx/2/galario_threads_per_block()+1, nx/2/galario_threads_per_block()+1), dim3(galario_threads_per_block(), galario_threads_per_block())>>>(nx, (dcomplex*) data_d);

     CCheck(cudaMemcpy(data, data_d, nbytes, cudaMemcpyDeviceToHost));
     CCheck(cudaFree(data_d));
#else
    shift_h(nx, (dcomplex*) data);
    galario_fft2d(nx, (dcomplex*) data);
    shift_h(nx, (dcomplex*) data);
#endif
}

void _galario_fftshift_fft2d_fftshift(int nx, void* data) {
    galario_fftshift_fft2d_fftshift(nx, static_cast<dcomplex*>(data));
}

/**
 * @param nx number of pixels in x and y direction.
 * @param data Fourier transform of the image [nx * nx].
 * @param nd number of data points.
 * @param indu int(indu[i]) is the closest index into data smaller than the x value of the data point. The offset to int(indu[i]) gives the position in the pixel [nd]
 * @param indv same as indu but for the v direction [nd].
 * @param fint The image values obtained with bilinear interpolation at the data point values [nd].
 */
//    we need to re-define CUCSUB, CUCADD, CUCMUL if __CUDACC__ not defined.
//    suggestion: change CUCSUB -> CSUB ... that, CSUB=CUCSUB ifdef __CUDACC__, else CSUB: subtract between two complex numbers
#ifdef __CUDACC__
__host__ __device__ inline void interpolate_core
#else
inline void interpolate_core
#endif
        (int const idx_x, int const nx, dcomplex* const __restrict__ data, int const nd, dreal* const __restrict__ indu, dreal* const __restrict__ indv,  dcomplex* __restrict__ fint) {
    int const ii = int(indu[idx_x]);
    int const jj = int(indv[idx_x]);
    int const base = ii + jj * nx;

    dcomplex const dfu1 = CMPLXSUB(data[base + nx], data[base]);
    dcomplex const dfu2 = CMPLXSUB(data[base + nx + 1], data[base + 1]);

    // linear interpolation in u: f + df * (u - int(u))
    dcomplex const dindu {indu[idx_x] - int(indu[idx_x]), 0.0};
    dcomplex const fu1 = CMPLXADD(data[base], CMPLXMUL(dfu1, dindu));
    dcomplex const fu2 = CMPLXADD(data[base + nx], CMPLXMUL(dfu2, dindu));

    // linear interpolation in v: f + df * (v - int(v))
    dcomplex const dindv {indv[idx_x] - int(indv[idx_x]), 0.0};
    dcomplex const df = CMPLXSUB(fu2, fu1);

    fint[idx_x] = CMPLXADD(fu1, CMPLXMUL(df, dindv));

}

#ifdef __CUDACC__
__global__ void interpolate_d(int const nx, dcomplex* const __restrict__ data, int const nd, dreal* const indu, dreal* const indv, dcomplex* __restrict__ fint)
{
    //index
    int const idx_x0 = blockDim.x * blockIdx.x + threadIdx.x;

    // stride
    int const sx = blockDim.x * gridDim.x;

    for (auto idx_x = idx_x0; idx_x < nd; idx_x += sx)
    {
        interpolate_core(idx_x, nx, data, nd, indu, indv, fint);
     }
}
#endif

void interpolate_h(int const nx, dcomplex* const __restrict__ data, int const nd, dreal* const indu, dreal* const indv, dcomplex* __restrict__ fint) {
#pragma omp parallel for
    for (auto idx = 0; idx < nd; ++idx)
    {
        interpolate_core(idx, nx, data, nd, indu, indv, fint);
    }
}

void galario_interpolate(int nx, dcomplex* data, int nd, dreal* u, dreal* v, dcomplex* fint) {
#ifdef __CUDACC__
    // copy the image data
    dcomplex *data_d;
    size_t nbytes = sizeof(dcomplex)*nx*nx;
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
    interpolate_d<<<nd / galario_threads_per_block() + 1, galario_threads_per_block()>>>(nx, (dcomplex*) data_d, nd, (dreal*)u_d, (dreal*)v_d, (dcomplex*) fint_d);

    CCheck(cudaDeviceSynchronize());

    // retrieve interpolated values
    CCheck(cudaMemcpy(fint, fint_d, nbytes_fint, cudaMemcpyDeviceToHost));

    // free memories
    CCheck(cudaFree(data_d));
    CCheck(cudaFree(u_d));
    CCheck(cudaFree(v_d));
    CCheck(cudaFree(fint_d));
#else
    interpolate_h(nx, data, nd, u, v, fint);
#endif
}

void _galario_interpolate(int nx, void* data, int nd, void* u, void* v, void* fint) {
    galario_interpolate(nx, static_cast<dcomplex*>(data), nd, static_cast<dreal*>(u), static_cast<dreal*>(v), static_cast<dcomplex*>(fint));
}

// APPLY_PHASE TO SAMPLED POINTS //
#ifdef __CUDACC__
__host__ __device__ inline void apply_phase_sampled_core
#else
inline void apply_phase_sampled_core
#endif
        (int const idx_x, dreal* const u, dreal* const v, dcomplex* const __restrict__ fint, dreal const dRA, dreal const dDec) {

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
__host__ __device__ inline void apply_phase_core
#else
inline void apply_phase_core
#endif
        (int const idx_x, int const idx_y, int const nx, dcomplex* const __restrict__ data, dreal const dRA, dreal const dDec) {

    dreal const u = idx_x/(dreal)nx - 0.5;
    dreal const v = idx_y/(dreal)nx - 0.5;
    dreal const angle = u*dRA + v*dDec;
    auto const idx = idx_x + idx_y*nx;

    dcomplex const phase = dcomplex{dreal(cos(angle)), dreal(sin(angle))};
    data[idx] = CMPLXMUL(data[idx], phase);
}


#ifdef __CUDACC__
__global__ void apply_phase_d(int const nx, dcomplex* const __restrict__ data, dreal dRA, dreal dDec) {

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
        for (auto y = idx_y0; y < nx; y += sy) {
            apply_phase_core(x, y, nx, data, dRA, dDec);
        }
    }
}
#endif

void apply_phase_h(int const nx, dcomplex* const __restrict__ data, dreal dRA, dreal dDec) {

    if ((dRA==0) || (dDec==0)) {
        return;
    }

    dRA *= 2.*(dreal)M_PI;
    dDec *= 2.*(dreal)M_PI;

#pragma omp parallel for shared(dRA, dDec) schedule(static)
    for (auto x = 0; x < nx; ++x) {
        for (auto y = 0; y < nx; ++y) {
            apply_phase_core(x, y, nx, data, dRA, dDec);
        }
    }
}

void galario_apply_phase_2d(int nx, dcomplex* data, dreal dRA, dreal dDec) {
#ifdef __CUDACC__
    dcomplex *data_d;

    size_t nbytes = sizeof(dcomplex)*nx*nx;

    CCheck(cudaMalloc((void**)&data_d, nbytes));
    CCheck(cudaMemcpy(data_d, data, nbytes, cudaMemcpyHostToDevice));

    apply_phase_d<<<dim3(nx/galario_threads_per_block()+1, nx/galario_threads_per_block()+1), dim3(galario_threads_per_block(), galario_threads_per_block())>>>(nx, (dcomplex*) data_d, dRA, dDec);

    CCheck(cudaDeviceSynchronize());
    CCheck(cudaMemcpy(data, data_d, nbytes, cudaMemcpyDeviceToHost));
    CCheck(cudaFree(data_d));
#else
    apply_phase_h(nx, (dcomplex*) data, dRA, dDec);
#endif
}

void _galario_apply_phase_2d(int nx, void* data, dreal dRA, dreal dDec) {
    galario_apply_phase_2d(nx, static_cast<dcomplex*>(data), dRA, dDec);
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
        (int const i, int const nx, dreal const u0, dreal const du, dreal const* const u, dreal const* const v, dreal* const __restrict__ indu, dreal*  const __restrict__ indv) {

    // u
    int index = floor((u[i] - u0) / du);
    dreal center_ind = u0 + index * du;
    indu[i] = index + (u[i] - center_ind) / du;

    // v
    index = floor((v[i] - u0) / du);
    center_ind = u0 + index * du;
    indv[i] = index + (v[i] - center_ind) / du;

}


#ifdef __CUDACC__
__global__ void uv_idx_d(int nx, dreal const u0, dreal du, int nd, dreal const* u, dreal const* v, dreal* const __restrict__ indu, dreal* const __restrict__ indv)
    {
        // index
        int const i0 = blockDim.x * blockIdx.x + threadIdx.x;

        // stride
        int const si = blockDim.x * gridDim.x;

        for (auto i = i0; i < nd; i += si) {
            uv_idx_core(i, nx, u0, du, u, v, indu, indv);
        }
    }
#endif

void uv_idx_h(int nx, dreal const u0, dreal du, int nd, dreal const* u, dreal const* v, dreal* const __restrict__ indu, dreal* const __restrict__ indv) {
#pragma omp parallel for
    for (auto i = 0; i < nd; ++i) {
        uv_idx_core(i, nx, u0, du, u, v, indu, indv);
    }
}


void galario_get_uv_idx(int nx, dreal du, int nd, dreal* u, dreal* v, dreal* indu, dreal* indv) {
    assert(nx >= 2);

    const dreal u0 = -du*nx/2.;

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

    uv_idx_d<<<nd / galario_threads_per_block() + 1, galario_threads_per_block()>>>(nx, u0, du, nd, u_d, v_d, indu_d, indv_d);

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
    uv_idx_h(nx, u0, du, nd, (dreal*) u, (dreal*) v, (dreal*) indu, (dreal*) indv);
#endif
}

void _galario_get_uv_idx(int nx, dreal du, int nd, void* u, void* v, void* indu, void* indv) {
    galario_get_uv_idx(nx, du, nd, static_cast<dreal*>(u), static_cast<dreal*>(v), static_cast<dreal*>(indu), static_cast<dreal*>(indv));
}

#ifdef __CUDACC__
inline void sample_d(int nx, dcomplex* data_d, dreal dRA, dreal dDec, int nd, dreal u0, dreal du, dreal* u_d, dreal* v_d, dreal* indu_d, dreal* indv_d, dcomplex* fint_d)
{
     // ################################
     // ########### KERNELS ############
     // ################################
     // Kernel for shift --> FFT --> shift
     shift_d<<<dim3(nx/2/galario_threads_per_block()+1, nx/2/galario_threads_per_block()+1), dim3(galario_threads_per_block(), galario_threads_per_block())>>>(nx, (dcomplex*) data_d);
     fft_d(nx, (dcomplex*) data_d);
     shift_d<<<dim3(nx/2/galario_threads_per_block()+1, nx/2/galario_threads_per_block()+1), dim3(galario_threads_per_block(), galario_threads_per_block())>>>(nx, (dcomplex*) data_d);
     CCheck(cudaDeviceSynchronize());

     // Kernel for phase
     apply_phase_d<<<dim3(nx/galario_threads_per_block()+1, nx/galario_threads_per_block()+1), dim3(galario_threads_per_block(), galario_threads_per_block())>>>(nx, (dcomplex*) data_d, dRA, dDec);

     // Kernel for uv_idx and interpolate
     uv_idx_d<<<nd / galario_threads_per_block() + 1, galario_threads_per_block()>>>(nx, u0, du, nd, u_d, v_d, indu_d, indv_d);

     // oversubscribe blocks because we don't know if #(data points) divisible by nthreads
     interpolate_d<<<nd / galario_threads_per_block() + 1, galario_threads_per_block()>>>(nx, data_d, nd, indu_d, indv_d, fint_d);
}

__global__ void real_to_complex_d(int nx, dreal* realdata, dcomplex* data) {
    int idx_x0 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y0 = blockIdx.y * blockDim.y + threadIdx.y;

    int sx = gridDim.x * blockDim.x;
    int sy = gridDim.y * blockDim.y;

    for (int idx_x=idx_x0; idx_x < nx; idx_x+=sx) {
        for (int idx_y=idx_y0; idx_y < nx; idx_y+=sy ) {
            auto const idx = idx_y + idx_x*nx;  // row-wise
            data[idx] = CMPLX(realdata[idx], 0.0);
        }
    }
}

/**
 * Return device pointer to complex image made from real image on the host.
 *
 * Caller is responsible for freeing the device memory.
 */
dcomplex* copy_real_to_device(int nx, dreal* realdata) {
    dcomplex *data_d;
    size_t nbytes = sizeof(dcomplex)*nx*nx;

    CCheck(cudaMalloc((void**)&data_d, nbytes));

    dreal* realdata_d;
    CCheck(cudaMalloc((void**)&realdata_d, nbytes/2));
    CCheck(cudaMemcpy(realdata_d, realdata, nbytes/2, cudaMemcpyHostToDevice));

    dim3 nblocks(nx/galario_threads_per_block()+1);
    dim3 nthreads(galario_threads_per_block(), galario_threads_per_block());
    real_to_complex_d<<<nblocks, nthreads>>>(nx, realdata_d, data_d);
    CCheck(cudaFree(realdata_d));

    return data_d;
}
#endif

/**
 * return result in `fint`
 */
void galario_sample(int nx, dreal* realdata, dreal dRA, dreal dDec, dreal du, int nd, dreal* u, dreal* v, dcomplex* fint) {
    // Initialization for uv_idx and interpolate
    assert(nx >= 2);

    const dreal u0 = -du*nx/2.;
    const dreal arcsec_to_uv = (dreal)M_PI / 3600. / 180. * du * nx;
    dRA *= arcsec_to_uv;
    dDec *= arcsec_to_uv;

#ifdef __CUDACC__
    // ################################
    // ### ALLOCATION, INITIALIZATION ###
    // ################################

    dcomplex *data_d = copy_real_to_device(nx, realdata);

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
    sample_d(nx, data_d, dRA, dDec, nd, u0, du, u_d, v_d, indu_d, indv_d, fint_d);

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
    // transform image from real to complex
    std::vector<dcomplex> buffer(nx*nx);
#pragma omp parallel for shared(buffer, realdata)
    for (size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] = std::complex<dreal>(realdata[i], 0.0);
    }
    dcomplex* data = &buffer[0];

    // shift
    shift_h(nx, data);

    // cuda fft
    fft_h(nx, data);

    // shift
    shift_h(nx, data);

    // apply phase
    apply_phase_h(nx, data, dRA, dDec);

    // uv_idx_h
    dreal* indu = (dreal*) malloc(sizeof(dreal)*nd);
    dreal* indv = (dreal*) malloc(sizeof(dreal)*nd);
    uv_idx_h(nx, u0, du, nd, (dreal*) u, (dreal*) v, indu, indv);

    // interpolate
    interpolate_h(nx, data, nd, indu, indv, fint);

    free(indu);
    free(indv);
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

    // conversions
    // for the CPU case, these are inside galario_sample
    const dreal u0 = -du*nx/2.;
    const dreal arcsec_to_uv = (dreal)M_PI / 3600. / 180. * du * nx;
    dRA *= arcsec_to_uv;
    dDec *= arcsec_to_uv;

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

     sample_d(nx, data_d, dRA, dDec, nd, u0, du, u_d, v_d, indu_d, indv_d, fint_d);
     reduce_chi2_d(nd, fobs_re_d, fobs_im_d, fint_d, weights_d, chi2);
     // ################################
     // ########### CLEANUP ############
     // ################################
     CCheck(cudaFree(data_d));
     CCheck(cudaFree(u_d));
     CCheck(cudaFree(v_d));
     CCheck(cudaFree(indu_d));
     CCheck(cudaFree(indv_d));
     CCheck(cudaFree(fint_d));
     CCheck(cudaFree(fobs_re_d));
     CCheck(cudaFree(fobs_im_d));
     CCheck(cudaFree(weights_d));

#else

     dcomplex* fint = (dcomplex*) malloc(sizeof(dcomplex)*nd);
     galario_sample(nx, realdata, dRA, dDec, du, nd, u, v, fint);

     // diff weigthed and chi2
     diff_weighted_h(nd, fobs_re, fobs_im, fint, weights);

     // TODO: if available, use BLAS (mkl?) functions cblas_scnrm2 or cblas_dznrm2 for float/double complex
     // compute the Euclidean norm
     dreal y = 0.;
     for (auto i = 0; i<nd; ++i) {
         y += real(CMPLXMUL(fint[i], conj(fint[i])));
     }
     *chi2 = y;

     free(fint);

#endif

}

void _galario_chi2(int nx, void* realdata, dreal dRA, dreal dDec, dreal du, int nd, void* u, void* v, void* fobs_re, void* fobs_im, void* weights, dreal* chi2) {
    galario_chi2(nx, static_cast<dreal*>(realdata), dRA, dDec, du, nd, static_cast<dreal*>(u), static_cast<dreal*>(v), static_cast<dreal*>(fobs_re), static_cast<dreal*>(fobs_im), static_cast<dreal*>(weights), chi2);
}
