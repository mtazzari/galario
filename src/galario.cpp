#include "galario.hpp"

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
        #define CMPLXSUB cuCsub
        #define CMPLXADD cuCadd
        #define CMPLXMUL cuCmul
        #define CUBLASNRM2 cublasDznrm2

    #else
        #define CUFFTEXEC cufftExecC2C
        #define CUFFTTYPE CUFFT_C2C
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
#else
    #define SQRT sqrtf
#endif

#include <cassert>
#include <cmath>

#define NRANK 2
#define BATCH 1

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
#ifdef DOUBLE_PRECISION
#ifdef _OPENMP
    fftw_cleanup_threads();
#endif
    fftw_cleanup();
#else
#ifdef _OPENMP
    fftwf_cleanup_threads();
#endif
    fftwf_cleanup();
#endif
}

#endif


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

#ifdef DOUBLE_PRECISION
void fft_h(int nx, dcomplex* data) {
    // FFTW replacement
    fftw_complex* fftw_data = reinterpret_cast<fftw_complex*>(data);
    // TODO: should ascertain that data has already been aligned

    // TODO: find a way to store the plan (maybe homogeneously with the cuFFTPlan
    fftw_plan p = fftw_plan_dft_2d(nx, nx, fftw_data, fftw_data, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    fftw_destroy_plan(p);

}
#else
void fft_h(int nx, dcomplex* data) {
    // FFTW replacement
    fftwf_complex* fftw_data = reinterpret_cast<fftwf_complex*>(data);
    // TODO: should ascertain that data has already been aligned

    // TODO: find a way to store the plan (maybe homogeneously with the cuFFTPlan
    fftwf_plan p = fftwf_plan_dft_2d(nx, nx, fftw_data, fftw_data, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);

    fftwf_destroy_plan(p);

}
#endif

#endif

void galario_fft2d(int nx, void* data) {
#ifdef __CUDACC__
    dcomplex *data_d;
     size_t nbytes = sizeof(dcomplex)*nx*nx;
     CCheck(cudaMalloc((void**)&data_d, nbytes));
     CCheck(cudaMemcpy(data_d, data, nbytes, cudaMemcpyHostToDevice));

     fft_d(nx, (dcomplex*) data_d);

     CCheck(cudaMemcpy(data, data_d, nbytes, cudaMemcpyDeviceToHost));
     CCheck(cudaFree(data_d));
#else
    fft_h(nx, (dcomplex*) data);
#endif

}

// `a` is a matrix (size: nx^2)
#ifdef __CUDACC__
__host__ __device__ inline void shift_core
#else
inline void shift_core
#endif
        (int const idx_x, int const idx_y, int const nx, dcomplex* const __restrict__ a) {

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

void shift_h(int const nx, dcomplex* const __restrict__ a) {
#pragma omp parallel for
    for (auto x = 0; x < nx/2; ++x) {
        for (auto y = 0; y < nx/2; ++y) {
            shift_core(x, y, nx, a);
        }
    }
}

void galario_fftshift(int nx, void* data) {
#ifdef __CUDACC__
    dcomplex *data_d;
     size_t nbytes = sizeof(dcomplex)*nx*nx;
     CCheck(cudaMalloc((void**)&data_d, nbytes));
     CCheck(cudaMemcpy(data_d, data, nbytes, cudaMemcpyHostToDevice));

     shift_d<<<dim3(nx/2/32+1, nx/2/32+1), dim3(32, 32)>>>(nx, (dcomplex*) data_d);

     CCheck(cudaDeviceSynchronize());
     CCheck(cudaMemcpy(data, data_d, nbytes, cudaMemcpyDeviceToHost));
     CCheck(cudaFree(data_d));
#else
    shift_h(nx, (dcomplex*) data);
#endif
}

void galario_fftshift_fft2d_fftshift(int nx, void* data) {
#ifdef __CUDACC__
    dcomplex *data_d;
     size_t nbytes = sizeof(dcomplex)*nx*nx;
     CCheck(cudaMalloc((void**)&data_d, nbytes));
     CCheck(cudaMemcpy(data_d, data, nbytes, cudaMemcpyHostToDevice));

     shift_d<<<dim3(nx/2/32+1, nx/2/32+1), dim3(32, 32)>>>(nx, (dcomplex*) data_d);
     fft_d(nx, (dcomplex*) data_d);
     shift_d<<<dim3(nx/2/32+1, nx/2/32+1), dim3(32, 32)>>>(nx, (dcomplex*) data_d);

     CCheck(cudaMemcpy(data, data_d, nbytes, cudaMemcpyDeviceToHost));
     CCheck(cudaFree(data_d));
#else
    shift_h(nx, (dcomplex*) data);
    galario_fft2d(nx, (dcomplex*) data);
    shift_h(nx, (dcomplex*) data);
#endif
}

/**
 * @param nx number of pixels in x and y direction.
 * @param data Fourier transform of the image [nx * nx].
 * @param nd number of data points.
 * @param indu int(indu[i]) is the closest index into data smaller than the x value of the data point. The offset to int(indu[i]) gives the position in the pixel [nd]
 * @param indv same as indu but for the v direction [nd].
 * @param fint The image values obtained with bilinear interpolation at the data point values [nd].
 */
//TODO convert galario_acc_interpolate with stride loops.
// 1) galario_acc_interpolate -> interpolate_d
// 2) define interpolate_core that performs the actions inside the for loop.
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
    int const base = ii * nx + jj;

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
    //printf("", indu_d[idx_x], indv_d[idx_x]);

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

void galario_interpolate(int nx, void* data, int nd, void* u, void* v, void* fint)
{
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
    interpolate_h(nx, (dcomplex*) data, nd, (dreal*)u, (dreal*)v, (dcomplex*) fint);

#endif
}

#ifdef __CUDACC__
__host__ __device__ inline void apply_phase_core
#else
inline void apply_phase_core
#endif
        (int const idx_x, int const idx_y, int const nx, dcomplex* const __restrict__ data, dreal const x0, dreal const y0) {

    dreal const u = idx_x/(dreal)nx - 0.5;
    dreal const v = idx_y/(dreal)nx - 0.5;

    dreal const angle = u*y0 + v*x0;

    dcomplex const phase = dcomplex{dreal(cos(angle)), -dreal(sin(angle))};

    auto const idx = idx_x + idx_y*nx;

    data[idx] = CMPLXMUL(data[idx], phase);
}


#ifdef __CUDACC__
__global__ void apply_phase_d(int const nx, dcomplex* const __restrict__ data, dreal const x0, dreal const y0) {

    // indices
    int const idx_x0 = blockDim.x * blockIdx.x + threadIdx.x;
    int const idx_y0 = blockDim.y * blockIdx.y + threadIdx.y;

    // stride
    int const sx = blockDim.x * gridDim.x;
    int const sy = blockDim.y * gridDim.y;

    for (auto x = idx_x0; x < nx; x += sx) {
        for (auto y = idx_y0; y < nx; y += sy) {
            apply_phase_core(x, y, nx, data, x0, y0);
        }
    }
}
#endif

void apply_phase_h(int const nx, dcomplex* const __restrict__ data, dreal const x0, dreal const y0) {
#pragma omp parallel for
    for (auto x = 0; x < nx; ++x) {
        for (auto y = 0; y < nx; ++y) {
            apply_phase_core(x, y, nx, data, x0, y0);
        }
    }
}


void galario_apply_phase_2d(int nx, void* data, dreal x0, dreal y0) {
#ifdef __CUDACC__
    dcomplex *data_d;

     size_t nbytes = sizeof(dcomplex)*nx*nx;

     CCheck(cudaMalloc((void**)&data_d, nbytes));
     CCheck(cudaMemcpy(data_d, data, nbytes, cudaMemcpyHostToDevice));

     apply_phase_d<<<dim3(nx/32+1, nx/32+1), dim3(32, 32)>>>(nx, (dcomplex*) data_d, x0, y0);

     CCheck(cudaDeviceSynchronize());
     CCheck(cudaMemcpy(data, data_d, nbytes, cudaMemcpyDeviceToHost));
     CCheck(cudaFree(data_d));
#else
    apply_phase_h(nx, (dcomplex*) data, x0, y0);
#endif
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
 * @param pixel_centers positions at which DFT is available
 * @param nd number of data points
 * @param u part of data points in u direction
 * @param v part of data points in v direction
 * @param indu Index in u direction [output]
 * @param indv Index in v direction [output]
 *
 * NOTE: this can be highly simplified by noting that:
 *       indu[i] = index + (u[i] - umin - index*du)/du =
 *               = index + (u[i] - umin)/du - index = (u[i] - umin)/du
 *       And the same applies for indv.
 *       I think we can delete this kernel and put it directly in interpolate_core.
 */
#ifdef __CUDACC__
__host__ __device__ inline void rotix_core
#else
inline void rotix_core
#endif
        (int const i, int const nx, dreal const umin, dreal const du, dreal const* const u, dreal const* const v, dreal* const __restrict__ indu, dreal*  const __restrict__ indv) {
    // u
    int index = floor((u[i] - umin) / du);
    dreal center_ind = umin + index * du;
    indu[i] = index + (u[i] - center_ind) / du;

    // v
    index = floor((v[i] - umin) / du);
    center_ind = umin + index * du;
    indv[i] = index + (v[i] - center_ind) / du;
}


#ifdef __CUDACC__
__global__ void rotix_d(int nx, dreal umin, dreal du, int nd, dreal const* u, dreal const* v, dreal* const __restrict__ indu, dreal* const __restrict__ indv)
    {
        // index
        int const i0 = blockDim.x * blockIdx.x + threadIdx.x;

        // stride
        int const si = blockDim.x * gridDim.x;

        for (auto i = i0; i < nd; i += si) {
            rotix_core(i, nx, umin, du, u, v, indu, indv);
        }
    }
#endif

void rotix_h(int nx, dreal umin, dreal du, int nd, dreal const* u, dreal const* v, dreal* const __restrict__ indu, dreal* const __restrict__ indv) {
#pragma omp parallel for
    for (auto i = 0; i < nd; ++i) {
        rotix_core(i, nx, umin, du, u, v, indu, indv);
    }
}


void galario_acc_rotix(int nx, void* vpixel_centers, int nd, void* u, void* v, void* indu, void* indv)
{
    assert(nx >= 2);

    // uniform distance between pixel centers
    dreal* pixel_centers = (dreal*) vpixel_centers;
    const dreal umin = pixel_centers[0];
    const dreal du = pixel_centers[1] - umin;

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

    rotix_d<<<nd / galario_threads_per_block() + 1, galario_threads_per_block()>>>(nx, umin, du, nd, u_d, v_d, indu_d, indv_d);

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
    rotix_h(nx, umin, du, nd, (dreal*) u, (dreal*) v, (dreal*) indu, (dreal*) indv);
#endif
}

#ifdef __CUDACC__
inline void sample_d(int nx, dcomplex* data_d, dreal x0, dreal y0, int nd, dreal umin, dreal du, dreal* u_d, dreal* v_d, dreal* indu_d, dreal* indv_d, dcomplex* fint_d)
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
     apply_phase_d<<<dim3(nx/galario_threads_per_block()+1, nx/galario_threads_per_block()+1), dim3(galario_threads_per_block(), galario_threads_per_block())>>>(nx, (dcomplex*) data_d, x0, y0);

     // Kernel for rotix and interpolate
     rotix_d<<<nd / galario_threads_per_block() + 1, galario_threads_per_block()>>>(nx, umin, du, nd, u_d, v_d, indu_d, indv_d);
     // oversubscribe blocks because we don't know if #(data points) divisible by nthreads
     interpolate_d<<<nd / galario_threads_per_block() + 1, galario_threads_per_block()>>>(nx, data_d, nd, indu_d, indv_d, fint_d);
}
#endif

/**
 * return result in `fint`
 */
void galario_sample(int nx, void* data, dreal x0, dreal y0, void* vpixel_centers, int nd, void* u, void* v, void* fint)
{
    // Initialization for rotix and interpolate
    assert(nx >= 2);
    dreal* pixel_centers = (dreal*) vpixel_centers;
    const dreal umin = pixel_centers[0];
    const dreal du = pixel_centers[1] - umin;

#ifdef __CUDACC__
    // ################################
     // ### ALLOCATION, INITIALIZATION ###
     // ################################

     // Initialization for FFT, shift (and apply phase)
     dcomplex *data_d;
     size_t nbytes = sizeof(dcomplex)*nx*nx;

     CCheck(cudaMalloc((void**)&data_d, nbytes));
     CCheck(cudaMemcpy(data_d, data, nbytes, cudaMemcpyHostToDevice));

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

     // do the work on the cpu
     sample_d(nx, data_d, x0, y0, nd, umin, du, u_d, v_d, indu_d, indv_d, fint_d);

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
    // shift
    shift_h(nx, (dcomplex*) data);

    // cuda fft
    fft_h(nx, (dcomplex*) data);

    // shift
    shift_h(nx, (dcomplex*) data);

    // apply phase
    apply_phase_h(nx, (dcomplex*) data, x0, y0);

    // rotix_h
    dreal* indu = (dreal*) malloc(sizeof(dreal)*nd);
    dreal* indv = (dreal*) malloc(sizeof(dreal)*nd);
    rotix_h(nx, umin, du, nd, (dreal*) u, (dreal*) v, indu, indv);

    // interpolate
    interpolate_h(nx, (dcomplex*) data, nd, indu, indv, (dcomplex*) fint);

    free(indu);
    free(indv);
#endif
}
/**
 * Compute weighted difference between observations (`fobs_re` and `fobs_im`) and model predictions `fint`, write to `fint`
 */
#ifdef __CUDACC__
__host__ __device__ inline void diff_weighted_core
#else
inline void diff_weighted_core
#endif
        (int const idx_x, int const nd, dreal const* const __restrict__ fobs_re, dreal const* const __restrict__ fobs_im, dcomplex* const __restrict__ fint, dreal const* const __restrict__ weights)
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

void galario_reduce_chi2
        (int nd, void* fobs_re, void* fobs_im, void* fint, void* weights, dreal* chi2)
{
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

    dcomplex* fint_cmplx = (dcomplex*) fint;

    diff_weighted_h(nd, (dreal*) fobs_re, (dreal*) fobs_im, fint_cmplx, (dreal*) weights);

    // TODO: if available, use BLAS (mkl?) functions cblas_scnrm2 or cblas_dznrm2 for float/double complex
    // compute the Euclidean norm
    dreal y = 0.;
    for (auto i = 0; i<nd; ++i) {
        dcomplex x = fint_cmplx[i];
        y += real(CMPLXMUL(x, conj(x)));
    }
    *chi2 = y;

#endif
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

void galario_chi2(int nx, void* data, dreal x0, dreal y0, void* vpixel_centers, int nd, void* u, void* v, void* fobs_re, void* fobs_im, void* weights, dreal* chi2) {

    // dcomplex* data_cmplx = (dcomplex*) data;  // casting all the times or only once?
    // Initilization for rotix and interpolate
    assert(nx >= 2);

#ifdef __CUDACC__

    dreal* pixel_centers = (dreal*) vpixel_centers;
    const dreal umin = pixel_centers[0];
    const dreal du = pixel_centers[1] - umin;

    // ################################
     // ### ALLOCATION, INITIALIZATION ###
     // ################################

     // Initialization for FFT, shift (and apply phase)
     dcomplex *data_d;
     size_t nbytes = sizeof(dcomplex)*nx*nx;

     CCheck(cudaMalloc((void**)&data_d, nbytes));
     CCheck(cudaMemcpy(data_d, data, nbytes, cudaMemcpyHostToDevice));

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


    //  // ################################
    //  // ########### KERNELS ############
    //  // ################################
    //  // Kernel for shift --> FFT --> shift
    //  shift_d<<<dim3(nx/2/galario_threads_per_block()+1, nx/2/galario_threads_per_block()+1), dim3(galario_threads_per_block(), galario_threads_per_block())>>>(nx, (dcomplex*) data_d);
    //  fft_d(nx, (dcomplex*) data_d);
    //  shift_d<<<dim3(nx/2/galario_threads_per_block()+1, nx/2/galario_threads_per_block()+1), dim3(galario_threads_per_block(), galario_threads_per_block())>>>(nx, (dcomplex*) data_d);
    //  CCheck(cudaDeviceSynchronize());

    //  // Kernel for phase
    //  apply_phase_d<<<dim3(nx/galario_threads_per_block()+1, nx/galario_threads_per_block()+1), dim3(galario_threads_per_block(), galario_threads_per_block())>>>(nx, (dcomplex*) data_d, x0, y0);

    //  // Kernel for rotix and interpolate
    //  rotix_d<<<nd / galario_threads_per_block() + 1, galario_threads_per_block()>>>(nx, umin, du, nd, u_d, v_d, indu_d, indv_d);
    //  // oversubscribe blocks because we don't know if #(data points) divisible by nthreads
    //  interpolate_d<<<nd / galario_threads_per_block() + 1, galario_threads_per_block()>>>(nx, data_d, nd, indu_d, indv_d, fint_d);
    //  CCheck(cudaDeviceSynchronize());

     // // Kernel for comparison and chi squared
     // cublasHandle_t handle;
     // CBlasCheck(cublasCreate(&handle));
     // diff_weighted_d<<<nd / galario_threads_per_block() + 1, galario_threads_per_block()>>>(nd, fobs_re_d, fobs_im_d, fint_d, weights_d);

     // // only device pointers! maybe not ... check with jiri
     // // compute the Euclidean norm
     // CUBLASNRM2(handle, nd, fint_d, 1, chi2);
     // // but we want the square of the norm
     // *chi2 *= *chi2;

     sample_d(nx, data_d, x0, y0, nd, umin, du, u_d, v_d, indu_d, indv_d, fint_d);
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
     // CBlasCheck(cublasDestroy(handle));

#else

    // // shift
    // shift_h(nx, (dcomplex*) data);

    // // cuda fft
    // fft_h(nx, (dcomplex*) data);

    // // shift
    // shift_h(nx, (dcomplex*) data);

    // // apply phase
    // apply_phase_h(nx, (dcomplex*) data, x0, y0);

    // // rotix_h
    // dreal* indu = (dreal*) malloc(sizeof(dreal)*nd);
    // dreal* indv = (dreal*) malloc(sizeof(dreal)*nd);
    // rotix_h(nx, umin, du, nd, (dreal*) u, (dreal*) v, indu, indv);

    // // interpolate
    //  dcomplex* fint = (dcomplex*) malloc(sizeof(dcomplex)*nd);
    //  interpolate_h(nx, (dcomplex*) data, nd, indu, indv, fint);

     dcomplex* fint = (dcomplex*) malloc(sizeof(dcomplex)*nd);
     galario_sample(nx, data, x0, y0, vpixel_centers, nd, u, v, fint);

    // diff weigthed and chi2
    diff_weighted_h(nd, (dreal*) fobs_re, (dreal*) fobs_im, fint, (dreal*) weights);

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
