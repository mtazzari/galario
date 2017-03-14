#pragma once

#include <cuda_runtime_api.h>
#include <cuda.h>

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
