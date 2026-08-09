/*
 * gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.
 *
 * gpuxtb self-declared NVIDIA cuBLAS / cuSOLVER / CUDA-driver C ABI surface.
 *
 * This header is ORIGINAL gpuxtb code. It re-declares only the host-side C
 * entry points that gpuxtb actually calls at runtime (the same curated set
 * that the runtime cohort preflight in cuda_dlopen.c and the implib symbol
 * lists require), plus the opaque handle types, enums, and constants those
 * calls need. It intentionally does not copy any NVIDIA header text,
 * layout, macros, or documentation: only function signatures, ABI-stable
 * integer values, and opaque type tags, which are interface facts.
 *
 * Why this exists: the cuBLAS/cuSOLVER/CUDA-driver shared libraries are
 * dlopen'd lazily at runtime via generated trampolines (see cuda_dlopen.c)
 * and are deliberately never linked. By declaring the interface here, the
 * CUDA backend can be built with only the CUDA compiler and the cudart
 * header set (cuda_runtime_api.h, library_types.h), without the proprietary
 * NVIDIA cublas_v2.h / cusolverDn.h / cuda.h headers or their libraries
 * present in the build environment.
 *
 * Compatibility contract (why this is safe as a fixed declaration set):
 *   - All symbols declared here are the versioned, ABI-frozen dlsym names
 *     (cublasCreate_v2, cudaStreamGetCaptureInfo_v2, cuMemGetAddressRange_v2,
 *     ...). NVIDIA adds a versioned suffix precisely when it must change a
 *     prototype, so these names are a stable binary contract across minor
 *     CUDA releases within a major version.
 *   - The handle types (cublasHandle_t, cusolverDnHandle_t,
 *     cusolverDnParams_t, syevjInfo_t) are opaque struct pointers and the
 *     status/enum values are documented stable integers, so the declarations
 *     cannot drift from the deployed library ABI.
 *   - Runtime enforcement: cuda_dlopen.c resolves every name in this header
 *     against the deployed SONAME cohort at first use and disables the GPU
 *     backend with a typed-fallback diagnostic if any symbol is absent. The
 *     capture-mode gate in gfn2_scc_setup_eigensolver.cu additionally
 *     verifies cublasGetVersion / cusolverGetProperty at runtime before
 *     enabling version-sensitive routes.
 *   - Build contract versions: CUBLAS_VERSION / CUSOLVER_VERSION below are
 *     pinned at the minimum versions the compiled routes require (CUDA 12.9
 *     era). Compiling against a newer deployed runtime is fine; the runtime
 *     version checks above decide which route is actually exercised.
 */

#ifndef GPU_XTB_NVIDIA_HOST_API_H
#define GPU_XTB_NVIDIA_HOST_API_H

#include <cuda_runtime_api.h>
#include <library_types.h>
#include <stddef.h>
#include <stdint.h>

#define CUBLAS_VERSION 120901  /* cuBLAS 12.9.1: minimum runtime the build targets */
#define CUSOLVER_VERSION 11705 /* cuSOLVER 11.7.5: minimum runtime the build targets */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* cuBLAS                                                              */
/* ------------------------------------------------------------------ */

typedef struct cublasContext* cublasHandle_t;

typedef enum {
  CUBLAS_STATUS_SUCCESS = 0,
  CUBLAS_STATUS_NOT_INITIALIZED = 1,
  CUBLAS_STATUS_ALLOC_FAILED = 3,
  CUBLAS_STATUS_INVALID_VALUE = 7,
  CUBLAS_STATUS_ARCH_MISMATCH = 8,
  CUBLAS_STATUS_INTERNAL_ERROR = 11,
  CUBLAS_STATUS_NOT_SUPPORTED = 12,
  CUBLAS_STATUS_EXECUTION_FAILED = 13
} cublasStatus_t;

typedef enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 } cublasOperation_t;

typedef enum {
  CUBLAS_FILL_MODE_LOWER = 0,
  CUBLAS_FILL_MODE_UPPER = 1,
  CUBLAS_FILL_MODE_FULL = 2
} cublasFillMode_t;

typedef enum { CUBLAS_DIAG_NON_UNIT = 0, CUBLAS_DIAG_UNIT = 1 } cublasDiagType_t;

typedef enum { CUBLAS_SIDE_LEFT = 0, CUBLAS_SIDE_RIGHT = 1 } cublasSideMode_t;

typedef enum { CUBLAS_POINTER_MODE_HOST = 0, CUBLAS_POINTER_MODE_DEVICE = 1 } cublasPointerMode_t;

typedef enum { CUBLAS_DEFAULT_MATH = 0, CUBLAS_PEDANTIC_MATH = 2 } cublasMath_t;

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int* version);
cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void* workspace,
                                     size_t workspaceSizeInBytes);
cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId);
cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t* streamId);
cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t* mode);
cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t mode);
cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode);
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);
cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side,
                                  cublasFillMode_t uplo, cublasOperation_t trans,
                                  cublasDiagType_t diag, int m, int n, const double* alpha,
                                  const double* const A[], int lda, double* const B[], int ldb,
                                  int batchCount);

/* The NVIDIA headers normalize these source names to the versioned ELF alias
 * that the runtime cohort preflight and implib symbol lists require. */
#define cublasCreate cublasCreate_v2
#define cublasDestroy cublasDestroy_v2
#define cublasGetVersion cublasGetVersion_v2
#define cublasSetWorkspace cublasSetWorkspace_v2
#define cublasSetStream cublasSetStream_v2
#define cublasGetStream cublasGetStream_v2
#define cublasGetPointerMode cublasGetPointerMode_v2
#define cublasSetPointerMode cublasSetPointerMode_v2

/* ------------------------------------------------------------------ */
/* cuSOLVER                                                            */
/* ------------------------------------------------------------------ */

typedef struct cusolverDnContext* cusolverDnHandle_t;
typedef struct syevjInfo* syevjInfo_t;
typedef struct cusolverDnParams* cusolverDnParams_t;

typedef enum {
  CUSOLVER_STATUS_SUCCESS = 0,
  CUSOLVER_STATUS_NOT_INITIALIZED = 1,
  CUSOLVER_STATUS_ALLOC_FAILED = 2,
  CUSOLVER_STATUS_INVALID_VALUE = 3,
  CUSOLVER_STATUS_ARCH_MISMATCH = 4,
  CUSOLVER_STATUS_EXECUTION_FAILED = 6,
  CUSOLVER_STATUS_INTERNAL_ERROR = 7
} cusolverStatus_t;

typedef enum { CUSOLVER_EIG_MODE_NOVECTOR = 0, CUSOLVER_EIG_MODE_VECTOR = 1 } cusolverEigMode_t;

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t* handle);
cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle);
cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t streamId);
cusolverStatus_t cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t* streamId);
cusolverStatus_t cusolverDnCreateParams(cusolverDnParams_t* params);
cusolverStatus_t cusolverDnDestroyParams(cusolverDnParams_t params);
cusolverStatus_t cusolverDnCreateSyevjInfo(syevjInfo_t* info);
cusolverStatus_t cusolverDnDestroySyevjInfo(syevjInfo_t info);
cusolverStatus_t cusolverDnXsyevjSetTolerance(syevjInfo_t info, double tolerance);
cusolverStatus_t cusolverDnXsyevjSetMaxSweeps(syevjInfo_t info, int max_sweeps);
cusolverStatus_t cusolverDnXsyevjSetSortEig(syevjInfo_t info, int sort_eig);
cusolverStatus_t cusolverDnDpotrfBatched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                         double* Aarray[], int lda, int* infoArray, int batchSize);
cusolverStatus_t cusolverDnDsytrd_bufferSize(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                             int n, const double* A, int lda, const double* d,
                                             const double* e, const double* tau, int* lwork);
cusolverStatus_t cusolverDnDsytrd(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                  double* A, int lda, double* d, double* e, double* tau,
                                  double* work, int lwork, int* info);
cusolverStatus_t cusolverDnDormtr_bufferSize(cusolverDnHandle_t handle, cublasSideMode_t side,
                                             cublasFillMode_t uplo, cublasOperation_t trans, int m,
                                             int n, const double* A, int lda, const double* tau,
                                             const double* C, int ldc, int* lwork);
cusolverStatus_t cusolverDnDormtr(cusolverDnHandle_t handle, cublasSideMode_t side,
                                  cublasFillMode_t uplo, cublasOperation_t trans, int m, int n,
                                  double* A, int lda, double* tau, double* C, int ldc, double* work,
                                  int lwork, int* info);
cusolverStatus_t cusolverDnDsyevjBatched_bufferSize(cusolverDnHandle_t handle,
                                                    cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                                    int n, const double* A, int lda,
                                                    const double* W, int* lwork, syevjInfo_t params,
                                                    int batchSize);
cusolverStatus_t cusolverDnDsyevjBatched(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                         cublasFillMode_t uplo, int n, double* A, int lda,
                                         double* W, double* work, int lwork, int* info,
                                         syevjInfo_t params, int batchSize);
cusolverStatus_t cusolverDnXsyevBatched_bufferSize(
    cusolverDnHandle_t handle, cusolverDnParams_t params, cusolverEigMode_t jobz,
    cublasFillMode_t uplo, int64_t n, cudaDataType dataTypeA, const void* A, int64_t lda,
    cudaDataType dataTypeW, const void* W, cudaDataType computeType,
    size_t* workspaceInBytesOnDevice, size_t* workspaceInBytesOnHost, int64_t batchSize);
cusolverStatus_t cusolverDnXsyevBatched(cusolverDnHandle_t handle, cusolverDnParams_t params,
                                        cusolverEigMode_t jobz, cublasFillMode_t uplo, int64_t n,
                                        cudaDataType dataTypeA, void* A, int64_t lda,
                                        cudaDataType dataTypeW, void* W, cudaDataType computeType,
                                        void* bufferOnDevice, size_t workspaceInBytesOnDevice,
                                        void* bufferOnHost, size_t workspaceInBytesOnHost,
                                        int* info, int64_t batchSize);
cusolverStatus_t cusolverGetProperty(libraryPropertyType type, int* value);

/* ------------------------------------------------------------------ */
/* CUDA driver API                                                     */
/* ------------------------------------------------------------------ */

typedef enum {
  CUDA_SUCCESS = 0,
  CUDA_ERROR_INVALID_VALUE = 1,
  CUDA_ERROR_NOT_INITIALIZED = 3,
  CUDA_ERROR_DEINITIALIZED = 4,
  CUDA_ERROR_INVALID_CONTEXT = 201,
  CUDA_ERROR_NOT_FOUND = 500
} CUresult;

#if defined(__LP64__)
typedef unsigned long long CUdeviceptr;
#else
#error "the gpuxtb CUDA dlopen backend requires a 64-bit ELF target"
#endif

CUresult cuGetErrorString(CUresult error, const char** pStr);
CUresult cuMemGetAddressRange_v2(CUdeviceptr* pbase, size_t* psize, CUdeviceptr dptr);
#define cuMemGetAddressRange cuMemGetAddressRange_v2

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GPU_XTB_NVIDIA_HOST_API_H */
