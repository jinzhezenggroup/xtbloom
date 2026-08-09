#ifndef _GNU_SOURCE
// gpuxtb's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define _GNU_SOURCE
#endif

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_dlopen_symbols.h"

#if defined(_WIN32) || defined(__APPLE__)
#error "the gpuxtb CUDA dlopen bootstrap requires ELF constructor ordering"
#endif

#ifndef RTLD_NODELETE
#error "the gpuxtb CUDA dlopen bootstrap requires RTLD_NODELETE"
#endif

#ifndef GPUXTB_CUDART_SONAME
#error "GPUXTB_CUDART_SONAME must name the exact build CUDA runtime SONAME"
#endif
#ifndef GPUXTB_CUBLAS_SONAME
#error "GPUXTB_CUBLAS_SONAME must name the exact build cuBLAS SONAME"
#endif
#ifndef GPUXTB_CUSOLVER_SONAME
#error "GPUXTB_CUSOLVER_SONAME must name the exact build cuSOLVER SONAME"
#endif
#ifndef GPUXTB_CUDA_DRIVER_SONAME
#error "GPUXTB_CUDA_DRIVER_SONAME must name the CUDA driver SONAME"
#endif

#if defined(__GNUC__) || defined(__clang__)
#define GPU_XTB_HIDDEN __attribute__((visibility("hidden")))
#else
#define GPU_XTB_HIDDEN
#endif

/* implib derives these names from the deliberately stable shim input basenames
 * libcudart.so, libcublas.so, libcusolver.so, and libcuda.so. The generator is
 * invoked with --no-dlopen: this translation unit owns every handle and fills
 * every table before ordinary NVCC registration constructors can run. */
GPU_XTB_HIDDEN void _libcudart_so_tramp_set_handle(void* handle);
GPU_XTB_HIDDEN void _libcudart_so_tramp_resolve_all(void);
GPU_XTB_HIDDEN void _libcublas_so_tramp_set_handle(void* handle);
GPU_XTB_HIDDEN void _libcublas_so_tramp_resolve_all(void);
GPU_XTB_HIDDEN void _libcusolver_so_tramp_set_handle(void* handle);
GPU_XTB_HIDDEN void _libcusolver_so_tramp_resolve_all(void);
GPU_XTB_HIDDEN void _libcuda_so_tramp_set_handle(void* handle);
GPU_XTB_HIDDEN void _libcuda_so_tramp_resolve_all(void);

enum gpu_xtb_diagnostic_kind {
  GPU_XTB_DIAGNOSTIC_LOAD = 1u << 0,
  GPU_XTB_DIAGNOSTIC_UNKNOWN_SYMBOL = 1u << 1,
};

static pthread_mutex_t gpu_xtb_diagnostic_mutex = PTHREAD_MUTEX_INITIALIZER;
static unsigned int gpu_xtb_reported_diagnostics;

static void gpu_xtb_report_once(enum gpu_xtb_diagnostic_kind kind, const char* library,
                                const char* symbol, const char* detail) {
  pthread_mutex_lock(&gpu_xtb_diagnostic_mutex);
  if ((gpu_xtb_reported_diagnostics & (unsigned int)kind) == 0u) {
    gpu_xtb_reported_diagnostics |= (unsigned int)kind;
    if (symbol != NULL) {
      fprintf(stderr,
              "gpuxtb: CUDA trampoline has no ABI-correct fallback for '%s' "
              "from %s; the GPU backend is disabled\n",
              symbol, library != NULL ? library : "an unknown cohort");
    } else {
      fprintf(stderr,
              "gpuxtb: CUDA cohort %s is unavailable%s%s; the GPU backend "
              "is disabled\n",
              library != NULL ? library : "<unknown>", detail != NULL ? ": " : "",
              detail != NULL ? detail : "");
    }
  }
  pthread_mutex_unlock(&gpu_xtb_diagnostic_mutex);
}

/* A failed cohort uses one impossible dlopen handle. gpu_xtb_cuda_dlsym then
 * installs typed fallbacks for every curated symbol rather than mixing real
 * entry points with substitutes from an ABI-incompatible library version. */
static unsigned char gpu_xtb_missing_cohort_sentinel;
#define GPU_XTB_MISSING_COHORT ((void*)&gpu_xtb_missing_cohort_sentinel)

static const char gpu_xtb_cuda_unavailable_text[] =
    "CUDA symbol unavailable because the gpuxtb NVIDIA runtime cohort did not load";

/* CUDA runtime fallbacks. Calls return cudaErrorSharedObjectSymbolNotFound
 * (302), while output-only arguments are put into deterministic empty states.
 * Device output arrays are deliberately not dereferenced. */
static cudaError_t gpu_xtb_cudart_error(void) { return cudaErrorSharedObjectSymbolNotFound; }

static const char* gpu_xtb_cuda_get_error_text(cudaError_t error) {
  (void)error;
  return gpu_xtb_cuda_unavailable_text;
}

static cudaError_t gpu_xtb_cuda_event_create(cudaEvent_t* event, unsigned int flags) {
  (void)flags;
  if (event != NULL) {
    *event = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_event_create_default(cudaEvent_t* event) {
  if (event != NULL) {
    *event = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_event_destroy(cudaEvent_t event) {
  (void)event;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_event_record(cudaEvent_t event, cudaStream_t stream) {
  (void)event;
  (void)stream;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_event_synchronize(cudaEvent_t event) {
  (void)event;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_event_elapsed_time(float* milliseconds, cudaEvent_t start,
                                                   cudaEvent_t end) {
  (void)start;
  (void)end;
  if (milliseconds != NULL) {
    *milliseconds = 0.0F;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_free(void* pointer) {
  (void)pointer;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_get_device(int* device) {
  if (device != NULL) {
    *device = 0;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_get_device_count(int* count) {
  if (count != NULL) {
    *count = 0;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_device_get_attribute(int* value, enum cudaDeviceAttr attribute,
                                                     int device) {
  (void)attribute;
  (void)device;
  if (value != NULL) {
    *value = 0;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_graph_add_node(cudaGraphNode_t* node, cudaGraph_t graph,
                                               const cudaGraphNode_t* dependencies,
                                               size_t dependency_count,
                                               struct cudaGraphNodeParams* parameters) {
  (void)graph;
  (void)dependencies;
  (void)dependency_count;
  (void)parameters;
  if (node != NULL) {
    *node = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_graph_child_get_graph(cudaGraphNode_t node, cudaGraph_t* graph) {
  (void)node;
  if (graph != NULL) {
    *graph = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_graph_conditional_handle_create(cudaGraphConditionalHandle* handle,
                                                                cudaGraph_t graph,
                                                                unsigned int default_value,
                                                                unsigned int flags) {
  (void)graph;
  (void)default_value;
  (void)flags;
  if (handle != NULL) {
    *handle = 0;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_graph_create(cudaGraph_t* graph, unsigned int flags) {
  (void)flags;
  if (graph != NULL) {
    *graph = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_graph_destroy(cudaGraph_t graph) {
  (void)graph;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_graph_exec_destroy(cudaGraphExec_t graph_exec) {
  (void)graph_exec;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_graph_get_nodes(cudaGraph_t graph, cudaGraphNode_t* nodes,
                                                size_t* node_count) {
  (void)graph;
  (void)nodes;
  if (node_count != NULL) {
    *node_count = 0;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_graph_memcpy_node_get_params(cudaGraphNode_t node,
                                                             struct cudaMemcpy3DParms* parameters) {
  (void)node;
  if (parameters != NULL) {
    memset(parameters, 0, sizeof(*parameters));
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_graph_instantiate(cudaGraphExec_t* graph_exec, cudaGraph_t graph,
                                                  unsigned long long flags) {
  (void)graph;
  (void)flags;
  if (graph_exec != NULL) {
    *graph_exec = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_graph_launch(cudaGraphExec_t graph_exec, cudaStream_t stream) {
  (void)graph_exec;
  (void)stream;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_graph_node_get_type(cudaGraphNode_t node,
                                                    enum cudaGraphNodeType* type) {
  (void)node;
  if (type != NULL) {
    *type = cudaGraphNodeTypeEmpty;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_graph_upload(cudaGraphExec_t graph_exec, cudaStream_t stream) {
  (void)graph_exec;
  (void)stream;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_host_get_flags(unsigned int* flags, void* host_pointer) {
  (void)host_pointer;
  if (flags != NULL) {
    *flags = 0;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_host_register(void* pointer, size_t size, unsigned int flags) {
  (void)pointer;
  (void)size;
  (void)flags;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_host_unregister(void* pointer) {
  (void)pointer;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_launch_host_func(cudaStream_t stream, cudaHostFn_t function,
                                                 void* user_data) {
  (void)stream;
  (void)function;
  (void)user_data;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_launch_kernel(const void* function, dim3 grid_dimension,
                                              dim3 block_dimension, void** arguments,
                                              size_t shared_memory, cudaStream_t stream) {
  (void)function;
  (void)grid_dimension;
  (void)block_dimension;
  (void)arguments;
  (void)shared_memory;
  (void)stream;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_malloc(void** pointer, size_t size) {
  (void)size;
  if (pointer != NULL) {
    *pointer = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_malloc_managed(void** pointer, size_t size, unsigned int flags) {
  (void)size;
  (void)flags;
  if (pointer != NULL) {
    *pointer = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_memcpy(void* destination, const void* source, size_t count,
                                       enum cudaMemcpyKind kind) {
  (void)destination;
  (void)source;
  (void)count;
  (void)kind;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_memcpy_async(void* destination, const void* source, size_t count,
                                             enum cudaMemcpyKind kind, cudaStream_t stream) {
  (void)destination;
  (void)source;
  (void)count;
  (void)kind;
  (void)stream;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_memcpy_from_symbol(void* destination, const void* symbol,
                                                   size_t count, size_t offset,
                                                   enum cudaMemcpyKind kind) {
  (void)destination;
  (void)symbol;
  (void)count;
  (void)offset;
  (void)kind;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_memcpy_to_symbol(const void* symbol, const void* source,
                                                 size_t count, size_t offset,
                                                 enum cudaMemcpyKind kind) {
  (void)symbol;
  (void)source;
  (void)count;
  (void)offset;
  (void)kind;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_memset_async(void* pointer, int value, size_t count,
                                             cudaStream_t stream) {
  (void)pointer;
  (void)value;
  (void)count;
  (void)stream;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_memset(void* pointer, int value, size_t count) {
  (void)pointer;
  (void)value;
  (void)count;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_pointer_get_attributes(struct cudaPointerAttributes* attributes,
                                                       const void* pointer) {
  (void)pointer;
  if (attributes != NULL) {
    memset(attributes, 0, sizeof(*attributes));
    attributes->type = cudaMemoryTypeUnregistered;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_runtime_get_version(int* version) {
  if (version != NULL) {
    *version = 0;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_set_device(int device) {
  (void)device;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_stream_begin_capture(cudaStream_t stream,
                                                     enum cudaStreamCaptureMode mode) {
  (void)stream;
  (void)mode;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_stream_begin_capture_to_graph(
    cudaStream_t stream, cudaGraph_t graph, const cudaGraphNode_t* dependencies,
    const cudaGraphEdgeData* dependency_data, size_t dependency_count,
    enum cudaStreamCaptureMode mode) {
  (void)stream;
  (void)graph;
  (void)dependencies;
  (void)dependency_data;
  (void)dependency_count;
  (void)mode;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_stream_create(cudaStream_t* stream, unsigned int flags) {
  (void)flags;
  if (stream != NULL) {
    *stream = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_stream_create_default(cudaStream_t* stream) {
  if (stream != NULL) {
    *stream = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_stream_destroy(cudaStream_t stream) {
  (void)stream;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_stream_end_capture(cudaStream_t stream, cudaGraph_t* graph) {
  (void)stream;
  if (graph != NULL) {
    *graph = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_stream_get_capture_info(cudaStream_t stream,
                                                        enum cudaStreamCaptureStatus* status,
                                                        unsigned long long* id, cudaGraph_t* graph,
                                                        const cudaGraphNode_t** dependencies,
                                                        size_t* dependency_count) {
  (void)stream;
  if (status != NULL) {
    *status = cudaStreamCaptureStatusNone;
  }
  if (id != NULL) {
    *id = 0;
  }
  if (graph != NULL) {
    *graph = NULL;
  }
  if (dependencies != NULL) {
    *dependencies = NULL;
  }
  if (dependency_count != NULL) {
    *dependency_count = 0;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_stream_get_device(cudaStream_t stream, int* device) {
  (void)stream;
  if (device != NULL) {
    *device = 0;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_stream_is_capturing(cudaStream_t stream,
                                                    enum cudaStreamCaptureStatus* status) {
  (void)stream;
  if (status != NULL) {
    *status = cudaStreamCaptureStatusNone;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_stream_synchronize(cudaStream_t stream) {
  (void)stream;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_stream_update_capture_dependencies(cudaStream_t stream,
                                                                   cudaGraphNode_t* dependencies,
                                                                   size_t dependency_count,
                                                                   unsigned int flags) {
  (void)stream;
  (void)dependencies;
  (void)dependency_count;
  (void)flags;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_stream_wait_event(cudaStream_t stream, cudaEvent_t event,
                                                  unsigned int flags) {
  (void)stream;
  (void)event;
  (void)flags;
  return cudaErrorSharedObjectSymbolNotFound;
}

/* NVCC's default-priority constructors execute these registration entry points
 * during DSO load. A stable writable token keeps their failure path well-formed
 * when cudart is absent; the remaining registration operations are exact-signature
 * no-ops because no CUDA code can run after the cohort failed. */
static void* gpu_xtb_fatbinary_token_storage;

static void** gpu_xtb_cuda_register_fat_binary(void* fat_cubin) {
  (void)fat_cubin;
  return &gpu_xtb_fatbinary_token_storage;
}

static void gpu_xtb_cuda_register_fat_binary_end(void** fat_cubin_handle) {
  (void)fat_cubin_handle;
}

static void gpu_xtb_cuda_unregister_fat_binary(void** fat_cubin_handle) { (void)fat_cubin_handle; }

static void gpu_xtb_cuda_register_function(void** fat_cubin_handle, const char* host_function,
                                           char* device_function, const char* device_name,
                                           int thread_limit, uint3* thread_id, uint3* block_id,
                                           dim3* block_dimension, dim3* grid_dimension,
                                           int* warp_size) {
  (void)fat_cubin_handle;
  (void)host_function;
  (void)device_function;
  (void)device_name;
  (void)thread_limit;
  (void)thread_id;
  (void)block_id;
  (void)block_dimension;
  (void)grid_dimension;
  (void)warp_size;
}

static void gpu_xtb_cuda_register_var(void** fat_cubin_handle, char* host_variable,
                                      char* device_address, const char* device_name, int external,
                                      size_t size, int constant, int global) {
  (void)fat_cubin_handle;
  (void)host_variable;
  (void)device_address;
  (void)device_name;
  (void)external;
  (void)size;
  (void)constant;
  (void)global;
}

static char gpu_xtb_cuda_init_module(void** fat_cubin_handle) {
  (void)fat_cubin_handle;
  return 0;
}

static unsigned int gpu_xtb_cuda_push_call_configuration(dim3 grid_dimension, dim3 block_dimension,
                                                         size_t shared_memory,
                                                         cudaStream_t stream) {
  (void)grid_dimension;
  (void)block_dimension;
  (void)shared_memory;
  (void)stream;
  return (unsigned int)cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t gpu_xtb_cuda_pop_call_configuration(dim3* grid_dimension, dim3* block_dimension,
                                                       size_t* shared_memory, void* stream) {
  if (grid_dimension != NULL) {
    memset(grid_dimension, 0, sizeof(*grid_dimension));
  }
  if (block_dimension != NULL) {
    memset(block_dimension, 0, sizeof(*block_dimension));
  }
  if (shared_memory != NULL) {
    *shared_memory = 0;
  }
  if (stream != NULL) {
    *(cudaStream_t*)stream = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

/* cuBLAS fallbacks. */
static cublasStatus_t gpu_xtb_cublas_create(cublasHandle_t* handle) {
  if (handle != NULL) {
    *handle = NULL;
  }
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

static cublasStatus_t gpu_xtb_cublas_destroy(cublasHandle_t handle) {
  (void)handle;
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

static cublasStatus_t gpu_xtb_cublas_dtrsm_batched(
    cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t fill,
    cublasOperation_t operation, cublasDiagType_t diagonal, int row_count, int column_count,
    const double* alpha, const double* const matrices_a[], int leading_dimension_a,
    double* const matrices_b[], int leading_dimension_b, int batch_count) {
  (void)handle;
  (void)side;
  (void)fill;
  (void)operation;
  (void)diagonal;
  (void)row_count;
  (void)column_count;
  (void)alpha;
  (void)matrices_a;
  (void)leading_dimension_a;
  (void)matrices_b;
  (void)leading_dimension_b;
  (void)batch_count;
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

static cublasStatus_t gpu_xtb_cublas_get_math_mode(cublasHandle_t handle, cublasMath_t* mode) {
  (void)handle;
  if (mode != NULL) {
    *mode = CUBLAS_DEFAULT_MATH;
  }
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

static cublasStatus_t gpu_xtb_cublas_get_pointer_mode(cublasHandle_t handle,
                                                      cublasPointerMode_t* mode) {
  (void)handle;
  if (mode != NULL) {
    *mode = CUBLAS_POINTER_MODE_HOST;
  }
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

static cublasStatus_t gpu_xtb_cublas_get_stream(cublasHandle_t handle, cudaStream_t* stream) {
  (void)handle;
  if (stream != NULL) {
    *stream = NULL;
  }
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

static cublasStatus_t gpu_xtb_cublas_get_version(cublasHandle_t handle, int* version) {
  (void)handle;
  if (version != NULL) {
    *version = 0;
  }
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

static cublasStatus_t gpu_xtb_cublas_set_math_mode(cublasHandle_t handle, cublasMath_t mode) {
  (void)handle;
  (void)mode;
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

static cublasStatus_t gpu_xtb_cublas_set_pointer_mode(cublasHandle_t handle,
                                                      cublasPointerMode_t mode) {
  (void)handle;
  (void)mode;
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

static cublasStatus_t gpu_xtb_cublas_set_stream(cublasHandle_t handle, cudaStream_t stream) {
  (void)handle;
  (void)stream;
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

static cublasStatus_t gpu_xtb_cublas_set_workspace(cublasHandle_t handle, void* workspace,
                                                   size_t workspace_size) {
  (void)handle;
  (void)workspace;
  (void)workspace_size;
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

/* cuSOLVER fallbacks. Device-resident info arrays are not dereferenced. */
static cusolverStatus_t gpu_xtb_cusolver_create(cusolverDnHandle_t* handle) {
  if (handle != NULL) {
    *handle = NULL;
  }
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t gpu_xtb_cusolver_create_params(cusolverDnParams_t* parameters) {
  if (parameters != NULL) {
    *parameters = NULL;
  }
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t gpu_xtb_cusolver_create_syevj_info(syevjInfo_t* information) {
  if (information != NULL) {
    *information = NULL;
  }
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t gpu_xtb_cusolver_destroy(cusolverDnHandle_t handle) {
  (void)handle;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t gpu_xtb_cusolver_destroy_params(cusolverDnParams_t parameters) {
  (void)parameters;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t gpu_xtb_cusolver_destroy_syevj_info(syevjInfo_t information) {
  (void)information;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t gpu_xtb_cusolver_dpotrf_batched(cusolverDnHandle_t handle,
                                                        cublasFillMode_t fill, int dimension,
                                                        double* matrices[], int leading_dimension,
                                                        int* information, int batch_size) {
  (void)handle;
  (void)fill;
  (void)dimension;
  (void)matrices;
  (void)leading_dimension;
  (void)information;
  (void)batch_size;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t gpu_xtb_cusolver_dsyevj_batched_buffer_size(
    cusolverDnHandle_t handle, cusolverEigMode_t job, cublasFillMode_t fill, int dimension,
    const double* matrices, int leading_dimension, const double* eigenvalues, int* workspace_size,
    syevjInfo_t parameters, int batch_size) {
  (void)handle;
  (void)job;
  (void)fill;
  (void)dimension;
  (void)matrices;
  (void)leading_dimension;
  (void)eigenvalues;
  (void)parameters;
  (void)batch_size;
  if (workspace_size != NULL) {
    *workspace_size = 0;
  }
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t gpu_xtb_cusolver_dsyevj_batched(
    cusolverDnHandle_t handle, cusolverEigMode_t job, cublasFillMode_t fill, int dimension,
    double* matrices, int leading_dimension, double* eigenvalues, double* workspace,
    int workspace_size, int* information, syevjInfo_t parameters, int batch_size) {
  (void)handle;
  (void)job;
  (void)fill;
  (void)dimension;
  (void)matrices;
  (void)leading_dimension;
  (void)eigenvalues;
  (void)workspace;
  (void)workspace_size;
  (void)information;
  (void)parameters;
  (void)batch_size;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t gpu_xtb_cusolver_get_stream(cusolverDnHandle_t handle,
                                                    cudaStream_t* stream) {
  (void)handle;
  if (stream != NULL) {
    *stream = NULL;
  }
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t gpu_xtb_cusolver_set_stream(cusolverDnHandle_t handle,
                                                    cudaStream_t stream) {
  (void)handle;
  (void)stream;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t gpu_xtb_cusolver_syevj_set_max_sweeps(syevjInfo_t information,
                                                              int max_sweeps) {
  (void)information;
  (void)max_sweeps;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t gpu_xtb_cusolver_syevj_set_sort_eig(syevjInfo_t information,
                                                            int sort_eigenvalues) {
  (void)information;
  (void)sort_eigenvalues;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t gpu_xtb_cusolver_syevj_set_tolerance(syevjInfo_t information,
                                                             double tolerance) {
  (void)information;
  (void)tolerance;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t gpu_xtb_cusolver_xsyev_batched(
    cusolverDnHandle_t handle, cusolverDnParams_t parameters, cusolverEigMode_t job,
    cublasFillMode_t fill, int64_t dimension, cudaDataType matrix_type, void* matrices,
    int64_t leading_dimension, cudaDataType eigenvalue_type, void* eigenvalues,
    cudaDataType compute_type, void* device_workspace, size_t device_workspace_size,
    void* host_workspace, size_t host_workspace_size, int* information, int64_t batch_size) {
  (void)handle;
  (void)parameters;
  (void)job;
  (void)fill;
  (void)dimension;
  (void)matrix_type;
  (void)matrices;
  (void)leading_dimension;
  (void)eigenvalue_type;
  (void)eigenvalues;
  (void)compute_type;
  (void)device_workspace;
  (void)device_workspace_size;
  (void)host_workspace;
  (void)host_workspace_size;
  (void)information;
  (void)batch_size;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t gpu_xtb_cusolver_xsyev_batched_buffer_size(
    cusolverDnHandle_t handle, cusolverDnParams_t parameters, cusolverEigMode_t job,
    cublasFillMode_t fill, int64_t dimension, cudaDataType matrix_type, const void* matrices,
    int64_t leading_dimension, cudaDataType eigenvalue_type, const void* eigenvalues,
    cudaDataType compute_type, size_t* device_workspace_size, size_t* host_workspace_size,
    int64_t batch_size) {
  (void)handle;
  (void)parameters;
  (void)job;
  (void)fill;
  (void)dimension;
  (void)matrix_type;
  (void)matrices;
  (void)leading_dimension;
  (void)eigenvalue_type;
  (void)eigenvalues;
  (void)compute_type;
  (void)batch_size;
  if (device_workspace_size != NULL) {
    *device_workspace_size = 0;
  }
  if (host_workspace_size != NULL) {
    *host_workspace_size = 0;
  }
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t gpu_xtb_cusolver_get_property(libraryPropertyType property, int* value) {
  (void)property;
  if (value != NULL) {
    *value = 0;
  }
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

/* Driver fallbacks. Driver queries return CUDA_ERROR_NOT_INITIALIZED, which is
 * the driver API's documented initialization failure rather than cudart 302. */
static CUresult gpu_xtb_cu_get_error_text(CUresult error, const char** text) {
  (void)error;
  if (text != NULL) {
    *text = gpu_xtb_cuda_unavailable_text;
  }
  return CUDA_ERROR_NOT_INITIALIZED;
}

static CUresult gpu_xtb_cu_mem_get_address_range(CUdeviceptr* base, size_t* size,
                                                 CUdeviceptr pointer) {
  (void)pointer;
  if (base != NULL) {
    *base = 0;
  }
  if (size != NULL) {
    *size = 0;
  }
  return CUDA_ERROR_NOT_INITIALIZED;
}

/* Keep the curated fallbacks tied to the CUDA 12.9 declarations used by the
 * build. A header or manifest change that alters one function's ABI must fail
 * compilation here instead of reaching a trampoline call with a stale type. */
#if defined(__GNUC__) || defined(__clang__)
#define GPU_XTB_CHECK_SIGNATURE(api, fallback)                                              \
  _Static_assert(__builtin_types_compatible_p(__typeof__(&(api)), __typeof__(&(fallback))), \
                 "CUDA fallback signature does not match " #api)

GPU_XTB_CHECK_SIGNATURE(cudaDeviceGetAttribute, gpu_xtb_cuda_device_get_attribute);
GPU_XTB_CHECK_SIGNATURE(cudaDeviceSynchronize, gpu_xtb_cudart_error);
GPU_XTB_CHECK_SIGNATURE(cudaEventCreate, gpu_xtb_cuda_event_create_default);
GPU_XTB_CHECK_SIGNATURE(cudaEventCreateWithFlags, gpu_xtb_cuda_event_create);
GPU_XTB_CHECK_SIGNATURE(cudaEventDestroy, gpu_xtb_cuda_event_destroy);
GPU_XTB_CHECK_SIGNATURE(cudaEventElapsedTime, gpu_xtb_cuda_event_elapsed_time);
GPU_XTB_CHECK_SIGNATURE(cudaEventRecord, gpu_xtb_cuda_event_record);
GPU_XTB_CHECK_SIGNATURE(cudaEventSynchronize, gpu_xtb_cuda_event_synchronize);
GPU_XTB_CHECK_SIGNATURE(cudaFree, gpu_xtb_cuda_free);
GPU_XTB_CHECK_SIGNATURE(cudaFreeHost, gpu_xtb_cuda_free);
GPU_XTB_CHECK_SIGNATURE(cudaGetDevice, gpu_xtb_cuda_get_device);
GPU_XTB_CHECK_SIGNATURE(cudaGetDeviceCount, gpu_xtb_cuda_get_device_count);
GPU_XTB_CHECK_SIGNATURE(cudaGetErrorName, gpu_xtb_cuda_get_error_text);
GPU_XTB_CHECK_SIGNATURE(cudaGetErrorString, gpu_xtb_cuda_get_error_text);
GPU_XTB_CHECK_SIGNATURE(cudaGetLastError, gpu_xtb_cudart_error);
GPU_XTB_CHECK_SIGNATURE(cudaGraphAddNode, gpu_xtb_cuda_graph_add_node);
GPU_XTB_CHECK_SIGNATURE(cudaGraphChildGraphNodeGetGraph, gpu_xtb_cuda_graph_child_get_graph);
GPU_XTB_CHECK_SIGNATURE(cudaGraphConditionalHandleCreate,
                        gpu_xtb_cuda_graph_conditional_handle_create);
GPU_XTB_CHECK_SIGNATURE(cudaGraphCreate, gpu_xtb_cuda_graph_create);
GPU_XTB_CHECK_SIGNATURE(cudaGraphDestroy, gpu_xtb_cuda_graph_destroy);
GPU_XTB_CHECK_SIGNATURE(cudaGraphExecDestroy, gpu_xtb_cuda_graph_exec_destroy);
GPU_XTB_CHECK_SIGNATURE(cudaGraphGetNodes, gpu_xtb_cuda_graph_get_nodes);
GPU_XTB_CHECK_SIGNATURE(cudaGraphInstantiate, gpu_xtb_cuda_graph_instantiate);
GPU_XTB_CHECK_SIGNATURE(cudaGraphLaunch, gpu_xtb_cuda_graph_launch);
GPU_XTB_CHECK_SIGNATURE(cudaGraphMemcpyNodeGetParams, gpu_xtb_cuda_graph_memcpy_node_get_params);
GPU_XTB_CHECK_SIGNATURE(cudaGraphNodeGetType, gpu_xtb_cuda_graph_node_get_type);
GPU_XTB_CHECK_SIGNATURE(cudaGraphUpload, gpu_xtb_cuda_graph_upload);
GPU_XTB_CHECK_SIGNATURE(cudaHostGetFlags, gpu_xtb_cuda_host_get_flags);
GPU_XTB_CHECK_SIGNATURE(cudaHostRegister, gpu_xtb_cuda_host_register);
GPU_XTB_CHECK_SIGNATURE(cudaHostUnregister, gpu_xtb_cuda_host_unregister);
GPU_XTB_CHECK_SIGNATURE(cudaLaunchHostFunc, gpu_xtb_cuda_launch_host_func);
GPU_XTB_CHECK_SIGNATURE(cudaLaunchKernel, gpu_xtb_cuda_launch_kernel);
GPU_XTB_CHECK_SIGNATURE(cudaMalloc, gpu_xtb_cuda_malloc);
GPU_XTB_CHECK_SIGNATURE(cudaMallocHost, gpu_xtb_cuda_malloc);
GPU_XTB_CHECK_SIGNATURE(cudaMallocManaged, gpu_xtb_cuda_malloc_managed);
GPU_XTB_CHECK_SIGNATURE(cudaMemcpy, gpu_xtb_cuda_memcpy);
GPU_XTB_CHECK_SIGNATURE(cudaMemcpyAsync, gpu_xtb_cuda_memcpy_async);
GPU_XTB_CHECK_SIGNATURE(cudaMemcpyFromSymbol, gpu_xtb_cuda_memcpy_from_symbol);
GPU_XTB_CHECK_SIGNATURE(cudaMemcpyToSymbol, gpu_xtb_cuda_memcpy_to_symbol);
GPU_XTB_CHECK_SIGNATURE(cudaMemset, gpu_xtb_cuda_memset);
GPU_XTB_CHECK_SIGNATURE(cudaMemsetAsync, gpu_xtb_cuda_memset_async);
GPU_XTB_CHECK_SIGNATURE(cudaPeekAtLastError, gpu_xtb_cudart_error);
GPU_XTB_CHECK_SIGNATURE(cudaPointerGetAttributes, gpu_xtb_cuda_pointer_get_attributes);
GPU_XTB_CHECK_SIGNATURE(cudaRuntimeGetVersion, gpu_xtb_cuda_runtime_get_version);
GPU_XTB_CHECK_SIGNATURE(cudaSetDevice, gpu_xtb_cuda_set_device);
GPU_XTB_CHECK_SIGNATURE(cudaStreamBeginCapture, gpu_xtb_cuda_stream_begin_capture);
GPU_XTB_CHECK_SIGNATURE(cudaStreamBeginCaptureToGraph, gpu_xtb_cuda_stream_begin_capture_to_graph);
GPU_XTB_CHECK_SIGNATURE(cudaStreamCreate, gpu_xtb_cuda_stream_create_default);
GPU_XTB_CHECK_SIGNATURE(cudaStreamCreateWithFlags, gpu_xtb_cuda_stream_create);
GPU_XTB_CHECK_SIGNATURE(cudaStreamDestroy, gpu_xtb_cuda_stream_destroy);
GPU_XTB_CHECK_SIGNATURE(cudaStreamEndCapture, gpu_xtb_cuda_stream_end_capture);
GPU_XTB_CHECK_SIGNATURE(cudaStreamGetCaptureInfo_v2, gpu_xtb_cuda_stream_get_capture_info);
GPU_XTB_CHECK_SIGNATURE(cudaStreamGetDevice, gpu_xtb_cuda_stream_get_device);
GPU_XTB_CHECK_SIGNATURE(cudaStreamIsCapturing, gpu_xtb_cuda_stream_is_capturing);
GPU_XTB_CHECK_SIGNATURE(cudaStreamSynchronize, gpu_xtb_cuda_stream_synchronize);
GPU_XTB_CHECK_SIGNATURE(cudaStreamUpdateCaptureDependencies,
                        gpu_xtb_cuda_stream_update_capture_dependencies);
GPU_XTB_CHECK_SIGNATURE(cudaStreamWaitEvent, gpu_xtb_cuda_stream_wait_event);

GPU_XTB_CHECK_SIGNATURE(cublasCreate_v2, gpu_xtb_cublas_create);
GPU_XTB_CHECK_SIGNATURE(cublasDestroy_v2, gpu_xtb_cublas_destroy);
GPU_XTB_CHECK_SIGNATURE(cublasDtrsmBatched, gpu_xtb_cublas_dtrsm_batched);
GPU_XTB_CHECK_SIGNATURE(cublasGetMathMode, gpu_xtb_cublas_get_math_mode);
GPU_XTB_CHECK_SIGNATURE(cublasGetPointerMode_v2, gpu_xtb_cublas_get_pointer_mode);
GPU_XTB_CHECK_SIGNATURE(cublasGetStream_v2, gpu_xtb_cublas_get_stream);
GPU_XTB_CHECK_SIGNATURE(cublasGetVersion_v2, gpu_xtb_cublas_get_version);
GPU_XTB_CHECK_SIGNATURE(cublasSetMathMode, gpu_xtb_cublas_set_math_mode);
GPU_XTB_CHECK_SIGNATURE(cublasSetPointerMode_v2, gpu_xtb_cublas_set_pointer_mode);
GPU_XTB_CHECK_SIGNATURE(cublasSetStream_v2, gpu_xtb_cublas_set_stream);
GPU_XTB_CHECK_SIGNATURE(cublasSetWorkspace_v2, gpu_xtb_cublas_set_workspace);

GPU_XTB_CHECK_SIGNATURE(cusolverDnCreate, gpu_xtb_cusolver_create);
GPU_XTB_CHECK_SIGNATURE(cusolverDnCreateParams, gpu_xtb_cusolver_create_params);
GPU_XTB_CHECK_SIGNATURE(cusolverDnCreateSyevjInfo, gpu_xtb_cusolver_create_syevj_info);
GPU_XTB_CHECK_SIGNATURE(cusolverDnDestroy, gpu_xtb_cusolver_destroy);
GPU_XTB_CHECK_SIGNATURE(cusolverDnDestroyParams, gpu_xtb_cusolver_destroy_params);
GPU_XTB_CHECK_SIGNATURE(cusolverDnDestroySyevjInfo, gpu_xtb_cusolver_destroy_syevj_info);
GPU_XTB_CHECK_SIGNATURE(cusolverDnDpotrfBatched, gpu_xtb_cusolver_dpotrf_batched);
GPU_XTB_CHECK_SIGNATURE(cusolverDnDsyevjBatched, gpu_xtb_cusolver_dsyevj_batched);
GPU_XTB_CHECK_SIGNATURE(cusolverDnDsyevjBatched_bufferSize,
                        gpu_xtb_cusolver_dsyevj_batched_buffer_size);
GPU_XTB_CHECK_SIGNATURE(cusolverDnGetStream, gpu_xtb_cusolver_get_stream);
GPU_XTB_CHECK_SIGNATURE(cusolverDnSetStream, gpu_xtb_cusolver_set_stream);
GPU_XTB_CHECK_SIGNATURE(cusolverDnXsyevjSetMaxSweeps, gpu_xtb_cusolver_syevj_set_max_sweeps);
GPU_XTB_CHECK_SIGNATURE(cusolverDnXsyevjSetSortEig, gpu_xtb_cusolver_syevj_set_sort_eig);
GPU_XTB_CHECK_SIGNATURE(cusolverDnXsyevjSetTolerance, gpu_xtb_cusolver_syevj_set_tolerance);
GPU_XTB_CHECK_SIGNATURE(cusolverDnXsyevBatched, gpu_xtb_cusolver_xsyev_batched);
GPU_XTB_CHECK_SIGNATURE(cusolverDnXsyevBatched_bufferSize,
                        gpu_xtb_cusolver_xsyev_batched_buffer_size);
GPU_XTB_CHECK_SIGNATURE(cusolverGetProperty, gpu_xtb_cusolver_get_property);

GPU_XTB_CHECK_SIGNATURE(cuGetErrorString, gpu_xtb_cu_get_error_text);
GPU_XTB_CHECK_SIGNATURE(cuMemGetAddressRange_v2, gpu_xtb_cu_mem_get_address_range);

typedef void** (*gpu_xtb_register_fat_binary_signature)(void*);
typedef void (*gpu_xtb_register_fat_binary_end_signature)(void**);
typedef void (*gpu_xtb_unregister_fat_binary_signature)(void**);
typedef void (*gpu_xtb_register_function_signature)(void**, const char*, char*, const char*, int,
                                                    uint3*, uint3*, dim3*, dim3*, int*);
typedef void (*gpu_xtb_register_var_signature)(void**, char*, char*, const char*, int, size_t, int,
                                               int);
typedef char (*gpu_xtb_init_module_signature)(void**);
typedef unsigned int (*gpu_xtb_push_call_configuration_signature)(dim3, dim3, size_t, cudaStream_t);
typedef cudaError_t (*gpu_xtb_pop_call_configuration_signature)(dim3*, dim3*, size_t*, void*);

_Static_assert(__builtin_types_compatible_p(gpu_xtb_register_fat_binary_signature,
                                            __typeof__(&gpu_xtb_cuda_register_fat_binary)),
               "__cudaRegisterFatBinary fallback signature mismatch");
_Static_assert(__builtin_types_compatible_p(gpu_xtb_register_fat_binary_end_signature,
                                            __typeof__(&gpu_xtb_cuda_register_fat_binary_end)),
               "__cudaRegisterFatBinaryEnd fallback signature mismatch");
_Static_assert(__builtin_types_compatible_p(gpu_xtb_unregister_fat_binary_signature,
                                            __typeof__(&gpu_xtb_cuda_unregister_fat_binary)),
               "__cudaUnregisterFatBinary fallback signature mismatch");
_Static_assert(__builtin_types_compatible_p(gpu_xtb_register_function_signature,
                                            __typeof__(&gpu_xtb_cuda_register_function)),
               "__cudaRegisterFunction fallback signature mismatch");
_Static_assert(__builtin_types_compatible_p(gpu_xtb_register_var_signature,
                                            __typeof__(&gpu_xtb_cuda_register_var)),
               "__cudaRegisterVar fallback signature mismatch");
_Static_assert(__builtin_types_compatible_p(gpu_xtb_init_module_signature,
                                            __typeof__(&gpu_xtb_cuda_init_module)),
               "__cudaInitModule fallback signature mismatch");
_Static_assert(__builtin_types_compatible_p(gpu_xtb_push_call_configuration_signature,
                                            __typeof__(&gpu_xtb_cuda_push_call_configuration)),
               "__cudaPushCallConfiguration fallback signature mismatch");
_Static_assert(__builtin_types_compatible_p(gpu_xtb_pop_call_configuration_signature,
                                            __typeof__(&gpu_xtb_cuda_pop_call_configuration)),
               "__cudaPopCallConfiguration fallback signature mismatch");

#undef GPU_XTB_CHECK_SIGNATURE
#endif

static void* gpu_xtb_fallback_for_symbol(const char* symbol) {
#define GPU_XTB_SYMBOL(name, fallback)      \
  if (strcmp(symbol, name) == 0) {          \
    union {                                 \
      __typeof__(&fallback) function;       \
      void* object;                         \
    } conversion = {.function = &fallback}; \
    return conversion.object;               \
  }

  GPU_XTB_SYMBOL("cudaEventCreateWithFlags", gpu_xtb_cuda_event_create)
  GPU_XTB_SYMBOL("cudaEventCreate", gpu_xtb_cuda_event_create_default)
  GPU_XTB_SYMBOL("cudaEventDestroy", gpu_xtb_cuda_event_destroy)
  GPU_XTB_SYMBOL("cudaEventElapsedTime", gpu_xtb_cuda_event_elapsed_time)
  GPU_XTB_SYMBOL("cudaEventRecord", gpu_xtb_cuda_event_record)
  GPU_XTB_SYMBOL("cudaEventSynchronize", gpu_xtb_cuda_event_synchronize)
  GPU_XTB_SYMBOL("cudaFree", gpu_xtb_cuda_free)
  GPU_XTB_SYMBOL("cudaFreeHost", gpu_xtb_cuda_free)
  GPU_XTB_SYMBOL("cudaGetDevice", gpu_xtb_cuda_get_device)
  GPU_XTB_SYMBOL("cudaGetDeviceCount", gpu_xtb_cuda_get_device_count)
  GPU_XTB_SYMBOL("cudaDeviceGetAttribute", gpu_xtb_cuda_device_get_attribute)
  GPU_XTB_SYMBOL("cudaDeviceSynchronize", gpu_xtb_cudart_error)
  GPU_XTB_SYMBOL("cudaGetErrorName", gpu_xtb_cuda_get_error_text)
  GPU_XTB_SYMBOL("cudaGetErrorString", gpu_xtb_cuda_get_error_text)
  GPU_XTB_SYMBOL("cudaGetLastError", gpu_xtb_cudart_error)
  GPU_XTB_SYMBOL("cudaGraphAddNode", gpu_xtb_cuda_graph_add_node)
  GPU_XTB_SYMBOL("cudaGraphChildGraphNodeGetGraph", gpu_xtb_cuda_graph_child_get_graph)
  GPU_XTB_SYMBOL("cudaGraphConditionalHandleCreate", gpu_xtb_cuda_graph_conditional_handle_create)
  GPU_XTB_SYMBOL("cudaGraphCreate", gpu_xtb_cuda_graph_create)
  GPU_XTB_SYMBOL("cudaGraphDestroy", gpu_xtb_cuda_graph_destroy)
  GPU_XTB_SYMBOL("cudaGraphExecDestroy", gpu_xtb_cuda_graph_exec_destroy)
  GPU_XTB_SYMBOL("cudaGraphGetNodes", gpu_xtb_cuda_graph_get_nodes)
  GPU_XTB_SYMBOL("cudaGraphMemcpyNodeGetParams", gpu_xtb_cuda_graph_memcpy_node_get_params)
  GPU_XTB_SYMBOL("cudaGraphInstantiate", gpu_xtb_cuda_graph_instantiate)
  GPU_XTB_SYMBOL("cudaGraphLaunch", gpu_xtb_cuda_graph_launch)
  GPU_XTB_SYMBOL("cudaGraphNodeGetType", gpu_xtb_cuda_graph_node_get_type)
  GPU_XTB_SYMBOL("cudaGraphUpload", gpu_xtb_cuda_graph_upload)
  GPU_XTB_SYMBOL("cudaHostGetFlags", gpu_xtb_cuda_host_get_flags)
  GPU_XTB_SYMBOL("cudaHostRegister", gpu_xtb_cuda_host_register)
  GPU_XTB_SYMBOL("cudaHostUnregister", gpu_xtb_cuda_host_unregister)
  GPU_XTB_SYMBOL("cudaLaunchHostFunc", gpu_xtb_cuda_launch_host_func)
  GPU_XTB_SYMBOL("cudaLaunchKernel", gpu_xtb_cuda_launch_kernel)
  GPU_XTB_SYMBOL("cudaMalloc", gpu_xtb_cuda_malloc)
  GPU_XTB_SYMBOL("cudaMallocHost", gpu_xtb_cuda_malloc)
  GPU_XTB_SYMBOL("cudaMallocManaged", gpu_xtb_cuda_malloc_managed)
  GPU_XTB_SYMBOL("cudaMemcpy", gpu_xtb_cuda_memcpy)
  GPU_XTB_SYMBOL("cudaMemcpyAsync", gpu_xtb_cuda_memcpy_async)
  GPU_XTB_SYMBOL("cudaMemcpyFromSymbol", gpu_xtb_cuda_memcpy_from_symbol)
  GPU_XTB_SYMBOL("cudaMemcpyToSymbol", gpu_xtb_cuda_memcpy_to_symbol)
  GPU_XTB_SYMBOL("cudaMemset", gpu_xtb_cuda_memset)
  GPU_XTB_SYMBOL("cudaMemsetAsync", gpu_xtb_cuda_memset_async)
  GPU_XTB_SYMBOL("cudaPeekAtLastError", gpu_xtb_cudart_error)
  GPU_XTB_SYMBOL("cudaPointerGetAttributes", gpu_xtb_cuda_pointer_get_attributes)
  GPU_XTB_SYMBOL("cudaRuntimeGetVersion", gpu_xtb_cuda_runtime_get_version)
  GPU_XTB_SYMBOL("cudaSetDevice", gpu_xtb_cuda_set_device)
  GPU_XTB_SYMBOL("cudaStreamBeginCapture", gpu_xtb_cuda_stream_begin_capture)
  GPU_XTB_SYMBOL("cudaStreamBeginCaptureToGraph", gpu_xtb_cuda_stream_begin_capture_to_graph)
  GPU_XTB_SYMBOL("cudaStreamCreate", gpu_xtb_cuda_stream_create_default)
  GPU_XTB_SYMBOL("cudaStreamCreateWithFlags", gpu_xtb_cuda_stream_create)
  GPU_XTB_SYMBOL("cudaStreamDestroy", gpu_xtb_cuda_stream_destroy)
  GPU_XTB_SYMBOL("cudaStreamEndCapture", gpu_xtb_cuda_stream_end_capture)
  GPU_XTB_SYMBOL("cudaStreamGetCaptureInfo_v2", gpu_xtb_cuda_stream_get_capture_info)
  GPU_XTB_SYMBOL("cudaStreamGetDevice", gpu_xtb_cuda_stream_get_device)
  GPU_XTB_SYMBOL("cudaStreamIsCapturing", gpu_xtb_cuda_stream_is_capturing)
  GPU_XTB_SYMBOL("cudaStreamSynchronize", gpu_xtb_cuda_stream_synchronize)
  GPU_XTB_SYMBOL("cudaStreamUpdateCaptureDependencies",
                 gpu_xtb_cuda_stream_update_capture_dependencies)
  GPU_XTB_SYMBOL("cudaStreamWaitEvent", gpu_xtb_cuda_stream_wait_event)
  GPU_XTB_SYMBOL("__cudaInitModule", gpu_xtb_cuda_init_module)
  GPU_XTB_SYMBOL("__cudaPopCallConfiguration", gpu_xtb_cuda_pop_call_configuration)
  GPU_XTB_SYMBOL("__cudaPushCallConfiguration", gpu_xtb_cuda_push_call_configuration)
  GPU_XTB_SYMBOL("__cudaRegisterFatBinary", gpu_xtb_cuda_register_fat_binary)
  GPU_XTB_SYMBOL("__cudaRegisterFatBinaryEnd", gpu_xtb_cuda_register_fat_binary_end)
  GPU_XTB_SYMBOL("__cudaRegisterFunction", gpu_xtb_cuda_register_function)
  GPU_XTB_SYMBOL("__cudaRegisterVar", gpu_xtb_cuda_register_var)
  GPU_XTB_SYMBOL("__cudaUnregisterFatBinary", gpu_xtb_cuda_unregister_fat_binary)

  GPU_XTB_SYMBOL("cublasCreate_v2", gpu_xtb_cublas_create)
  GPU_XTB_SYMBOL("cublasDestroy_v2", gpu_xtb_cublas_destroy)
  GPU_XTB_SYMBOL("cublasDtrsmBatched", gpu_xtb_cublas_dtrsm_batched)
  GPU_XTB_SYMBOL("cublasGetMathMode", gpu_xtb_cublas_get_math_mode)
  GPU_XTB_SYMBOL("cublasGetPointerMode_v2", gpu_xtb_cublas_get_pointer_mode)
  GPU_XTB_SYMBOL("cublasGetStream_v2", gpu_xtb_cublas_get_stream)
  GPU_XTB_SYMBOL("cublasGetVersion_v2", gpu_xtb_cublas_get_version)
  GPU_XTB_SYMBOL("cublasSetMathMode", gpu_xtb_cublas_set_math_mode)
  GPU_XTB_SYMBOL("cublasSetPointerMode_v2", gpu_xtb_cublas_set_pointer_mode)
  GPU_XTB_SYMBOL("cublasSetStream_v2", gpu_xtb_cublas_set_stream)
  GPU_XTB_SYMBOL("cublasSetWorkspace_v2", gpu_xtb_cublas_set_workspace)

  GPU_XTB_SYMBOL("cusolverDnCreate", gpu_xtb_cusolver_create)
  GPU_XTB_SYMBOL("cusolverDnCreateParams", gpu_xtb_cusolver_create_params)
  GPU_XTB_SYMBOL("cusolverDnCreateSyevjInfo", gpu_xtb_cusolver_create_syevj_info)
  GPU_XTB_SYMBOL("cusolverDnDestroy", gpu_xtb_cusolver_destroy)
  GPU_XTB_SYMBOL("cusolverDnDestroyParams", gpu_xtb_cusolver_destroy_params)
  GPU_XTB_SYMBOL("cusolverDnDestroySyevjInfo", gpu_xtb_cusolver_destroy_syevj_info)
  GPU_XTB_SYMBOL("cusolverDnDpotrfBatched", gpu_xtb_cusolver_dpotrf_batched)
  GPU_XTB_SYMBOL("cusolverDnDsyevjBatched", gpu_xtb_cusolver_dsyevj_batched)
  GPU_XTB_SYMBOL("cusolverDnDsyevjBatched_bufferSize", gpu_xtb_cusolver_dsyevj_batched_buffer_size)
  GPU_XTB_SYMBOL("cusolverDnGetStream", gpu_xtb_cusolver_get_stream)
  GPU_XTB_SYMBOL("cusolverDnSetStream", gpu_xtb_cusolver_set_stream)
  GPU_XTB_SYMBOL("cusolverDnXsyevjSetMaxSweeps", gpu_xtb_cusolver_syevj_set_max_sweeps)
  GPU_XTB_SYMBOL("cusolverDnXsyevjSetSortEig", gpu_xtb_cusolver_syevj_set_sort_eig)
  GPU_XTB_SYMBOL("cusolverDnXsyevjSetTolerance", gpu_xtb_cusolver_syevj_set_tolerance)
  GPU_XTB_SYMBOL("cusolverDnXsyevBatched", gpu_xtb_cusolver_xsyev_batched)
  GPU_XTB_SYMBOL("cusolverDnXsyevBatched_bufferSize", gpu_xtb_cusolver_xsyev_batched_buffer_size)
  GPU_XTB_SYMBOL("cusolverGetProperty", gpu_xtb_cusolver_get_property)

  GPU_XTB_SYMBOL("cuGetErrorString", gpu_xtb_cu_get_error_text)
  GPU_XTB_SYMBOL("cuMemGetAddressRange_v2", gpu_xtb_cu_mem_get_address_range)

#undef GPU_XTB_SYMBOL
  return NULL;
}

/* Generated implib initializers call this only after constructor(101) has
 * selected either a fully preflighted real handle or the missing-cohort
 * sentinel. Unknown symbols stay NULL: guessing a generic C signature would
 * turn a clean backend-unavailable error into undefined behavior. */
GPU_XTB_HIDDEN void* gpu_xtb_cuda_dlsym(void* handle, const char* symbol) {
  if (symbol == NULL || *symbol == '\0') {
    gpu_xtb_report_once(GPU_XTB_DIAGNOSTIC_UNKNOWN_SYMBOL, "<invalid>", symbol, NULL);
    return NULL;
  }

  if (handle != GPU_XTB_MISSING_COHORT) {
    void* resolved = dlsym(handle, symbol);
    if (resolved != NULL) {
      return resolved;
    }
    /* Preflight makes this path an internal consistency failure. Do not hide
     * it behind a fallback for only one member of an otherwise real cohort. */
    gpu_xtb_report_once(GPU_XTB_DIAGNOSTIC_UNKNOWN_SYMBOL, "preflighted CUDA library", symbol,
                        NULL);
    return NULL;
  }

  void* fallback = gpu_xtb_fallback_for_symbol(symbol);
  if (fallback == NULL) {
    gpu_xtb_report_once(GPU_XTB_DIAGNOSTIC_UNKNOWN_SYMBOL, "missing CUDA library", symbol, NULL);
  }
  return fallback;
}

typedef void (*gpu_xtb_set_handle_fn)(void* handle);
typedef void (*gpu_xtb_resolve_all_fn)(void);

static void gpu_xtb_prepare_cohort(const char* soname, const char* const* required_symbols,
                                   gpu_xtb_set_handle_fn set_handle,
                                   gpu_xtb_resolve_all_fn resolve_all) {
  const char* force_fallback = NULL;
#if defined(GPUXTB_CUDA_TEST_HOOKS)
  force_fallback = getenv("GPUXTB_CUDA_FORCE_FALLBACK");
#endif
  void* handle = NULL;
  if (force_fallback == NULL || strcmp(force_fallback, "1") != 0) {
    dlerror();
    handle = dlopen(soname, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
  }
  if (handle == NULL) {
    const char* error = dlerror();
    gpu_xtb_report_once(GPU_XTB_DIAGNOSTIC_LOAD, soname, NULL,
                        force_fallback != NULL && strcmp(force_fallback, "1") == 0
                            ? "forced by GPUXTB_CUDA_FORCE_FALLBACK=1"
                            : error);
    handle = GPU_XTB_MISSING_COHORT;
  } else {
    for (size_t index = 0; required_symbols[index] != NULL; ++index) {
      dlerror();
      if (dlsym(handle, required_symbols[index]) == NULL) {
        const char* error = dlerror();
        char detail[512];
        snprintf(detail, sizeof(detail), "missing required symbol %s%s%s", required_symbols[index],
                 error != NULL ? ": " : "", error != NULL ? error : "");
        gpu_xtb_report_once(GPU_XTB_DIAGNOSTIC_LOAD, soname, NULL, detail);
        handle = GPU_XTB_MISSING_COHORT;
        break;
      }
    }
  }

  set_handle(handle);
  resolve_all();
}

static pthread_once_t gpu_xtb_cuda_bootstrap_once = PTHREAD_ONCE_INIT;

static void gpu_xtb_cuda_bootstrap_impl(void) {
  /* cudart must be complete first: NVCC's later registration constructors call
   * its __cuda* hooks. The math and driver cohorts are populated in the same
   * early constructor so no ordinary CUDA call can observe a partial table. */
  gpu_xtb_prepare_cohort(GPUXTB_CUDART_SONAME, gpu_xtb_cudart_required_symbols,
                         _libcudart_so_tramp_set_handle, _libcudart_so_tramp_resolve_all);
  gpu_xtb_prepare_cohort(GPUXTB_CUBLAS_SONAME, gpu_xtb_cublas_required_symbols,
                         _libcublas_so_tramp_set_handle, _libcublas_so_tramp_resolve_all);
  gpu_xtb_prepare_cohort(GPUXTB_CUSOLVER_SONAME, gpu_xtb_cusolver_required_symbols,
                         _libcusolver_so_tramp_set_handle, _libcusolver_so_tramp_resolve_all);
  gpu_xtb_prepare_cohort(GPUXTB_CUDA_DRIVER_SONAME, gpu_xtb_cuda_driver_required_symbols,
                         _libcuda_so_tramp_set_handle, _libcuda_so_tramp_resolve_all);
}

static void __attribute__((constructor(101))) gpu_xtb_cuda_bootstrap(void) {
  (void)pthread_once(&gpu_xtb_cuda_bootstrap_once, gpu_xtb_cuda_bootstrap_impl);
}
