#ifndef _GNU_SOURCE
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define _GNU_SOURCE
#endif

#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cuda_dlopen_symbols.h"
#include "nvidia_host_api.h"

#if defined(_WIN32) || defined(__APPLE__)
#error "the xtbloom CUDA dlopen bootstrap requires ELF constructor ordering"
#endif

#ifndef RTLD_NODELETE
#error "the xtbloom CUDA dlopen bootstrap requires RTLD_NODELETE"
#endif

#ifndef XTBLOOM_CUDART_SONAME
#error "XTBLOOM_CUDART_SONAME must name the exact build CUDA runtime SONAME"
#endif
#ifndef XTBLOOM_CUBLAS_SONAME
#error "XTBLOOM_CUBLAS_SONAME must name the exact build cuBLAS SONAME"
#endif
#ifndef XTBLOOM_CUSOLVER_SONAME
#error "XTBLOOM_CUSOLVER_SONAME must name the exact build cuSOLVER SONAME"
#endif
#ifndef XTBLOOM_CUDA_DRIVER_SONAME
#error "XTBLOOM_CUDA_DRIVER_SONAME must name the CUDA driver SONAME"
#endif

#if defined(__GNUC__) || defined(__clang__)
#define XTBLOOM_HIDDEN __attribute__((visibility("hidden")))
#else
#define XTBLOOM_HIDDEN
#endif

/* implib derives these names from the deliberately stable shim input basenames
 * libcudart.so, libcublas.so, libcusolver.so, and libcuda.so. The generator is
 * invoked with --no-dlopen: this translation unit owns every handle and fills
 * every table before ordinary NVCC registration constructors can run. */
XTBLOOM_HIDDEN void _libcudart_so_tramp_set_handle(void* handle);
XTBLOOM_HIDDEN void _libcudart_so_tramp_resolve_all(void);
XTBLOOM_HIDDEN void _libcublas_so_tramp_set_handle(void* handle);
XTBLOOM_HIDDEN void _libcublas_so_tramp_resolve_all(void);
XTBLOOM_HIDDEN void _libcusolver_so_tramp_set_handle(void* handle);
XTBLOOM_HIDDEN void _libcusolver_so_tramp_resolve_all(void);
XTBLOOM_HIDDEN void _libcuda_so_tramp_set_handle(void* handle);
XTBLOOM_HIDDEN void _libcuda_so_tramp_resolve_all(void);

enum xtbloom_diagnostic_kind {
  XTBLOOM_DIAGNOSTIC_LOAD = 1u << 0,
  XTBLOOM_DIAGNOSTIC_UNKNOWN_SYMBOL = 1u << 1,
};

static pthread_mutex_t xtbloom_diagnostic_mutex = PTHREAD_MUTEX_INITIALIZER;
static unsigned int xtbloom_reported_diagnostics;

static void xtbloom_report_once(enum xtbloom_diagnostic_kind kind, const char* library,
                                const char* symbol, const char* detail) {
  pthread_mutex_lock(&xtbloom_diagnostic_mutex);
  if ((xtbloom_reported_diagnostics & (unsigned int)kind) == 0u) {
    xtbloom_reported_diagnostics |= (unsigned int)kind;
    if (symbol != NULL) {
      fprintf(stderr,
              "xtbloom: CUDA trampoline has no ABI-correct fallback for '%s' "
              "from %s; the GPU backend is disabled\n",
              symbol, library != NULL ? library : "an unknown cohort");
    } else {
      fprintf(stderr,
              "xtbloom: CUDA cohort %s is unavailable%s%s; the GPU backend "
              "is disabled\n",
              library != NULL ? library : "<unknown>", detail != NULL ? ": " : "",
              detail != NULL ? detail : "");
    }
  }
  pthread_mutex_unlock(&xtbloom_diagnostic_mutex);
}

/* A failed cohort uses one impossible dlopen handle. xtbloom_cuda_dlsym then
 * installs typed fallbacks for every curated symbol rather than mixing real
 * entry points with substitutes from an ABI-incompatible library version. */
static unsigned char xtbloom_missing_cohort_sentinel;
#define XTBLOOM_MISSING_COHORT ((void*)&xtbloom_missing_cohort_sentinel)

static const char xtbloom_cuda_unavailable_text[] =
    "CUDA symbol unavailable because the xtbloom NVIDIA runtime cohort did not load";

/* CUDA runtime fallbacks. Calls return cudaErrorSharedObjectSymbolNotFound
 * (302), while output-only arguments are put into deterministic empty states.
 * Device output arrays are deliberately not dereferenced. */
static cudaError_t xtbloom_cudart_error(void) { return cudaErrorSharedObjectSymbolNotFound; }

static const char* xtbloom_cuda_get_error_text(cudaError_t error) {
  (void)error;
  return xtbloom_cuda_unavailable_text;
}

static cudaError_t xtbloom_cuda_event_create(cudaEvent_t* event, unsigned int flags) {
  (void)flags;
  if (event != NULL) {
    *event = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_event_create_default(cudaEvent_t* event) {
  if (event != NULL) {
    *event = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_event_destroy(cudaEvent_t event) {
  (void)event;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_event_record(cudaEvent_t event, cudaStream_t stream) {
  (void)event;
  (void)stream;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_event_synchronize(cudaEvent_t event) {
  (void)event;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_event_elapsed_time(float* milliseconds, cudaEvent_t start,
                                                   cudaEvent_t end) {
  (void)start;
  (void)end;
  if (milliseconds != NULL) {
    *milliseconds = 0.0F;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_free(void* pointer) {
  (void)pointer;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_get_device(int* device) {
  if (device != NULL) {
    *device = 0;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_get_device_count(int* count) {
  if (count != NULL) {
    *count = 0;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_device_get_attribute(int* value, enum cudaDeviceAttr attribute,
                                                     int device) {
  (void)attribute;
  (void)device;
  if (value != NULL) {
    *value = 0;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_graph_add_node(cudaGraphNode_t* node, cudaGraph_t graph,
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

static cudaError_t xtbloom_cuda_graph_child_get_graph(cudaGraphNode_t node, cudaGraph_t* graph) {
  (void)node;
  if (graph != NULL) {
    *graph = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_graph_conditional_handle_create(cudaGraphConditionalHandle* handle,
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

static cudaError_t xtbloom_cuda_graph_create(cudaGraph_t* graph, unsigned int flags) {
  (void)flags;
  if (graph != NULL) {
    *graph = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_graph_destroy(cudaGraph_t graph) {
  (void)graph;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_graph_exec_destroy(cudaGraphExec_t graph_exec) {
  (void)graph_exec;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_graph_get_nodes(cudaGraph_t graph, cudaGraphNode_t* nodes,
                                                size_t* node_count) {
  (void)graph;
  (void)nodes;
  if (node_count != NULL) {
    *node_count = 0;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_graph_memcpy_node_get_params(cudaGraphNode_t node,
                                                             struct cudaMemcpy3DParms* parameters) {
  (void)node;
  if (parameters != NULL) {
    memset(parameters, 0, sizeof(*parameters));
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_graph_instantiate(cudaGraphExec_t* graph_exec, cudaGraph_t graph,
                                                  unsigned long long flags) {
  (void)graph;
  (void)flags;
  if (graph_exec != NULL) {
    *graph_exec = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_graph_launch(cudaGraphExec_t graph_exec, cudaStream_t stream) {
  (void)graph_exec;
  (void)stream;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_graph_node_get_type(cudaGraphNode_t node,
                                                    enum cudaGraphNodeType* type) {
  (void)node;
  if (type != NULL) {
    *type = cudaGraphNodeTypeEmpty;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_graph_upload(cudaGraphExec_t graph_exec, cudaStream_t stream) {
  (void)graph_exec;
  (void)stream;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_host_get_flags(unsigned int* flags, void* host_pointer) {
  (void)host_pointer;
  if (flags != NULL) {
    *flags = 0;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_host_register(void* pointer, size_t size, unsigned int flags) {
  (void)pointer;
  (void)size;
  (void)flags;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_host_unregister(void* pointer) {
  (void)pointer;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_launch_host_func(cudaStream_t stream, cudaHostFn_t function,
                                                 void* user_data) {
  (void)stream;
  (void)function;
  (void)user_data;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_launch_kernel(const void* function, dim3 grid_dimension,
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

static cudaError_t xtbloom_cuda_malloc(void** pointer, size_t size) {
  (void)size;
  if (pointer != NULL) {
    *pointer = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_malloc_managed(void** pointer, size_t size, unsigned int flags) {
  (void)size;
  (void)flags;
  if (pointer != NULL) {
    *pointer = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_memcpy(void* destination, const void* source, size_t count,
                                       enum cudaMemcpyKind kind) {
  (void)destination;
  (void)source;
  (void)count;
  (void)kind;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_memcpy_async(void* destination, const void* source, size_t count,
                                             enum cudaMemcpyKind kind, cudaStream_t stream) {
  (void)destination;
  (void)source;
  (void)count;
  (void)kind;
  (void)stream;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_memcpy_from_symbol(void* destination, const void* symbol,
                                                   size_t count, size_t offset,
                                                   enum cudaMemcpyKind kind) {
  (void)destination;
  (void)symbol;
  (void)count;
  (void)offset;
  (void)kind;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_memcpy_to_symbol(const void* symbol, const void* source,
                                                 size_t count, size_t offset,
                                                 enum cudaMemcpyKind kind) {
  (void)symbol;
  (void)source;
  (void)count;
  (void)offset;
  (void)kind;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_memset_async(void* pointer, int value, size_t count,
                                             cudaStream_t stream) {
  (void)pointer;
  (void)value;
  (void)count;
  (void)stream;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_memset(void* pointer, int value, size_t count) {
  (void)pointer;
  (void)value;
  (void)count;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_pointer_get_attributes(struct cudaPointerAttributes* attributes,
                                                       const void* pointer) {
  (void)pointer;
  if (attributes != NULL) {
    memset(attributes, 0, sizeof(*attributes));
    attributes->type = cudaMemoryTypeUnregistered;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_runtime_get_version(int* version) {
  if (version != NULL) {
    *version = 0;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_set_device(int device) {
  (void)device;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_stream_begin_capture(cudaStream_t stream,
                                                     enum cudaStreamCaptureMode mode) {
  (void)stream;
  (void)mode;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_stream_begin_capture_to_graph(
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

static cudaError_t xtbloom_cuda_stream_create(cudaStream_t* stream, unsigned int flags) {
  (void)flags;
  if (stream != NULL) {
    *stream = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_stream_create_default(cudaStream_t* stream) {
  if (stream != NULL) {
    *stream = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_stream_destroy(cudaStream_t stream) {
  (void)stream;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_stream_end_capture(cudaStream_t stream, cudaGraph_t* graph) {
  (void)stream;
  if (graph != NULL) {
    *graph = NULL;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_stream_get_capture_info(cudaStream_t stream,
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

static cudaError_t xtbloom_cuda_stream_get_device(cudaStream_t stream, int* device) {
  (void)stream;
  if (device != NULL) {
    *device = 0;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_stream_is_capturing(cudaStream_t stream,
                                                    enum cudaStreamCaptureStatus* status) {
  (void)stream;
  if (status != NULL) {
    *status = cudaStreamCaptureStatusNone;
  }
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_stream_synchronize(cudaStream_t stream) {
  (void)stream;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_stream_update_capture_dependencies(cudaStream_t stream,
                                                                   cudaGraphNode_t* dependencies,
                                                                   size_t dependency_count,
                                                                   unsigned int flags) {
  (void)stream;
  (void)dependencies;
  (void)dependency_count;
  (void)flags;
  return cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_stream_wait_event(cudaStream_t stream, cudaEvent_t event,
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
static void* xtbloom_fatbinary_token_storage;

static void** xtbloom_cuda_register_fat_binary(void* fat_cubin) {
  (void)fat_cubin;
  return &xtbloom_fatbinary_token_storage;
}

static void xtbloom_cuda_register_fat_binary_end(void** fat_cubin_handle) {
  (void)fat_cubin_handle;
}

static void xtbloom_cuda_unregister_fat_binary(void** fat_cubin_handle) { (void)fat_cubin_handle; }

static void xtbloom_cuda_register_function(void** fat_cubin_handle, const char* host_function,
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

static void xtbloom_cuda_register_var(void** fat_cubin_handle, char* host_variable,
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

static char xtbloom_cuda_init_module(void** fat_cubin_handle) {
  (void)fat_cubin_handle;
  return 0;
}

static unsigned int xtbloom_cuda_push_call_configuration(dim3 grid_dimension, dim3 block_dimension,
                                                         size_t shared_memory,
                                                         cudaStream_t stream) {
  (void)grid_dimension;
  (void)block_dimension;
  (void)shared_memory;
  (void)stream;
  return (unsigned int)cudaErrorSharedObjectSymbolNotFound;
}

static cudaError_t xtbloom_cuda_pop_call_configuration(dim3* grid_dimension, dim3* block_dimension,
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
static cublasStatus_t xtbloom_cublas_create(cublasHandle_t* handle) {
  if (handle != NULL) {
    *handle = NULL;
  }
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

static cublasStatus_t xtbloom_cublas_destroy(cublasHandle_t handle) {
  (void)handle;
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

static cublasStatus_t xtbloom_cublas_dtrsm_batched(
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

static cublasStatus_t xtbloom_cublas_get_math_mode(cublasHandle_t handle, cublasMath_t* mode) {
  (void)handle;
  if (mode != NULL) {
    *mode = CUBLAS_DEFAULT_MATH;
  }
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

static cublasStatus_t xtbloom_cublas_get_pointer_mode(cublasHandle_t handle,
                                                      cublasPointerMode_t* mode) {
  (void)handle;
  if (mode != NULL) {
    *mode = CUBLAS_POINTER_MODE_HOST;
  }
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

static cublasStatus_t xtbloom_cublas_get_stream(cublasHandle_t handle, cudaStream_t* stream) {
  (void)handle;
  if (stream != NULL) {
    *stream = NULL;
  }
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

static cublasStatus_t xtbloom_cublas_get_version(cublasHandle_t handle, int* version) {
  (void)handle;
  if (version != NULL) {
    *version = 0;
  }
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

static cublasStatus_t xtbloom_cublas_set_math_mode(cublasHandle_t handle, cublasMath_t mode) {
  (void)handle;
  (void)mode;
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

static cublasStatus_t xtbloom_cublas_set_pointer_mode(cublasHandle_t handle,
                                                      cublasPointerMode_t mode) {
  (void)handle;
  (void)mode;
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

static cublasStatus_t xtbloom_cublas_set_stream(cublasHandle_t handle, cudaStream_t stream) {
  (void)handle;
  (void)stream;
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

static cublasStatus_t xtbloom_cublas_set_workspace(cublasHandle_t handle, void* workspace,
                                                   size_t workspace_size) {
  (void)handle;
  (void)workspace;
  (void)workspace_size;
  return CUBLAS_STATUS_NOT_INITIALIZED;
}

/* cuSOLVER fallbacks. Device-resident info arrays are not dereferenced. */
static cusolverStatus_t xtbloom_cusolver_create(cusolverDnHandle_t* handle) {
  if (handle != NULL) {
    *handle = NULL;
  }
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t xtbloom_cusolver_create_params(cusolverDnParams_t* parameters) {
  if (parameters != NULL) {
    *parameters = NULL;
  }
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t xtbloom_cusolver_create_syevj_info(syevjInfo_t* information) {
  if (information != NULL) {
    *information = NULL;
  }
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t xtbloom_cusolver_destroy(cusolverDnHandle_t handle) {
  (void)handle;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t xtbloom_cusolver_destroy_params(cusolverDnParams_t parameters) {
  (void)parameters;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t xtbloom_cusolver_destroy_syevj_info(syevjInfo_t information) {
  (void)information;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t xtbloom_cusolver_dormtr_buffer_size(
    cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t fill,
    cublasOperation_t operation, int rows, int columns, const double* reflectors,
    int reflector_leading_dimension, const double* tau, const double* matrix,
    int matrix_leading_dimension, int* workspace_size) {
  (void)handle;
  (void)side;
  (void)fill;
  (void)operation;
  (void)rows;
  (void)columns;
  (void)reflectors;
  (void)reflector_leading_dimension;
  (void)tau;
  (void)matrix;
  (void)matrix_leading_dimension;
  if (workspace_size != NULL) *workspace_size = 0;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t xtbloom_cusolver_dormtr(cusolverDnHandle_t handle, cublasSideMode_t side,
                                                cublasFillMode_t fill, cublasOperation_t operation,
                                                int rows, int columns, double* reflectors,
                                                int reflector_leading_dimension, double* tau,
                                                double* matrix, int matrix_leading_dimension,
                                                double* workspace, int workspace_size,
                                                int* information) {
  (void)handle;
  (void)side;
  (void)fill;
  (void)operation;
  (void)rows;
  (void)columns;
  (void)reflectors;
  (void)reflector_leading_dimension;
  (void)tau;
  (void)matrix;
  (void)matrix_leading_dimension;
  (void)workspace;
  (void)workspace_size;
  (void)information;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t xtbloom_cusolver_dpotrf_batched(cusolverDnHandle_t handle,
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

static cusolverStatus_t xtbloom_cusolver_dsytrd_buffer_size(
    cusolverDnHandle_t handle, cublasFillMode_t fill, int dimension, const double* matrix,
    int leading_dimension, const double* diagonal, const double* off_diagonal, const double* tau,
    int* workspace_size) {
  (void)handle;
  (void)fill;
  (void)dimension;
  (void)matrix;
  (void)leading_dimension;
  (void)diagonal;
  (void)off_diagonal;
  (void)tau;
  if (workspace_size != NULL) *workspace_size = 0;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t xtbloom_cusolver_dsytrd(cusolverDnHandle_t handle, cublasFillMode_t fill,
                                                int dimension, double* matrix,
                                                int leading_dimension, double* diagonal,
                                                double* off_diagonal, double* tau,
                                                double* workspace, int workspace_size,
                                                int* information) {
  (void)handle;
  (void)fill;
  (void)dimension;
  (void)matrix;
  (void)leading_dimension;
  (void)diagonal;
  (void)off_diagonal;
  (void)tau;
  (void)workspace;
  (void)workspace_size;
  (void)information;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t xtbloom_cusolver_dsyevj_batched_buffer_size(
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

static cusolverStatus_t xtbloom_cusolver_dsyevj_batched(
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

static cusolverStatus_t xtbloom_cusolver_get_stream(cusolverDnHandle_t handle,
                                                    cudaStream_t* stream) {
  (void)handle;
  if (stream != NULL) {
    *stream = NULL;
  }
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t xtbloom_cusolver_set_stream(cusolverDnHandle_t handle,
                                                    cudaStream_t stream) {
  (void)handle;
  (void)stream;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t xtbloom_cusolver_syevj_set_max_sweeps(syevjInfo_t information,
                                                              int max_sweeps) {
  (void)information;
  (void)max_sweeps;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t xtbloom_cusolver_syevj_set_sort_eig(syevjInfo_t information,
                                                            int sort_eigenvalues) {
  (void)information;
  (void)sort_eigenvalues;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t xtbloom_cusolver_syevj_set_tolerance(syevjInfo_t information,
                                                             double tolerance) {
  (void)information;
  (void)tolerance;
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

static cusolverStatus_t xtbloom_cusolver_xsyev_batched(
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

static cusolverStatus_t xtbloom_cusolver_xsyev_batched_buffer_size(
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

static cusolverStatus_t xtbloom_cusolver_get_property(libraryPropertyType property, int* value) {
  (void)property;
  if (value != NULL) {
    *value = 0;
  }
  return CUSOLVER_STATUS_NOT_INITIALIZED;
}

/* Driver fallbacks. Driver queries return CUDA_ERROR_NOT_INITIALIZED, which is
 * the driver API's documented initialization failure rather than cudart 302. */
static CUresult xtbloom_cu_get_error_text(CUresult error, const char** text) {
  (void)error;
  if (text != NULL) {
    *text = xtbloom_cuda_unavailable_text;
  }
  return CUDA_ERROR_NOT_INITIALIZED;
}

static CUresult xtbloom_cu_mem_get_address_range(CUdeviceptr* base, size_t* size,
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
#define XTBLOOM_CHECK_SIGNATURE(api, fallback)                                              \
  _Static_assert(__builtin_types_compatible_p(__typeof__(&(api)), __typeof__(&(fallback))), \
                 "CUDA fallback signature does not match " #api)

XTBLOOM_CHECK_SIGNATURE(cudaDeviceGetAttribute, xtbloom_cuda_device_get_attribute);
XTBLOOM_CHECK_SIGNATURE(cudaDeviceSynchronize, xtbloom_cudart_error);
XTBLOOM_CHECK_SIGNATURE(cudaEventCreate, xtbloom_cuda_event_create_default);
XTBLOOM_CHECK_SIGNATURE(cudaEventCreateWithFlags, xtbloom_cuda_event_create);
XTBLOOM_CHECK_SIGNATURE(cudaEventDestroy, xtbloom_cuda_event_destroy);
XTBLOOM_CHECK_SIGNATURE(cudaEventElapsedTime, xtbloom_cuda_event_elapsed_time);
XTBLOOM_CHECK_SIGNATURE(cudaEventRecord, xtbloom_cuda_event_record);
XTBLOOM_CHECK_SIGNATURE(cudaEventSynchronize, xtbloom_cuda_event_synchronize);
XTBLOOM_CHECK_SIGNATURE(cudaFree, xtbloom_cuda_free);
XTBLOOM_CHECK_SIGNATURE(cudaFreeHost, xtbloom_cuda_free);
XTBLOOM_CHECK_SIGNATURE(cudaGetDevice, xtbloom_cuda_get_device);
XTBLOOM_CHECK_SIGNATURE(cudaGetDeviceCount, xtbloom_cuda_get_device_count);
XTBLOOM_CHECK_SIGNATURE(cudaGetErrorName, xtbloom_cuda_get_error_text);
XTBLOOM_CHECK_SIGNATURE(cudaGetErrorString, xtbloom_cuda_get_error_text);
XTBLOOM_CHECK_SIGNATURE(cudaGetLastError, xtbloom_cudart_error);
XTBLOOM_CHECK_SIGNATURE(cudaGraphAddNode, xtbloom_cuda_graph_add_node);
XTBLOOM_CHECK_SIGNATURE(cudaGraphChildGraphNodeGetGraph, xtbloom_cuda_graph_child_get_graph);
XTBLOOM_CHECK_SIGNATURE(cudaGraphConditionalHandleCreate,
                        xtbloom_cuda_graph_conditional_handle_create);
XTBLOOM_CHECK_SIGNATURE(cudaGraphCreate, xtbloom_cuda_graph_create);
XTBLOOM_CHECK_SIGNATURE(cudaGraphDestroy, xtbloom_cuda_graph_destroy);
XTBLOOM_CHECK_SIGNATURE(cudaGraphExecDestroy, xtbloom_cuda_graph_exec_destroy);
XTBLOOM_CHECK_SIGNATURE(cudaGraphGetNodes, xtbloom_cuda_graph_get_nodes);
XTBLOOM_CHECK_SIGNATURE(cudaGraphInstantiate, xtbloom_cuda_graph_instantiate);
XTBLOOM_CHECK_SIGNATURE(cudaGraphLaunch, xtbloom_cuda_graph_launch);
XTBLOOM_CHECK_SIGNATURE(cudaGraphMemcpyNodeGetParams, xtbloom_cuda_graph_memcpy_node_get_params);
XTBLOOM_CHECK_SIGNATURE(cudaGraphNodeGetType, xtbloom_cuda_graph_node_get_type);
XTBLOOM_CHECK_SIGNATURE(cudaGraphUpload, xtbloom_cuda_graph_upload);
XTBLOOM_CHECK_SIGNATURE(cudaHostGetFlags, xtbloom_cuda_host_get_flags);
XTBLOOM_CHECK_SIGNATURE(cudaHostRegister, xtbloom_cuda_host_register);
XTBLOOM_CHECK_SIGNATURE(cudaHostUnregister, xtbloom_cuda_host_unregister);
XTBLOOM_CHECK_SIGNATURE(cudaLaunchHostFunc, xtbloom_cuda_launch_host_func);
XTBLOOM_CHECK_SIGNATURE(cudaLaunchKernel, xtbloom_cuda_launch_kernel);
XTBLOOM_CHECK_SIGNATURE(cudaMalloc, xtbloom_cuda_malloc);
XTBLOOM_CHECK_SIGNATURE(cudaMallocHost, xtbloom_cuda_malloc);
XTBLOOM_CHECK_SIGNATURE(cudaMallocManaged, xtbloom_cuda_malloc_managed);
XTBLOOM_CHECK_SIGNATURE(cudaMemcpy, xtbloom_cuda_memcpy);
XTBLOOM_CHECK_SIGNATURE(cudaMemcpyAsync, xtbloom_cuda_memcpy_async);
XTBLOOM_CHECK_SIGNATURE(cudaMemcpyFromSymbol, xtbloom_cuda_memcpy_from_symbol);
XTBLOOM_CHECK_SIGNATURE(cudaMemcpyToSymbol, xtbloom_cuda_memcpy_to_symbol);
XTBLOOM_CHECK_SIGNATURE(cudaMemset, xtbloom_cuda_memset);
XTBLOOM_CHECK_SIGNATURE(cudaMemsetAsync, xtbloom_cuda_memset_async);
XTBLOOM_CHECK_SIGNATURE(cudaPeekAtLastError, xtbloom_cudart_error);
XTBLOOM_CHECK_SIGNATURE(cudaPointerGetAttributes, xtbloom_cuda_pointer_get_attributes);
XTBLOOM_CHECK_SIGNATURE(cudaRuntimeGetVersion, xtbloom_cuda_runtime_get_version);
XTBLOOM_CHECK_SIGNATURE(cudaSetDevice, xtbloom_cuda_set_device);
XTBLOOM_CHECK_SIGNATURE(cudaStreamBeginCapture, xtbloom_cuda_stream_begin_capture);
XTBLOOM_CHECK_SIGNATURE(cudaStreamBeginCaptureToGraph, xtbloom_cuda_stream_begin_capture_to_graph);
XTBLOOM_CHECK_SIGNATURE(cudaStreamCreate, xtbloom_cuda_stream_create_default);
XTBLOOM_CHECK_SIGNATURE(cudaStreamCreateWithFlags, xtbloom_cuda_stream_create);
XTBLOOM_CHECK_SIGNATURE(cudaStreamDestroy, xtbloom_cuda_stream_destroy);
XTBLOOM_CHECK_SIGNATURE(cudaStreamEndCapture, xtbloom_cuda_stream_end_capture);
XTBLOOM_CHECK_SIGNATURE(cudaStreamGetCaptureInfo_v2, xtbloom_cuda_stream_get_capture_info);
XTBLOOM_CHECK_SIGNATURE(cudaStreamGetDevice, xtbloom_cuda_stream_get_device);
XTBLOOM_CHECK_SIGNATURE(cudaStreamIsCapturing, xtbloom_cuda_stream_is_capturing);
XTBLOOM_CHECK_SIGNATURE(cudaStreamSynchronize, xtbloom_cuda_stream_synchronize);
XTBLOOM_CHECK_SIGNATURE(cudaStreamUpdateCaptureDependencies,
                        xtbloom_cuda_stream_update_capture_dependencies);
XTBLOOM_CHECK_SIGNATURE(cudaStreamWaitEvent, xtbloom_cuda_stream_wait_event);

XTBLOOM_CHECK_SIGNATURE(cublasCreate_v2, xtbloom_cublas_create);
XTBLOOM_CHECK_SIGNATURE(cublasDestroy_v2, xtbloom_cublas_destroy);
XTBLOOM_CHECK_SIGNATURE(cublasDtrsmBatched, xtbloom_cublas_dtrsm_batched);
XTBLOOM_CHECK_SIGNATURE(cublasGetMathMode, xtbloom_cublas_get_math_mode);
XTBLOOM_CHECK_SIGNATURE(cublasGetPointerMode_v2, xtbloom_cublas_get_pointer_mode);
XTBLOOM_CHECK_SIGNATURE(cublasGetStream_v2, xtbloom_cublas_get_stream);
XTBLOOM_CHECK_SIGNATURE(cublasGetVersion_v2, xtbloom_cublas_get_version);
XTBLOOM_CHECK_SIGNATURE(cublasSetMathMode, xtbloom_cublas_set_math_mode);
XTBLOOM_CHECK_SIGNATURE(cublasSetPointerMode_v2, xtbloom_cublas_set_pointer_mode);
XTBLOOM_CHECK_SIGNATURE(cublasSetStream_v2, xtbloom_cublas_set_stream);
XTBLOOM_CHECK_SIGNATURE(cublasSetWorkspace_v2, xtbloom_cublas_set_workspace);

XTBLOOM_CHECK_SIGNATURE(cusolverDnCreate, xtbloom_cusolver_create);
XTBLOOM_CHECK_SIGNATURE(cusolverDnCreateParams, xtbloom_cusolver_create_params);
XTBLOOM_CHECK_SIGNATURE(cusolverDnCreateSyevjInfo, xtbloom_cusolver_create_syevj_info);
XTBLOOM_CHECK_SIGNATURE(cusolverDnDestroy, xtbloom_cusolver_destroy);
XTBLOOM_CHECK_SIGNATURE(cusolverDnDestroyParams, xtbloom_cusolver_destroy_params);
XTBLOOM_CHECK_SIGNATURE(cusolverDnDestroySyevjInfo, xtbloom_cusolver_destroy_syevj_info);
XTBLOOM_CHECK_SIGNATURE(cusolverDnDormtr, xtbloom_cusolver_dormtr);
XTBLOOM_CHECK_SIGNATURE(cusolverDnDormtr_bufferSize, xtbloom_cusolver_dormtr_buffer_size);
XTBLOOM_CHECK_SIGNATURE(cusolverDnDpotrfBatched, xtbloom_cusolver_dpotrf_batched);
XTBLOOM_CHECK_SIGNATURE(cusolverDnDsytrd, xtbloom_cusolver_dsytrd);
XTBLOOM_CHECK_SIGNATURE(cusolverDnDsytrd_bufferSize, xtbloom_cusolver_dsytrd_buffer_size);
XTBLOOM_CHECK_SIGNATURE(cusolverDnDsyevjBatched, xtbloom_cusolver_dsyevj_batched);
XTBLOOM_CHECK_SIGNATURE(cusolverDnDsyevjBatched_bufferSize,
                        xtbloom_cusolver_dsyevj_batched_buffer_size);
XTBLOOM_CHECK_SIGNATURE(cusolverDnGetStream, xtbloom_cusolver_get_stream);
XTBLOOM_CHECK_SIGNATURE(cusolverDnSetStream, xtbloom_cusolver_set_stream);
XTBLOOM_CHECK_SIGNATURE(cusolverDnXsyevjSetMaxSweeps, xtbloom_cusolver_syevj_set_max_sweeps);
XTBLOOM_CHECK_SIGNATURE(cusolverDnXsyevjSetSortEig, xtbloom_cusolver_syevj_set_sort_eig);
XTBLOOM_CHECK_SIGNATURE(cusolverDnXsyevjSetTolerance, xtbloom_cusolver_syevj_set_tolerance);
XTBLOOM_CHECK_SIGNATURE(cusolverDnXsyevBatched, xtbloom_cusolver_xsyev_batched);
XTBLOOM_CHECK_SIGNATURE(cusolverDnXsyevBatched_bufferSize,
                        xtbloom_cusolver_xsyev_batched_buffer_size);
XTBLOOM_CHECK_SIGNATURE(cusolverGetProperty, xtbloom_cusolver_get_property);

XTBLOOM_CHECK_SIGNATURE(cuGetErrorString, xtbloom_cu_get_error_text);
XTBLOOM_CHECK_SIGNATURE(cuMemGetAddressRange_v2, xtbloom_cu_mem_get_address_range);

typedef void** (*xtbloom_register_fat_binary_signature)(void*);
typedef void (*xtbloom_register_fat_binary_end_signature)(void**);
typedef void (*xtbloom_unregister_fat_binary_signature)(void**);
typedef void (*xtbloom_register_function_signature)(void**, const char*, char*, const char*, int,
                                                    uint3*, uint3*, dim3*, dim3*, int*);
typedef void (*xtbloom_register_var_signature)(void**, char*, char*, const char*, int, size_t, int,
                                               int);
typedef char (*xtbloom_init_module_signature)(void**);
typedef unsigned int (*xtbloom_push_call_configuration_signature)(dim3, dim3, size_t, cudaStream_t);
typedef cudaError_t (*xtbloom_pop_call_configuration_signature)(dim3*, dim3*, size_t*, void*);

_Static_assert(__builtin_types_compatible_p(xtbloom_register_fat_binary_signature,
                                            __typeof__(&xtbloom_cuda_register_fat_binary)),
               "__cudaRegisterFatBinary fallback signature mismatch");
_Static_assert(__builtin_types_compatible_p(xtbloom_register_fat_binary_end_signature,
                                            __typeof__(&xtbloom_cuda_register_fat_binary_end)),
               "__cudaRegisterFatBinaryEnd fallback signature mismatch");
_Static_assert(__builtin_types_compatible_p(xtbloom_unregister_fat_binary_signature,
                                            __typeof__(&xtbloom_cuda_unregister_fat_binary)),
               "__cudaUnregisterFatBinary fallback signature mismatch");
_Static_assert(__builtin_types_compatible_p(xtbloom_register_function_signature,
                                            __typeof__(&xtbloom_cuda_register_function)),
               "__cudaRegisterFunction fallback signature mismatch");
_Static_assert(__builtin_types_compatible_p(xtbloom_register_var_signature,
                                            __typeof__(&xtbloom_cuda_register_var)),
               "__cudaRegisterVar fallback signature mismatch");
_Static_assert(__builtin_types_compatible_p(xtbloom_init_module_signature,
                                            __typeof__(&xtbloom_cuda_init_module)),
               "__cudaInitModule fallback signature mismatch");
_Static_assert(__builtin_types_compatible_p(xtbloom_push_call_configuration_signature,
                                            __typeof__(&xtbloom_cuda_push_call_configuration)),
               "__cudaPushCallConfiguration fallback signature mismatch");
_Static_assert(__builtin_types_compatible_p(xtbloom_pop_call_configuration_signature,
                                            __typeof__(&xtbloom_cuda_pop_call_configuration)),
               "__cudaPopCallConfiguration fallback signature mismatch");

#undef XTBLOOM_CHECK_SIGNATURE
#endif

static void* xtbloom_fallback_for_symbol(const char* symbol) {
#define XTBLOOM_SYMBOL(name, fallback)      \
  if (strcmp(symbol, name) == 0) {          \
    union {                                 \
      __typeof__(&fallback) function;       \
      void* object;                         \
    } conversion = {.function = &fallback}; \
    return conversion.object;               \
  }

  XTBLOOM_SYMBOL("cudaEventCreateWithFlags", xtbloom_cuda_event_create)
  XTBLOOM_SYMBOL("cudaEventCreate", xtbloom_cuda_event_create_default)
  XTBLOOM_SYMBOL("cudaEventDestroy", xtbloom_cuda_event_destroy)
  XTBLOOM_SYMBOL("cudaEventElapsedTime", xtbloom_cuda_event_elapsed_time)
  XTBLOOM_SYMBOL("cudaEventRecord", xtbloom_cuda_event_record)
  XTBLOOM_SYMBOL("cudaEventSynchronize", xtbloom_cuda_event_synchronize)
  XTBLOOM_SYMBOL("cudaFree", xtbloom_cuda_free)
  XTBLOOM_SYMBOL("cudaFreeHost", xtbloom_cuda_free)
  XTBLOOM_SYMBOL("cudaGetDevice", xtbloom_cuda_get_device)
  XTBLOOM_SYMBOL("cudaGetDeviceCount", xtbloom_cuda_get_device_count)
  XTBLOOM_SYMBOL("cudaDeviceGetAttribute", xtbloom_cuda_device_get_attribute)
  XTBLOOM_SYMBOL("cudaDeviceSynchronize", xtbloom_cudart_error)
  XTBLOOM_SYMBOL("cudaGetErrorName", xtbloom_cuda_get_error_text)
  XTBLOOM_SYMBOL("cudaGetErrorString", xtbloom_cuda_get_error_text)
  XTBLOOM_SYMBOL("cudaGetLastError", xtbloom_cudart_error)
  XTBLOOM_SYMBOL("cudaGraphAddNode", xtbloom_cuda_graph_add_node)
  XTBLOOM_SYMBOL("cudaGraphChildGraphNodeGetGraph", xtbloom_cuda_graph_child_get_graph)
  XTBLOOM_SYMBOL("cudaGraphConditionalHandleCreate", xtbloom_cuda_graph_conditional_handle_create)
  XTBLOOM_SYMBOL("cudaGraphCreate", xtbloom_cuda_graph_create)
  XTBLOOM_SYMBOL("cudaGraphDestroy", xtbloom_cuda_graph_destroy)
  XTBLOOM_SYMBOL("cudaGraphExecDestroy", xtbloom_cuda_graph_exec_destroy)
  XTBLOOM_SYMBOL("cudaGraphGetNodes", xtbloom_cuda_graph_get_nodes)
  XTBLOOM_SYMBOL("cudaGraphMemcpyNodeGetParams", xtbloom_cuda_graph_memcpy_node_get_params)
  XTBLOOM_SYMBOL("cudaGraphInstantiate", xtbloom_cuda_graph_instantiate)
  XTBLOOM_SYMBOL("cudaGraphLaunch", xtbloom_cuda_graph_launch)
  XTBLOOM_SYMBOL("cudaGraphNodeGetType", xtbloom_cuda_graph_node_get_type)
  XTBLOOM_SYMBOL("cudaGraphUpload", xtbloom_cuda_graph_upload)
  XTBLOOM_SYMBOL("cudaHostGetFlags", xtbloom_cuda_host_get_flags)
  XTBLOOM_SYMBOL("cudaHostRegister", xtbloom_cuda_host_register)
  XTBLOOM_SYMBOL("cudaHostUnregister", xtbloom_cuda_host_unregister)
  XTBLOOM_SYMBOL("cudaLaunchHostFunc", xtbloom_cuda_launch_host_func)
  XTBLOOM_SYMBOL("cudaLaunchKernel", xtbloom_cuda_launch_kernel)
  XTBLOOM_SYMBOL("cudaMalloc", xtbloom_cuda_malloc)
  XTBLOOM_SYMBOL("cudaMallocHost", xtbloom_cuda_malloc)
  XTBLOOM_SYMBOL("cudaMallocManaged", xtbloom_cuda_malloc_managed)
  XTBLOOM_SYMBOL("cudaMemcpy", xtbloom_cuda_memcpy)
  XTBLOOM_SYMBOL("cudaMemcpyAsync", xtbloom_cuda_memcpy_async)
  XTBLOOM_SYMBOL("cudaMemcpyFromSymbol", xtbloom_cuda_memcpy_from_symbol)
  XTBLOOM_SYMBOL("cudaMemcpyToSymbol", xtbloom_cuda_memcpy_to_symbol)
  XTBLOOM_SYMBOL("cudaMemset", xtbloom_cuda_memset)
  XTBLOOM_SYMBOL("cudaMemsetAsync", xtbloom_cuda_memset_async)
  XTBLOOM_SYMBOL("cudaPeekAtLastError", xtbloom_cudart_error)
  XTBLOOM_SYMBOL("cudaPointerGetAttributes", xtbloom_cuda_pointer_get_attributes)
  XTBLOOM_SYMBOL("cudaRuntimeGetVersion", xtbloom_cuda_runtime_get_version)
  XTBLOOM_SYMBOL("cudaSetDevice", xtbloom_cuda_set_device)
  XTBLOOM_SYMBOL("cudaStreamBeginCapture", xtbloom_cuda_stream_begin_capture)
  XTBLOOM_SYMBOL("cudaStreamBeginCaptureToGraph", xtbloom_cuda_stream_begin_capture_to_graph)
  XTBLOOM_SYMBOL("cudaStreamCreate", xtbloom_cuda_stream_create_default)
  XTBLOOM_SYMBOL("cudaStreamCreateWithFlags", xtbloom_cuda_stream_create)
  XTBLOOM_SYMBOL("cudaStreamDestroy", xtbloom_cuda_stream_destroy)
  XTBLOOM_SYMBOL("cudaStreamEndCapture", xtbloom_cuda_stream_end_capture)
  XTBLOOM_SYMBOL("cudaStreamGetCaptureInfo_v2", xtbloom_cuda_stream_get_capture_info)
  XTBLOOM_SYMBOL("cudaStreamGetDevice", xtbloom_cuda_stream_get_device)
  XTBLOOM_SYMBOL("cudaStreamIsCapturing", xtbloom_cuda_stream_is_capturing)
  XTBLOOM_SYMBOL("cudaStreamSynchronize", xtbloom_cuda_stream_synchronize)
  XTBLOOM_SYMBOL("cudaStreamUpdateCaptureDependencies",
                 xtbloom_cuda_stream_update_capture_dependencies)
  XTBLOOM_SYMBOL("cudaStreamWaitEvent", xtbloom_cuda_stream_wait_event)
  XTBLOOM_SYMBOL("__cudaInitModule", xtbloom_cuda_init_module)
  XTBLOOM_SYMBOL("__cudaPopCallConfiguration", xtbloom_cuda_pop_call_configuration)
  XTBLOOM_SYMBOL("__cudaPushCallConfiguration", xtbloom_cuda_push_call_configuration)
  XTBLOOM_SYMBOL("__cudaRegisterFatBinary", xtbloom_cuda_register_fat_binary)
  XTBLOOM_SYMBOL("__cudaRegisterFatBinaryEnd", xtbloom_cuda_register_fat_binary_end)
  XTBLOOM_SYMBOL("__cudaRegisterFunction", xtbloom_cuda_register_function)
  XTBLOOM_SYMBOL("__cudaRegisterVar", xtbloom_cuda_register_var)
  XTBLOOM_SYMBOL("__cudaUnregisterFatBinary", xtbloom_cuda_unregister_fat_binary)

  XTBLOOM_SYMBOL("cublasCreate_v2", xtbloom_cublas_create)
  XTBLOOM_SYMBOL("cublasDestroy_v2", xtbloom_cublas_destroy)
  XTBLOOM_SYMBOL("cublasDtrsmBatched", xtbloom_cublas_dtrsm_batched)
  XTBLOOM_SYMBOL("cublasGetMathMode", xtbloom_cublas_get_math_mode)
  XTBLOOM_SYMBOL("cublasGetPointerMode_v2", xtbloom_cublas_get_pointer_mode)
  XTBLOOM_SYMBOL("cublasGetStream_v2", xtbloom_cublas_get_stream)
  XTBLOOM_SYMBOL("cublasGetVersion_v2", xtbloom_cublas_get_version)
  XTBLOOM_SYMBOL("cublasSetMathMode", xtbloom_cublas_set_math_mode)
  XTBLOOM_SYMBOL("cublasSetPointerMode_v2", xtbloom_cublas_set_pointer_mode)
  XTBLOOM_SYMBOL("cublasSetStream_v2", xtbloom_cublas_set_stream)
  XTBLOOM_SYMBOL("cublasSetWorkspace_v2", xtbloom_cublas_set_workspace)

  XTBLOOM_SYMBOL("cusolverDnCreate", xtbloom_cusolver_create)
  XTBLOOM_SYMBOL("cusolverDnCreateParams", xtbloom_cusolver_create_params)
  XTBLOOM_SYMBOL("cusolverDnCreateSyevjInfo", xtbloom_cusolver_create_syevj_info)
  XTBLOOM_SYMBOL("cusolverDnDestroy", xtbloom_cusolver_destroy)
  XTBLOOM_SYMBOL("cusolverDnDestroyParams", xtbloom_cusolver_destroy_params)
  XTBLOOM_SYMBOL("cusolverDnDestroySyevjInfo", xtbloom_cusolver_destroy_syevj_info)
  XTBLOOM_SYMBOL("cusolverDnDormtr", xtbloom_cusolver_dormtr)
  XTBLOOM_SYMBOL("cusolverDnDormtr_bufferSize", xtbloom_cusolver_dormtr_buffer_size)
  XTBLOOM_SYMBOL("cusolverDnDpotrfBatched", xtbloom_cusolver_dpotrf_batched)
  XTBLOOM_SYMBOL("cusolverDnDsytrd", xtbloom_cusolver_dsytrd)
  XTBLOOM_SYMBOL("cusolverDnDsytrd_bufferSize", xtbloom_cusolver_dsytrd_buffer_size)
  XTBLOOM_SYMBOL("cusolverDnDsyevjBatched", xtbloom_cusolver_dsyevj_batched)
  XTBLOOM_SYMBOL("cusolverDnDsyevjBatched_bufferSize", xtbloom_cusolver_dsyevj_batched_buffer_size)
  XTBLOOM_SYMBOL("cusolverDnGetStream", xtbloom_cusolver_get_stream)
  XTBLOOM_SYMBOL("cusolverDnSetStream", xtbloom_cusolver_set_stream)
  XTBLOOM_SYMBOL("cusolverDnXsyevjSetMaxSweeps", xtbloom_cusolver_syevj_set_max_sweeps)
  XTBLOOM_SYMBOL("cusolverDnXsyevjSetSortEig", xtbloom_cusolver_syevj_set_sort_eig)
  XTBLOOM_SYMBOL("cusolverDnXsyevjSetTolerance", xtbloom_cusolver_syevj_set_tolerance)
  XTBLOOM_SYMBOL("cusolverDnXsyevBatched", xtbloom_cusolver_xsyev_batched)
  XTBLOOM_SYMBOL("cusolverDnXsyevBatched_bufferSize", xtbloom_cusolver_xsyev_batched_buffer_size)
  XTBLOOM_SYMBOL("cusolverGetProperty", xtbloom_cusolver_get_property)

  XTBLOOM_SYMBOL("cuGetErrorString", xtbloom_cu_get_error_text)
  XTBLOOM_SYMBOL("cuMemGetAddressRange_v2", xtbloom_cu_mem_get_address_range)

#undef XTBLOOM_SYMBOL
  return NULL;
}

/* Generated implib initializers call this only after constructor(101) has
 * selected either a fully preflighted real handle or the missing-cohort
 * sentinel. Unknown symbols stay NULL: guessing a generic C signature would
 * turn a clean backend-unavailable error into undefined behavior. */
XTBLOOM_HIDDEN void* xtbloom_cuda_dlsym(void* handle, const char* symbol) {
  if (symbol == NULL || *symbol == '\0') {
    xtbloom_report_once(XTBLOOM_DIAGNOSTIC_UNKNOWN_SYMBOL, "<invalid>", symbol, NULL);
    return NULL;
  }

  if (handle != XTBLOOM_MISSING_COHORT) {
    void* resolved = dlsym(handle, symbol);
    if (resolved != NULL) {
      return resolved;
    }
    /* Preflight makes this path an internal consistency failure. Do not hide
     * it behind a fallback for only one member of an otherwise real cohort. */
    xtbloom_report_once(XTBLOOM_DIAGNOSTIC_UNKNOWN_SYMBOL, "preflighted CUDA library", symbol,
                        NULL);
    return NULL;
  }

  void* fallback = xtbloom_fallback_for_symbol(symbol);
  if (fallback == NULL) {
    xtbloom_report_once(XTBLOOM_DIAGNOSTIC_UNKNOWN_SYMBOL, "missing CUDA library", symbol, NULL);
  }
  return fallback;
}

typedef void (*xtbloom_set_handle_fn)(void* handle);
typedef void (*xtbloom_resolve_all_fn)(void);

static void xtbloom_prepare_cohort(const char* soname, const char* const* required_symbols,
                                   xtbloom_set_handle_fn set_handle,
                                   xtbloom_resolve_all_fn resolve_all) {
  const char* force_fallback = NULL;
#if defined(XTBLOOM_CUDA_TEST_HOOKS)
  force_fallback = getenv("XTBLOOM_CUDA_FORCE_FALLBACK");
#endif
  void* handle = NULL;
  if (force_fallback == NULL || strcmp(force_fallback, "1") != 0) {
    dlerror();
    handle = dlopen(soname, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
  }
  if (handle == NULL) {
    const char* error = dlerror();
    xtbloom_report_once(XTBLOOM_DIAGNOSTIC_LOAD, soname, NULL,
                        force_fallback != NULL && strcmp(force_fallback, "1") == 0
                            ? "forced by XTBLOOM_CUDA_FORCE_FALLBACK=1"
                            : error);
    handle = XTBLOOM_MISSING_COHORT;
  } else {
    for (size_t index = 0; required_symbols[index] != NULL; ++index) {
      dlerror();
      if (dlsym(handle, required_symbols[index]) == NULL) {
        const char* error = dlerror();
        char detail[512];
        snprintf(detail, sizeof(detail), "missing required symbol %s%s%s", required_symbols[index],
                 error != NULL ? ": " : "", error != NULL ? error : "");
        xtbloom_report_once(XTBLOOM_DIAGNOSTIC_LOAD, soname, NULL, detail);
        handle = XTBLOOM_MISSING_COHORT;
        break;
      }
    }
  }

  set_handle(handle);
  resolve_all();
}

static pthread_once_t xtbloom_cuda_bootstrap_once = PTHREAD_ONCE_INIT;

static void xtbloom_cuda_bootstrap_impl(void) {
  /* cudart must be complete first: NVCC's later registration constructors call
   * its __cuda* hooks. The math and driver cohorts are populated in the same
   * early constructor so no ordinary CUDA call can observe a partial table. */
  xtbloom_prepare_cohort(XTBLOOM_CUDART_SONAME, xtbloom_cudart_required_symbols,
                         _libcudart_so_tramp_set_handle, _libcudart_so_tramp_resolve_all);
  xtbloom_prepare_cohort(XTBLOOM_CUBLAS_SONAME, xtbloom_cublas_required_symbols,
                         _libcublas_so_tramp_set_handle, _libcublas_so_tramp_resolve_all);
  xtbloom_prepare_cohort(XTBLOOM_CUSOLVER_SONAME, xtbloom_cusolver_required_symbols,
                         _libcusolver_so_tramp_set_handle, _libcusolver_so_tramp_resolve_all);
  xtbloom_prepare_cohort(XTBLOOM_CUDA_DRIVER_SONAME, xtbloom_cuda_driver_required_symbols,
                         _libcuda_so_tramp_set_handle, _libcuda_so_tramp_resolve_all);
}

static void __attribute__((constructor(101))) xtbloom_cuda_bootstrap(void) {
  (void)pthread_once(&xtbloom_cuda_bootstrap_once, xtbloom_cuda_bootstrap_impl);
}
