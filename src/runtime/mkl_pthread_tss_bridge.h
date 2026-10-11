// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#ifndef XTBLOOM_RUNTIME_MKL_PTHREAD_TSS_BRIDGE_H
#define XTBLOOM_RUNTIME_MKL_PTHREAD_TSS_BRIDGE_H

#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*xtbloom_mkl_pthread_key_create_fn)(pthread_key_t*, void (*)(void*));
typedef int (*xtbloom_mkl_pthread_key_delete_fn)(pthread_key_t);
typedef void* (*xtbloom_mkl_pthread_getspecific_fn)(pthread_key_t);
typedef int (*xtbloom_mkl_pthread_setspecific_fn)(pthread_key_t, const void*);

/* Base-namespace pthread entry points injected before any provider dependency
 * is added to the private link-map. The bridge DSO has no DT_NEEDED entries,
 * so initializing this table cannot re-enter a namespace-local libc/libdl. */
typedef struct xtbloom_mkl_pthread_tss_api {
  xtbloom_mkl_pthread_key_create_fn key_create;
  xtbloom_mkl_pthread_key_delete_fn key_delete;
  xtbloom_mkl_pthread_getspecific_fn getspecific;
  xtbloom_mkl_pthread_setspecific_fn setspecific;
} xtbloom_mkl_pthread_tss_api;

#define XTBLOOM_MKL_PTHREAD_TSS_BRIDGE_INITIALIZE_SYMBOL "xtbloom_mkl_pthread_tss_bridge_initialize"

typedef int (*xtbloom_mkl_pthread_tss_bridge_initialize_fn)(const xtbloom_mkl_pthread_tss_api*);

#ifdef __cplusplus
}
#endif

#endif
