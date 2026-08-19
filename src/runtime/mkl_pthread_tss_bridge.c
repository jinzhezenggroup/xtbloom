// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

/*
 * Thread-specific-storage bridge for the private MKL link-map.
 *
 * glibc link-map namespaces duplicate the pthread-key registry but share each
 * thread's specific-data slots. A private registry can therefore reuse a key
 * owned by CPython or another host and interpret the host value as private
 * state. glibc 2.33 and older also make libdl's _dlerror_run call the internal
 * __pthread_* aliases before an ordinary DSO can resolve anything with dlsym.
 *
 * This DSO is deliberately linked with -nostdlib and has zero DT_NEEDED
 * entries. The base loader first places it alone in a new namespace, injects
 * the already-resolved base pthread entry points, and only then loads the MKL
 * shim into that namespace with this bridge as its first dependency. Both the
 * public and internal pthread TSS spellings consequently share the host key
 * registry before private libc, libdl, or MKL can run a relocation/constructor.
 */

#include "runtime/mkl_pthread_tss_bridge.h"

#include <errno.h>

#if defined(__GNUC__) || defined(__clang__)
#define XTBLOOM_MKL_TSS_EXPORT __attribute__((visibility("default")))
#else
#define XTBLOOM_MKL_TSS_EXPORT
#endif

static xtbloom_mkl_pthread_key_create_fn base_key_create;
static xtbloom_mkl_pthread_key_delete_fn base_key_delete;
static xtbloom_mkl_pthread_getspecific_fn base_getspecific;
static xtbloom_mkl_pthread_setspecific_fn base_setspecific;
static int bridge_initialized;

static int bridge_is_initialized(void) {
  return __atomic_load_n(&bridge_initialized, __ATOMIC_ACQUIRE) != 0;
}

XTBLOOM_MKL_TSS_EXPORT int xtbloom_mkl_pthread_tss_bridge_initialize(
    const xtbloom_mkl_pthread_tss_api* api) {
  if (api == 0 || api->key_create == 0 || api->key_delete == 0 || api->getspecific == 0 ||
      api->setspecific == 0) {
    return EINVAL;
  }
  if (bridge_is_initialized()) {
    return base_key_create == api->key_create && base_key_delete == api->key_delete &&
                   base_getspecific == api->getspecific && base_setspecific == api->setspecific
               ? 0
               : EBUSY;
  }

  /* The bridge is initialized before any dependent DSO is admitted to this
   * namespace, so these writes cannot race a forwarding call. The release
   * publication also documents the immutable table seen by later workers. */
  base_key_create = api->key_create;
  base_key_delete = api->key_delete;
  base_getspecific = api->getspecific;
  base_setspecific = api->setspecific;
  __atomic_store_n(&bridge_initialized, 1, __ATOMIC_RELEASE);
  return 0;
}

static int forward_key_create(pthread_key_t* key, void (*destructor)(void*)) {
  return bridge_is_initialized() ? base_key_create(key, destructor) : EAGAIN;
}

static int forward_key_delete(pthread_key_t key) {
  return bridge_is_initialized() ? base_key_delete(key) : EINVAL;
}

static void* forward_getspecific(pthread_key_t key) {
  return bridge_is_initialized() ? base_getspecific(key) : 0;
}

static int forward_setspecific(pthread_key_t key, const void* value) {
  return bridge_is_initialized() ? base_setspecific(key, value) : EINVAL;
}

XTBLOOM_MKL_TSS_EXPORT int pthread_key_create(pthread_key_t* key, void (*destructor)(void*)) {
  return forward_key_create(key, destructor);
}

XTBLOOM_MKL_TSS_EXPORT int pthread_key_delete(pthread_key_t key) { return forward_key_delete(key); }

XTBLOOM_MKL_TSS_EXPORT void* pthread_getspecific(pthread_key_t key) {
  return forward_getspecific(key);
}

XTBLOOM_MKL_TSS_EXPORT int pthread_setspecific(pthread_key_t key, const void* value) {
  return forward_setspecific(key, value);
}

/* glibc <= 2.33 libdl uses these weak internal aliases in _dlerror_run rather
 * than the four public symbols above. They must be present before the private
 * libdl is loaded; adding only public interposition leaves the bootstrap bug. */
XTBLOOM_MKL_TSS_EXPORT int __pthread_key_create(pthread_key_t* key, void (*destructor)(void*)) {
  return forward_key_create(key, destructor);
}

XTBLOOM_MKL_TSS_EXPORT int __pthread_key_delete(pthread_key_t key) {
  return forward_key_delete(key);
}

XTBLOOM_MKL_TSS_EXPORT void* __pthread_getspecific(pthread_key_t key) {
  return forward_getspecific(key);
}

XTBLOOM_MKL_TSS_EXPORT int __pthread_setspecific(pthread_key_t key, const void* value) {
  return forward_setspecific(key, value);
}
