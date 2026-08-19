#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <dlfcn.h>
#include <pthread.h>

#if defined(__GNUC__) || defined(__clang__)
#define XTBLOOM_TEST_EXPORT __attribute__((visibility("default")))
#else
#define XTBLOOM_TEST_EXPORT
#endif

extern int __pthread_key_create(pthread_key_t*, void (*)(void*));
extern int __pthread_key_delete(pthread_key_t);
extern void* __pthread_getspecific(pthread_key_t);
extern int __pthread_setspecific(pthread_key_t, const void*);

/* This call is intentionally issued from the private namespace. On glibc
 * 2.33 and older it enters libdl's _dlerror_run through the internal
 * __pthread_* aliases before performing the requested lookup. */
XTBLOOM_TEST_EXPORT void* xtbloom_mkl_tss_fixture_private_dlsym(void* handle, const char* name) {
  return dlsym(handle, name);
}

XTBLOOM_TEST_EXPORT int xtbloom_mkl_tss_fixture_key_create(pthread_key_t* key) {
  return pthread_key_create(key, 0);
}

XTBLOOM_TEST_EXPORT int xtbloom_mkl_tss_fixture_key_delete(pthread_key_t key) {
  return pthread_key_delete(key);
}

XTBLOOM_TEST_EXPORT void* xtbloom_mkl_tss_fixture_getspecific(pthread_key_t key) {
  return pthread_getspecific(key);
}

XTBLOOM_TEST_EXPORT int xtbloom_mkl_tss_fixture_setspecific(pthread_key_t key, const void* value) {
  return pthread_setspecific(key, value);
}

XTBLOOM_TEST_EXPORT int xtbloom_mkl_tss_fixture_internal_key_create(pthread_key_t* key) {
  return __pthread_key_create(key, 0);
}

XTBLOOM_TEST_EXPORT int xtbloom_mkl_tss_fixture_internal_key_delete(pthread_key_t key) {
  return __pthread_key_delete(key);
}

XTBLOOM_TEST_EXPORT void* xtbloom_mkl_tss_fixture_internal_getspecific(pthread_key_t key) {
  return __pthread_getspecific(key);
}

XTBLOOM_TEST_EXPORT int xtbloom_mkl_tss_fixture_internal_setspecific(pthread_key_t key,
                                                                     const void* value) {
  return __pthread_setspecific(key, value);
}
