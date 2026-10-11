#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <dlfcn.h>
#include <errno.h>
#include <link.h>
#include <pthread.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "runtime/mkl_pthread_tss_bridge.h"

typedef void* (*private_dlsym_fn)(void*, const char*);
typedef int (*fixture_key_create_fn)(pthread_key_t*);
typedef int (*fixture_key_delete_fn)(pthread_key_t);
typedef void* (*fixture_getspecific_fn)(pthread_key_t);
typedef int (*fixture_setspecific_fn)(pthread_key_t, const void*);

static int load_symbol(void* handle, const char* name, void* function, size_t function_size) {
  void* symbol;
  const char* error;
  dlerror();
  symbol = dlsym(handle, name);
  error = dlerror();
  if (symbol == 0 || error != 0 || function_size != sizeof(symbol)) {
    fprintf(stderr, "failed to resolve %s: %s\n", name, error == 0 ? "unknown error" : error);
    return 0;
  }
  memcpy(function, &symbol, sizeof(symbol));
  return 1;
}

#define CHILD_CHECK(condition)                                                      \
  do {                                                                              \
    if (!(condition)) {                                                             \
      fprintf(stderr, "child CHECK failed at line %d: %s\n", __LINE__, #condition); \
      return 1;                                                                     \
    }                                                                               \
  } while (0)

static int run_child(const char* bridge_path, const char* fixture_path, pthread_key_t host_key,
                     void* host_sentinel) {
  void* bridge_handle;
  void* fixture_handle;
  Lmid_t namespace_id = LM_ID_BASE;
  xtbloom_mkl_pthread_tss_bridge_initialize_fn initialize = 0;
  xtbloom_mkl_pthread_key_create_fn bridge_key_create = 0;
  xtbloom_mkl_pthread_key_create_fn bridge_internal_key_create = 0;
  private_dlsym_fn private_dlsym = 0;
  fixture_key_create_fn fixture_key_create = 0;
  fixture_key_delete_fn fixture_key_delete = 0;
  fixture_getspecific_fn fixture_getspecific = 0;
  fixture_setspecific_fn fixture_setspecific = 0;
  fixture_key_create_fn internal_key_create = 0;
  fixture_key_delete_fn internal_key_delete = 0;
  fixture_getspecific_fn internal_getspecific = 0;
  fixture_setspecific_fn internal_setspecific = 0;
  pthread_key_t public_key = (pthread_key_t)-1;
  pthread_key_t internal_key = (pthread_key_t)-1;
  int public_sentinel = 1;
  int internal_sentinel = 2;
  xtbloom_mkl_pthread_tss_api api;

  bridge_handle = dlmopen(LM_ID_NEWLM, bridge_path, RTLD_NOW | RTLD_LOCAL);
  CHILD_CHECK(bridge_handle != 0);
  CHILD_CHECK(dlinfo(bridge_handle, RTLD_DI_LMID, &namespace_id) == 0);
  CHILD_CHECK(namespace_id != LM_ID_BASE);
  CHILD_CHECK(load_symbol(bridge_handle, XTBLOOM_MKL_PTHREAD_TSS_BRIDGE_INITIALIZE_SYMBOL,
                          &initialize, sizeof(initialize)));
  CHILD_CHECK(load_symbol(bridge_handle, "pthread_key_create", &bridge_key_create,
                          sizeof(bridge_key_create)));
  CHILD_CHECK(load_symbol(bridge_handle, "__pthread_key_create", &bridge_internal_key_create,
                          sizeof(bridge_internal_key_create)));

  /* Even an accidental pre-initialization call must fail closed without
   * allocating from a private key registry or touching the host sentinel. */
  CHILD_CHECK(bridge_key_create(&public_key, 0) == EAGAIN);
  CHILD_CHECK(bridge_internal_key_create(&internal_key, 0) == EAGAIN);
  CHILD_CHECK(pthread_getspecific(host_key) == host_sentinel);

  api.key_create = pthread_key_create;
  api.key_delete = pthread_key_delete;
  api.getspecific = pthread_getspecific;
  api.setspecific = pthread_setspecific;
  CHILD_CHECK(initialize(&api) == 0);

  fixture_handle = dlmopen(namespace_id, fixture_path, RTLD_NOW | RTLD_LOCAL);
  CHILD_CHECK(fixture_handle != 0);
  CHILD_CHECK(load_symbol(fixture_handle, "xtbloom_mkl_tss_fixture_private_dlsym", &private_dlsym,
                          sizeof(private_dlsym)));
  CHILD_CHECK(load_symbol(fixture_handle, "xtbloom_mkl_tss_fixture_key_create", &fixture_key_create,
                          sizeof(fixture_key_create)));
  CHILD_CHECK(load_symbol(fixture_handle, "xtbloom_mkl_tss_fixture_key_delete", &fixture_key_delete,
                          sizeof(fixture_key_delete)));
  CHILD_CHECK(load_symbol(fixture_handle, "xtbloom_mkl_tss_fixture_getspecific",
                          &fixture_getspecific, sizeof(fixture_getspecific)));
  CHILD_CHECK(load_symbol(fixture_handle, "xtbloom_mkl_tss_fixture_setspecific",
                          &fixture_setspecific, sizeof(fixture_setspecific)));
  CHILD_CHECK(load_symbol(fixture_handle, "xtbloom_mkl_tss_fixture_internal_key_create",
                          &internal_key_create, sizeof(internal_key_create)));
  CHILD_CHECK(load_symbol(fixture_handle, "xtbloom_mkl_tss_fixture_internal_key_delete",
                          &internal_key_delete, sizeof(internal_key_delete)));
  CHILD_CHECK(load_symbol(fixture_handle, "xtbloom_mkl_tss_fixture_internal_getspecific",
                          &internal_getspecific, sizeof(internal_getspecific)));
  CHILD_CHECK(load_symbol(fixture_handle, "xtbloom_mkl_tss_fixture_internal_setspecific",
                          &internal_setspecific, sizeof(internal_setspecific)));

  /* This is the old-glibc bootstrap edge: private libdl enters _dlerror_run,
   * whose weak __pthread_* references must already resolve to the bridge. */
  CHILD_CHECK(private_dlsym(fixture_handle, "pthread_key_create") != 0);
  CHILD_CHECK(pthread_getspecific(host_key) == host_sentinel);

  CHILD_CHECK(fixture_key_create(&public_key) == 0);
  CHILD_CHECK(internal_key_create(&internal_key) == 0);
  CHILD_CHECK(public_key != host_key);
  CHILD_CHECK(internal_key != host_key);
  CHILD_CHECK(internal_key != public_key);

  CHILD_CHECK(fixture_setspecific(public_key, &public_sentinel) == 0);
  CHILD_CHECK(pthread_getspecific(public_key) == &public_sentinel);
  CHILD_CHECK(pthread_setspecific(public_key, &internal_sentinel) == 0);
  CHILD_CHECK(fixture_getspecific(public_key) == &internal_sentinel);

  CHILD_CHECK(internal_setspecific(internal_key, &internal_sentinel) == 0);
  CHILD_CHECK(pthread_getspecific(internal_key) == &internal_sentinel);
  CHILD_CHECK(pthread_setspecific(internal_key, &public_sentinel) == 0);
  CHILD_CHECK(internal_getspecific(internal_key) == &public_sentinel);
  CHILD_CHECK(pthread_getspecific(host_key) == host_sentinel);

  CHILD_CHECK(fixture_key_delete(public_key) == 0);
  CHILD_CHECK(internal_key_delete(internal_key) == 0);
  return 0;
}

int main(int argc, char** argv) {
  pthread_key_t host_key = (pthread_key_t)-1;
  void* host_sentinel;
  pid_t child;
  int child_status = 0;

  if (argc != 3) {
    fprintf(stderr, "usage: %s BRIDGE_DSO FIXTURE_DSO\n", argv[0]);
    return 2;
  }

  /* This must be the first pthread key created by the test. PROT_NONE makes
   * the historical private-libdl misinterpretation a deterministic SIGSEGV. */
  if (pthread_key_create(&host_key, 0) != 0) {
    fprintf(stderr, "failed to create host pthread key\n");
    return 3;
  }
  if (host_key != (pthread_key_t)0) {
    fprintf(stderr, "bootstrap test requires the host sentinel to own pthread key 0\n");
    return 4;
  }
  host_sentinel = mmap(0, 1, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (host_sentinel == MAP_FAILED || pthread_setspecific(host_key, host_sentinel) != 0) {
    fprintf(stderr, "failed to install host TSS sentinel\n");
    return 5;
  }

  child = fork();
  if (child == 0) {
    _exit(run_child(argv[1], argv[2], host_key, host_sentinel));
  }
  if (child < 0 || waitpid(child, &child_status, 0) != child) {
    fprintf(stderr, "failed to run bootstrap child\n");
    return 6;
  }
  if (WIFSIGNALED(child_status)) {
    fprintf(stderr, "bootstrap child terminated by signal %d (%s)\n", WTERMSIG(child_status),
            strsignal(WTERMSIG(child_status)));
    return 7;
  }
  if (!WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0) {
    fprintf(stderr, "bootstrap child failed with status %d\n",
            WIFEXITED(child_status) ? WEXITSTATUS(child_status) : -1);
    return 8;
  }
  if (pthread_getspecific(host_key) != host_sentinel) {
    fprintf(stderr, "parent host TSS sentinel changed\n");
    return 9;
  }

  pthread_setspecific(host_key, 0);
  pthread_key_delete(host_key);
  munmap(host_sentinel, 1);
  return 0;
}
