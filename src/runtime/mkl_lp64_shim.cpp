// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

// Private host-isolated MKL provider shim.
//
// xtbloom never dlopens libmkl_rt directly. libmkl_rt is the MKL interface-layer
// dispatcher: initializing it in LP64 mode mutates process-global MKL state that
// an embedding application may already own, and reading MKL_INTERFACE_LAYER can
// make xtbloom depend on the host's MKL lifecycle. Instead, CMake links this
// translation unit into a private shared object with fixed DT_NEEDED dependencies
// on libmkl_intel_lp64, libmkl_sequential, and libmkl_core. Loading those three
// component libraries directly (never libmkl_rt) yields an LP64 + sequential
// provider. The runtime loads this shim in a new glibc link-map namespace because
// RTLD_LOCAL alone would still allow pre-existing global host symbols to
// interpose. The host's interface/threading state is therefore unchanged, and
// LP64 xtbloom calls remain correct even when the host uses ILP64.
//
// glibc link-map namespaces do not isolate pthread thread-specific-data slots:
// each namespace has its own pthread-key allocator, while THREAD_SELF specific
// storage is shared. A pthread_key_create call from the dlmopen namespace can
// therefore reuse a key owned by CPython, Torch, or another host library and
// overwrite that host's thread state (glibc bug nptl/24776). Keep the MKL ELF
// namespace isolation, but interpose the four pthread TSD entry points for the
// whole private dependency scope and forward them to the already-loaded base-
// namespace pthread implementation. This preserves one process-wide key
// registry without exposing any MKL/LAPACK symbol through RTLD_DEFAULT.
//
// The shim otherwise intentionally exports no provider symbols; the eigensolver
// factory only dlopens it and resolves LAPACKE/CBLAS/thread-control symbols
// through its dependency scope.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <dlfcn.h>
#include <link.h>
#include <pthread.h>

#include <atomic>
#include <cerrno>
#include <cstring>

namespace {

void* base_pthread_symbol_address(const char* name) noexcept {
  // On glibc < 2.34 pthread lives in libpthread; on newer glibc the public
  // entry points live in libc and libpthread is only a compatibility DSO.
  // RTLD_NOLOAD is deliberate: the host already owns the base runtime, and
  // xTBloom must not introduce a second host pthread implementation merely to
  // service the private MKL namespace.
  const char* const libraries[] = {"libpthread.so.0", "libc.so.6"};
  for (const char* library : libraries) {
    void* handle = dlmopen(LM_ID_BASE, library, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    if (handle == nullptr) {
      continue;
    }
    dlerror();
    void* symbol = dlsym(handle, name);
    const char* error = dlerror();
    static_cast<void>(dlclose(handle));
    if (symbol != nullptr && error == nullptr) {
      return symbol;
    }
  }
  return nullptr;
}

template <typename Function>
Function cached_base_pthread_symbol(std::atomic<void*>& slot, const char* name) noexcept {
  void* symbol = slot.load(std::memory_order_acquire);
  if (symbol == nullptr) {
    void* resolved = base_pthread_symbol_address(name);
    if (resolved == nullptr) {
      return nullptr;
    }
    void* expected = nullptr;
    if (!slot.compare_exchange_strong(expected, resolved, std::memory_order_release,
                                      std::memory_order_acquire)) {
      symbol = expected;
    } else {
      symbol = resolved;
    }
  }
  Function function = nullptr;
  static_assert(sizeof(function) == sizeof(symbol));
  std::memcpy(&function, &symbol, sizeof(function));
  return function;
}

std::atomic<void*> g_pthread_key_create{nullptr};
std::atomic<void*> g_pthread_key_delete{nullptr};
std::atomic<void*> g_pthread_getspecific{nullptr};
std::atomic<void*> g_pthread_setspecific{nullptr};

#if defined(__GNUC__) || defined(__clang__)
#define XTBLOOM_MKL_SHIM_EXPORT __attribute__((visibility("default")))
#else
#define XTBLOOM_MKL_SHIM_EXPORT
#endif

}  // namespace

extern "C" XTBLOOM_MKL_SHIM_EXPORT int pthread_key_create(pthread_key_t* key,
                                                          void (*destructor)(void*)) noexcept {
  using Function = int (*)(pthread_key_t*, void (*)(void*));
  const Function function =
      cached_base_pthread_symbol<Function>(g_pthread_key_create, "pthread_key_create");
  return function == nullptr ? EAGAIN : function(key, destructor);
}

extern "C" XTBLOOM_MKL_SHIM_EXPORT int pthread_key_delete(pthread_key_t key) noexcept {
  using Function = int (*)(pthread_key_t);
  const Function function =
      cached_base_pthread_symbol<Function>(g_pthread_key_delete, "pthread_key_delete");
  return function == nullptr ? EINVAL : function(key);
}

extern "C" XTBLOOM_MKL_SHIM_EXPORT void* pthread_getspecific(pthread_key_t key) noexcept {
  using Function = void* (*)(pthread_key_t);
  const Function function =
      cached_base_pthread_symbol<Function>(g_pthread_getspecific, "pthread_getspecific");
  return function == nullptr ? nullptr : function(key);
}

extern "C" XTBLOOM_MKL_SHIM_EXPORT int pthread_setspecific(pthread_key_t key,
                                                           const void* value) noexcept {
  using Function = int (*)(pthread_key_t, const void*);
  const Function function =
      cached_base_pthread_symbol<Function>(g_pthread_setspecific, "pthread_setspecific");
  return function == nullptr ? EINVAL : function(key, value);
}

namespace xtbloom {
namespace detail {
namespace gfn2 {
namespace {

int xtbloom_mkl_lp64_shim_unit_marker = 0;

}  // namespace
}  // namespace gfn2
}  // namespace detail
}  // namespace xtbloom
