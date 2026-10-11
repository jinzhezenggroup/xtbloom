// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <pthread.h>

static pthread_key_t provider_key;

static void provider_key_destructor(void* value) { (void)value; }

#if defined(__GNUC__) || defined(__clang__)
__attribute__((constructor))
#endif
static void register_provider_key(void) {
  /* The production bridge routes this key into the base pthread registry, but
   * its destructor remains code owned by this private provider. Any factory
   * failure after this constructor runs must therefore retain the namespace
   * until process exit instead of dlclosing it before the worker terminates. */
  if (pthread_key_create(&provider_key, provider_key_destructor) != 0 ||
      pthread_setspecific(provider_key, &provider_key) != 0) {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_trap();
#endif
  }
}
