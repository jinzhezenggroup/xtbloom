/*
 * gpuxtb CUDA runtime loader.
 *
 * The CUDA backend is built into libgpuxtb, but the proprietary NVIDIA CUDA
 * runtime, cuBLAS, cuSOLVER, and driver libraries (cudart/cublas/cusolver/
 * libcuda) are never linked into it: libgpuxtb therefore carries no
 * DT_NEEDED entry on a GPL-incompatible library.  Instead, the build
 * generates one ELF trampoline shim per library (see cmake/3rdparty/implib)
 * and links the shims directly into libgpuxtb.  The first call to any wrapped
 * symbol enters the trampoline below, which loads the real library lazily
 * with dlopen and resolves the symbol with dlsym.
 *
 * This keeps libgpuxtb's distribution dependency model "dlopen-based, not
 * linked": on a machine without any NVIDIA runtime the library still loads
 * (GPU calls report a clean "backend unavailable" error) because every
 * wrapped symbol degrades to an error-returning stub instead of crashing.
 * The same model as the CPU eigensolver's lazy BLAS/LAPACK loading in
 * src/model/gfn2/eigensolver.cpp.
 *
 * For the GPL compatibility policy this is a mechanism, not the legal
 * basis.  Issue #162 records the owner/legal decision that documents why
 * loading these libraries as separate works through their published C APIs
 * is allowed for GPL-3.0-or-later distribution.
 */

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* cudaErrorSharedObjectSymbolNotFound == 300.  Kept as a numeric literal so
 * this translation unit can build without any CUDA toolkit header. */
#define GPU_XTB_CUDA_ERROR_SHARED_SYMBOL_NOT_FOUND 300

#if defined(_WIN32)
#error "the gpuxtb CUDA dlopen loader is implemented for POSIX dlopen only"
#endif

/* Directory candidates searched in addition to the loader's default dlopen
 * path.  The Python layer (python/gpuxtb/library.py) preloads the packaged
 * nvidia-* runtimes before compute, and LD_LIBRARY_PATH covers most native
 * deployments; these paths mainly help toolkit-only installs that are not in
 * ldconfig. */
static const char* const kCudaLibrarySearchDirs[] = {
    "/usr/local/cuda/lib64",
    "/usr/local/cuda/targets/x86_64-linux/lib",
    "/usr/local/cuda/targets/sbsa-linux/lib",
};

/* Resolve a directory potentially provided by the environment
 * (CUDA_HOME/CUDA_PATH), with "/lib64" appended. */
static void gpu_xtb_append_env_libdir(const char* variable, char output[512]) {
  const char* root = getenv(variable);
  if (root == NULL || *root == '\0') {
    output[0] = '\0';
    return;
  }
  const int written = snprintf(output, 512, "%s/lib64", root);
  if (written < 0 || written >= 512) {
    output[0] = '\0';
  }
}

static void* gpu_xtb_try_dlopen(const char* name) {
  void* handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
  if (handle != NULL) {
    return handle;
  }
  char env_libdirs[2][512];
  gpu_xtb_append_env_libdir("CUDA_HOME", env_libdirs[0]);
  gpu_xtb_append_env_libdir("CUDA_PATH", env_libdirs[1]);
  for (size_t i = 0; i < sizeof(kCudaLibrarySearchDirs) / sizeof(*kCudaLibrarySearchDirs); ++i) {
    const char* directory = kCudaLibrarySearchDirs[i];
    if (directory == NULL || *directory == '\0') {
      continue;
    }
    char candidate[512];
    const int written = snprintf(candidate, sizeof(candidate), "%s/%s", directory, name);
    if (written < 0 || (size_t)written >= sizeof(candidate)) {
      continue;
    }
    handle = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
    if (handle != NULL) {
      return handle;
    }
  }
  for (size_t i = 0; i < 2; ++i) {
    if (env_libdirs[i][0] == '\0') {
      continue;
    }
    char candidate[512];
    const int written = snprintf(candidate, sizeof(candidate), "%s/%s", env_libdirs[i], name);
    if (written < 0 || (size_t)written >= sizeof(candidate)) {
      continue;
    }
    handle = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
    if (handle != NULL) {
      return handle;
    }
  }
  return NULL;
}

/* gpuxtb_cuda_dlopen is referenced by the generated init.c trampolines as the
 * --dlopen-callback.  ``lib_name`` is the SONAME implib-gen baked in from the
 * toolkit library (for example libcudart.so.12). */
void* gpuxtb_cuda_dlopen(const char* lib_name) {
  if (lib_name == NULL || *lib_name == '\0') {
    return NULL;
  }

  /* Load the baked SONAME first, then SONAME aliases that make the loader
   * resilient to the runtime library having a different minor version than
   * the toolkit that produced the shims.  The driver is loaded as
   * libcuda.so.1 regardless of what the toolkit stub was named. */
  const char* candidates[4];
  size_t candidate_count = 0;
  candidates[candidate_count++] = lib_name;
  if (strstr(lib_name, "libcuda.so") != NULL) {
    candidates[candidate_count++] = "libcuda.so.1";
  } else if (strstr(lib_name, "libcudart.so") != NULL) {
    candidates[candidate_count++] = "libcudart.so.12";
    candidates[candidate_count++] = "libcudart.so.11";
    candidates[candidate_count++] = "libcudart.so.10";
  } else if (strstr(lib_name, "libcublas.so") != NULL) {
    candidates[candidate_count++] = "libcublas.so.12";
    candidates[candidate_count++] = "libcublas.so.11";
  } else if (strstr(lib_name, "libcusolver.so") != NULL) {
    candidates[candidate_count++] = "libcusolver.so.11";
    candidates[candidate_count++] = "libcusolver.so.12";
    candidates[candidate_count++] = "libcusolver.so.10";
  }

  void* handle = NULL;
  for (size_t i = 0; i < candidate_count && i < sizeof(candidates) / sizeof(*candidates); ++i) {
    if (candidates[i] == NULL) {
      continue;
    }
    handle = gpu_xtb_try_dlopen(candidates[i]);
    if (handle != NULL) {
      break;
    }
  }

  if (handle == NULL) {
    /* Report once per process; the wrapping shims degrade every symbol to a
     * stub so a CPU-only deployment loads cleanly. */
    static int reported = 0;
    if (!reported) {
      reported = 1;
      fprintf(stderr,
              "gpuxtb: could not load the NVIDIA CUDA runtime '%s' "
              "(dlopen error: %s); the GPU backend is disabled\n",
              lib_name, dlerror());
    }
  }
  return handle;
}

/* Crash-safe fallbacks used when a wrapped CUDA symbol cannot be resolved
 * (runtime not installed, or an older runtime without that symbol).  They let
 * failure paths format their error strings and return real error codes
 * instead of jumping through a NULL function pointer. */

/* Thread-local buffer keeps error-text helpers usable from several threads. */
static _Thread_local char kGpuXtbErrorText[128];

static const char* gpu_xtb_shared_symbol_error_text(int code) {
  snprintf(kGpuXtbErrorText, sizeof(kGpuXtbErrorText),
           "CUDA runtime symbol unavailable (code %d); the gpuxtb GPU backend "
           "is disabled because an NVIDIA runtime library is not loaded",
           code);
  return kGpuXtbErrorText;
}

static const char* gpuxtb_cuda_get_error_string_fallback(int code) {
  return gpu_xtb_shared_symbol_error_text(code);
}

static const char* gpuxtb_cuda_get_error_name_fallback(int code) {
  return gpu_xtb_shared_symbol_error_text(code);
}

static int gpu_xtb_driver_error_string_fallback(int code, const char** detail) {
  if (detail != NULL) {
    *detail = gpu_xtb_shared_symbol_error_text(code);
  }
  return (int)GPU_XTB_CUDA_ERROR_SHARED_SYMBOL_NOT_FOUND;
}

static int gpu_xtb_driver_error_name_fallback(int code, const char** name) {
  if (name != NULL) {
    *name = gpu_xtb_shared_symbol_error_text(code);
  }
  return (int)GPU_XTB_CUDA_ERROR_SHARED_SYMBOL_NOT_FOUND;
}

/* Default stub for any other symbol: return
 * cudaErrorSharedObjectSymbolNotFound so the caller observes a failed CUDA
 * operation rather than dereferencing garbage. */
static int gpu_xtb_symbol_unavailable(void) {
  return (int)GPU_XTB_CUDA_ERROR_SHARED_SYMBOL_NOT_FOUND;
}

/* gpuxtb_cuda_dlsym is referenced by the generated init.c trampolines as the
 * --dlsym-callback.  ``sym_name`` is one wrapped symbol; a NULL return would
 * leave a NULL trampoline target, so missing symbols always resolve to a
 * safe stub. */
void* gpuxtb_cuda_dlsym(void* handle, const char* sym_name) {
  if (handle != NULL && sym_name != NULL && *sym_name != '\0') {
    void* symbol = dlsym(handle, sym_name);
    if (symbol != NULL) {
      return symbol;
    }
  }

  if (sym_name == NULL) {
    return (void*)&gpu_xtb_symbol_unavailable;
  }
  /* Text-returning helpers take a dedicated stub so the failure path can
   * format an error message without a bogus pointer. */
  if (strcmp(sym_name, "cudaGetErrorString") == 0) {
    return (void*)&gpuxtb_cuda_get_error_string_fallback;
  }
  if (strcmp(sym_name, "cudaGetErrorName") == 0) {
    return (void*)&gpuxtb_cuda_get_error_name_fallback;
  }
  if (strcmp(sym_name, "cuGetErrorString") == 0) {
    return (void*)&gpu_xtb_driver_error_string_fallback;
  }
  if (strcmp(sym_name, "cuGetErrorName") == 0) {
    return (void*)&gpu_xtb_driver_error_name_fallback;
  }
  return (void*)&gpu_xtb_symbol_unavailable;
}