// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.
/*
 * Narrow LP64 LAPACKE/CBLAS adapter for the reviewed Pyodide OpenBLAS side
 * module. Pyodide builds OpenBLAS without LAPACKE, while xTBloom uses only the
 * column-major ``*_work`` entry points below. Every dispatch call is resolved
 * from the exact private provider path prepared by the Python wheel loader;
 * ordinary WebAssembly imports are deliberately not used for computation,
 * because Emscripten has no namespace or deep-bind mechanism that could stop
 * a previously loaded SciPy provider from interposing on generic symbols.
 */

#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef XTBLOOM_PYODIDE_OPENBLAS_FILENAME
#error "XTBLOOM_PYODIDE_OPENBLAS_FILENAME must name the reviewed private provider"
#endif

#if defined(__GNUC__)
#define XTBLOOM_PYODIDE_EXPORT __attribute__((visibility("default")))
#define XTBLOOM_PYODIDE_USED __attribute__((used))
#else
#define XTBLOOM_PYODIDE_EXPORT
#define XTBLOOM_PYODIDE_USED
#endif

typedef int32_t lapack_int;

typedef int (*dpotrf_function)(const char*, const lapack_int*, double*, const lapack_int*,
                               lapack_int*);
typedef int (*dpocon_function)(const char*, const lapack_int*, const double*, const lapack_int*,
                               const double*, double*, double*, lapack_int*, lapack_int*);
typedef int (*dsyevd_function)(const char*, const char*, const lapack_int*, double*,
                               const lapack_int*, double*, double*, const lapack_int*, lapack_int*,
                               const lapack_int*, lapack_int*);
/* Pyodide's Emscripten OpenBLAS exports the CBLAS entry points with an i32
 * result even though native OpenBLAS declares them void. WebAssembly indirect
 * calls require an exact function type, so the raw provider dispatch retains
 * that ABI result and the public adapter wrappers deliberately discard it. */
typedef int (*dtrsm_function)(int, int, int, int, int, lapack_int, lapack_int, double,
                              const double*, lapack_int, double*, lapack_int);
typedef int (*dgemm_function)(int, int, int, lapack_int, lapack_int, lapack_int, double,
                              const double*, lapack_int, const double*, lapack_int, double, double*,
                              lapack_int);
typedef const char* (*get_config_function)(void);
typedef int (*set_threads_local_function)(int);

struct provider_dispatch {
  void* handle;
  dpotrf_function dpotrf;
  dpocon_function dpocon;
  dsyevd_function dsyevd;
  dtrsm_function dtrsm;
  dgemm_function dgemm;
  get_config_function get_config;
  set_threads_local_function set_threads_local;
  int initialized;
  int ready;
};

static struct provider_dispatch dispatch;

static int path_has_expected_basename(const char* path) {
  const char* slash;
  const char* backslash;
  const char* basename;
  /* The wheel loader publishes an installed absolute path. Rejecting relative
   * names keeps the adapter from consulting the process working directory or
   * Emscripten's generic dynamic-library search order. */
  if (path == NULL || path[0] != '/') {
    return 0;
  }
  slash = strrchr(path, '/');
  backslash = strrchr(path, '\\');
  basename = slash;
  if (backslash != NULL && (basename == NULL || backslash > basename)) {
    basename = backslash;
  }
  basename = basename == NULL ? path : basename + 1;
  return strcmp(basename, XTBLOOM_PYODIDE_OPENBLAS_FILENAME) == 0;
}

static int load_function(void* handle, const char* name, void* output, size_t output_size) {
  void* symbol;
  if (output_size != sizeof(symbol)) {
    return 0;
  }
  dlerror();
  symbol = dlsym(handle, name);
  if (symbol == NULL || dlerror() != NULL) {
    return 0;
  }
  memcpy(output, &symbol, sizeof(symbol));
  return 1;
}

#define XTBLOOM_LOAD(handle, name, field) load_function((handle), (name), &(field), sizeof(field))

static int resolve_provider(void) {
  const char* path;
  if (dispatch.initialized) {
    return dispatch.ready;
  }
  dispatch.initialized = 1;
  path = getenv("XTBLOOM_PYODIDE_OPENBLAS");
  if (!path_has_expected_basename(path)) {
    return 0;
  }
  dispatch.handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (dispatch.handle == NULL) {
    return 0;
  }
  dispatch.ready =
      XTBLOOM_LOAD(dispatch.handle, "dpotrf_", dispatch.dpotrf) &&
      XTBLOOM_LOAD(dispatch.handle, "dpocon_", dispatch.dpocon) &&
      XTBLOOM_LOAD(dispatch.handle, "dsyevd_", dispatch.dsyevd) &&
      XTBLOOM_LOAD(dispatch.handle, "cblas_dtrsm", dispatch.dtrsm) &&
      XTBLOOM_LOAD(dispatch.handle, "cblas_dgemm", dispatch.dgemm) &&
      XTBLOOM_LOAD(dispatch.handle, "openblas_get_config", dispatch.get_config) &&
      XTBLOOM_LOAD(dispatch.handle, "openblas_set_num_threads_local", dispatch.set_threads_local);
  return dispatch.ready;
}

/*
 * This one ordinary import exists only to make the adapter's dylink NEEDED
 * edge visible to ``pyodide auditwheel repair``. Production computation never
 * calls through it; every function above comes from the exact absolute handle.
 */
extern const char* openblas_get_config(void);
XTBLOOM_PYODIDE_EXPORT XTBLOOM_PYODIDE_USED const char* xtbloom_pyodide_openblas_dependency_anchor(
    void) {
  return openblas_get_config();
}

XTBLOOM_PYODIDE_EXPORT const char* xtbloom_pyodide_openblas_get_config(void) {
  return resolve_provider() ? dispatch.get_config() : NULL;
}

XTBLOOM_PYODIDE_EXPORT int xtbloom_pyodide_openblas_set_num_threads_local(int threads) {
  return resolve_provider() ? dispatch.set_threads_local(threads) : 0;
}

XTBLOOM_PYODIDE_EXPORT lapack_int xtbloom_pyodide_LAPACKE_dpotrf_work(
    int matrix_layout, char uplo, lapack_int n, double* matrix, lapack_int leading_dimension) {
  lapack_int info = 0;
  if (matrix_layout != 102) {
    return -1;
  }
  if (!resolve_provider()) {
    return -1024;
  }
  (void)dispatch.dpotrf(&uplo, &n, matrix, &leading_dimension, &info);
  return info < 0 ? info - 1 : info;
}

XTBLOOM_PYODIDE_EXPORT lapack_int xtbloom_pyodide_LAPACKE_dpocon_work(
    int matrix_layout, char uplo, lapack_int n, const double* factor, lapack_int leading_dimension,
    double matrix_one_norm, double* reciprocal_condition, double* work, lapack_int* integer_work) {
  lapack_int info = 0;
  if (matrix_layout != 102) {
    return -1;
  }
  if (!resolve_provider()) {
    return -1024;
  }
  (void)dispatch.dpocon(&uplo, &n, factor, &leading_dimension, &matrix_one_norm,
                        reciprocal_condition, work, integer_work, &info);
  return info < 0 ? info - 1 : info;
}

XTBLOOM_PYODIDE_EXPORT lapack_int xtbloom_pyodide_LAPACKE_dsyevd_work(
    int matrix_layout, char job_vectors, char uplo, lapack_int n, double* matrix,
    lapack_int leading_dimension, double* eigenvalues, double* work, lapack_int work_count,
    lapack_int* integer_work, lapack_int integer_work_count) {
  lapack_int info = 0;
  if (matrix_layout != 102) {
    return -1;
  }
  if (!resolve_provider()) {
    return -1024;
  }
  (void)dispatch.dsyevd(&job_vectors, &uplo, &n, matrix, &leading_dimension, eigenvalues, work,
                        &work_count, integer_work, &integer_work_count, &info);
  return info < 0 ? info - 1 : info;
}

XTBLOOM_PYODIDE_EXPORT void xtbloom_pyodide_cblas_dtrsm(
    int layout, int side, int triangle, int transpose, int diagonal, lapack_int rows,
    lapack_int columns, double alpha, const double* triangular_matrix,
    lapack_int leading_triangular, double* right_hand_side, lapack_int leading_rhs) {
  if (resolve_provider()) {
    (void)dispatch.dtrsm(layout, side, triangle, transpose, diagonal, rows, columns, alpha,
                         triangular_matrix, leading_triangular, right_hand_side, leading_rhs);
  }
}

XTBLOOM_PYODIDE_EXPORT void xtbloom_pyodide_cblas_dgemm(
    int layout, int transpose_left, int transpose_right, lapack_int rows, lapack_int columns,
    lapack_int inner, double alpha, const double* left, lapack_int leading_left,
    const double* right, lapack_int leading_right, double beta, double* result,
    lapack_int leading_result) {
  if (resolve_provider()) {
    (void)dispatch.dgemm(layout, transpose_left, transpose_right, rows, columns, inner, alpha, left,
                         leading_left, right, leading_right, beta, result, leading_result);
  }
}
