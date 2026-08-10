// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.
/* Verify column-major LAPACKE translation and exact-provider CBLAS dispatch. */

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int32_t lapack_int;

const char* xtbloom_pyodide_openblas_get_config(void);
int xtbloom_pyodide_openblas_set_num_threads_local(int threads);
lapack_int xtbloom_pyodide_LAPACKE_dpotrf_work(int, char, lapack_int, double*, lapack_int);
lapack_int xtbloom_pyodide_LAPACKE_dpocon_work(int, char, lapack_int, const double*, lapack_int,
                                               double, double*, double*, lapack_int*);
lapack_int xtbloom_pyodide_LAPACKE_dsyevd_work(int, char, char, lapack_int, double*, lapack_int,
                                               double*, double*, lapack_int, lapack_int*,
                                               lapack_int);
void xtbloom_pyodide_cblas_dtrsm(int, int, int, int, int, lapack_int, lapack_int, double,
                                 const double*, lapack_int, double*, lapack_int);
void xtbloom_pyodide_cblas_dgemm(int, int, int, lapack_int, lapack_int, lapack_int, double,
                                 const double*, lapack_int, const double*, lapack_int, double,
                                 double*, lapack_int);

int main(void) {
  double matrix[1] = {4.0};
  double eigenvalue[1] = {0.0};
  double work[9] = {0.0};
  lapack_int integer_work[8] = {0};
  double reciprocal_condition = 0.0;
  double rhs[1] = {4.0};
  double product[1] = {0.0};
  const double triangular[1] = {2.0};

  if (getenv("XTBLOOM_EXPECT_PYODIDE_PROVIDER_UNAVAILABLE") != NULL) {
    assert(xtbloom_pyodide_openblas_get_config() == NULL);
    assert(xtbloom_pyodide_LAPACKE_dpotrf_work(102, 'L', 1, matrix, 1) == -1024);
    return 0;
  }

  assert(strncmp(xtbloom_pyodide_openblas_get_config(), "OpenBLAS 0.3.28", 15) == 0);
  assert(xtbloom_pyodide_openblas_set_num_threads_local(1) == 1);
  assert(xtbloom_pyodide_LAPACKE_dpotrf_work(101, 'L', 1, matrix, 1) == -1);
  assert(xtbloom_pyodide_LAPACKE_dpotrf_work(102, 'L', 1, matrix, 1) == 0);
  assert(matrix[0] == 2.0);
  assert(xtbloom_pyodide_LAPACKE_dpocon_work(102, 'L', 1, matrix, 1, 4.0, &reciprocal_condition,
                                             work, integer_work) == 0);
  assert(reciprocal_condition == 0.5);
  assert(xtbloom_pyodide_LAPACKE_dsyevd_work(102, 'V', 'L', 1, matrix, 1, eigenvalue, work, 9,
                                             integer_work, 8) == 0);
  assert(matrix[0] == 1.0 && eigenvalue[0] == 2.0);

  xtbloom_pyodide_cblas_dtrsm(102, 141, 122, 111, 131, 1, 1, 1.0, triangular, 1, rhs, 1);
  xtbloom_pyodide_cblas_dgemm(102, 111, 112, 1, 1, 1, 1.0, rhs, 1, rhs, 1, 0.0, product, 1);
  assert(rhs[0] == 2.0 && product[0] == 4.0);
  return 0;
}
