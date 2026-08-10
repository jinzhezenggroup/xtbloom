// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.
#include <stdint.h>

typedef int32_t lapack_int;

const char* openblas_get_config(void) { return "OpenBLAS 0.3.28 TEST"; }

int openblas_set_num_threads_local(int threads) { return threads; }

int dpotrf_(const char* uplo, const lapack_int* n, double* matrix,
            const lapack_int* leading_dimension, lapack_int* info) {
  (void)uplo;
  (void)n;
  (void)leading_dimension;
  matrix[0] = 2.0;
  *info = 0;
  return 0;
}

int dpocon_(const char* uplo, const lapack_int* n, const double* factor,
            const lapack_int* leading_dimension, const double* matrix_one_norm,
            double* reciprocal_condition, double* work, lapack_int* integer_work,
            lapack_int* info) {
  (void)uplo;
  (void)n;
  (void)leading_dimension;
  (void)matrix_one_norm;
  (void)work;
  (void)integer_work;
  *reciprocal_condition = factor[0] / 4.0;
  *info = 0;
  return 0;
}

int dsyevd_(const char* job_vectors, const char* uplo, const lapack_int* n, double* matrix,
            const lapack_int* leading_dimension, double* eigenvalues, double* work,
            const lapack_int* work_count, lapack_int* integer_work,
            const lapack_int* integer_work_count, lapack_int* info) {
  (void)job_vectors;
  (void)uplo;
  (void)n;
  (void)leading_dimension;
  (void)work;
  (void)work_count;
  (void)integer_work;
  (void)integer_work_count;
  eigenvalues[0] = matrix[0];
  matrix[0] = 1.0;
  *info = 0;
  return 0;
}

int cblas_dtrsm(int layout, int side, int triangle, int transpose, int diagonal, lapack_int rows,
                lapack_int columns, double alpha, const double* triangular_matrix,
                lapack_int leading_triangular, double* right_hand_side, lapack_int leading_rhs) {
  (void)layout;
  (void)side;
  (void)triangle;
  (void)transpose;
  (void)diagonal;
  (void)rows;
  (void)columns;
  (void)leading_triangular;
  (void)leading_rhs;
  right_hand_side[0] = alpha * right_hand_side[0] / triangular_matrix[0];
  return 0;
}

int cblas_dgemm(int layout, int transpose_left, int transpose_right, lapack_int rows,
                lapack_int columns, lapack_int inner, double alpha, const double* left,
                lapack_int leading_left, const double* right, lapack_int leading_right, double beta,
                double* result, lapack_int leading_result) {
  (void)layout;
  (void)transpose_left;
  (void)transpose_right;
  (void)rows;
  (void)columns;
  (void)inner;
  (void)leading_left;
  (void)leading_right;
  (void)leading_result;
  result[0] = alpha * left[0] * right[0] + beta * result[0];
  return 0;
}
