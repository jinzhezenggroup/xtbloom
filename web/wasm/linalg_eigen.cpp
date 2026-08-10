/*
 * Eigen-backed LP64 LAPACKE/CBLAS compatibility layer for the browser build.
 *
 * xTBloom's CPU eigensolver intentionally discovers one dlopen-able provider
 * with 32-bit BLAS/LAPACK integers. The browser keeps that production loader
 * path unchanged by preloading this target-width Emscripten side module as
 * ``libscipy_openblas.so``. Eigen supplies the numerical algorithms; these C
 * wrappers preserve only the small ABI surface xTBloom loads.
 *
 * Eigen 5.0.1 is vendored and hash-pinned under cmake/3rdparty/eigen. Its
 * upstream MPL-2.0, BSD, Apache-2.0, and embedded permissive notices are
 * retained with the source tree and the deployed browser artifact.
 */

#define EIGEN_NO_DEBUG

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

using lapack_int = std::int32_t;

namespace {

enum : int {
  kCblasRowMajor = 101,
  kCblasColMajor = 102,
  kCblasNoTrans = 111,
  kCblasTrans = 112,
  kCblasConjTrans = 113,
  kCblasUpper = 121,
  kCblasLower = 122,
  kCblasNonUnit = 131,
  kCblasUnit = 132,
  kCblasLeft = 141,
  kCblasRight = 142,
};

using ColumnMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

template <typename Matrix>
using ConstStridedMap =
    Eigen::Map<const Matrix, Eigen::Unaligned, Eigen::OuterStride<Eigen::Dynamic>>;

template <typename Matrix>
using StridedMap = Eigen::Map<Matrix, Eigen::Unaligned, Eigen::OuterStride<Eigen::Dynamic>>;

bool valid_layout(int layout) { return layout == kCblasRowMajor || layout == kCblasColMajor; }

bool valid_transpose(int transpose) {
  return transpose == kCblasNoTrans || transpose == kCblasTrans || transpose == kCblasConjTrans;
}

bool is_lower(char uplo) { return uplo == 'L' || uplo == 'l'; }

bool is_upper(char uplo) { return uplo == 'U' || uplo == 'u'; }

bool valid_uplo(char uplo) { return is_lower(uplo) || is_upper(uplo); }

double matrix_value(const double* matrix, int layout, lapack_int leading_dimension, lapack_int row,
                    lapack_int column) {
  if (layout == kCblasColMajor) {
    return matrix[static_cast<std::size_t>(row) +
                  static_cast<std::size_t>(column) * leading_dimension];
  }
  return matrix[static_cast<std::size_t>(row) * leading_dimension + column];
}

void set_matrix_value(double* matrix, int layout, lapack_int leading_dimension, lapack_int row,
                      lapack_int column, double value) {
  if (layout == kCblasColMajor) {
    matrix[static_cast<std::size_t>(row) + static_cast<std::size_t>(column) * leading_dimension] =
        value;
  } else {
    matrix[static_cast<std::size_t>(row) * leading_dimension + column] = value;
  }
}

ColumnMatrix copy_symmetric_triangle(int layout, char uplo, lapack_int n, const double* matrix,
                                     lapack_int leading_dimension) {
  ColumnMatrix symmetric(n, n);
  const bool lower = is_lower(uplo);
  for (lapack_int column = 0; column < n; ++column) {
    for (lapack_int row = 0; row < n; ++row) {
      const lapack_int source_row = lower ? std::max(row, column) : std::min(row, column);
      const lapack_int source_column = lower ? std::min(row, column) : std::max(row, column);
      symmetric(row, column) =
          matrix_value(matrix, layout, leading_dimension, source_row, source_column);
    }
  }
  return symmetric;
}

template <int Triangle>
lapack_int first_non_positive_leading_minor(const ColumnMatrix& matrix) {
  for (Eigen::Index extent = 1; extent <= matrix.rows(); ++extent) {
    Eigen::LLT<ColumnMatrix, Triangle> prefix;
    prefix.compute(matrix.topLeftCorner(extent, extent));
    if (prefix.info() != Eigen::Success) {
      return static_cast<lapack_int>(extent);
    }
  }
  return static_cast<lapack_int>(matrix.rows());
}

template <int Triangle>
lapack_int factor_cholesky(int layout, lapack_int n, double* matrix, lapack_int leading_dimension,
                           const ColumnMatrix& symmetric) {
  Eigen::LLT<ColumnMatrix, Triangle> factorization;
  factorization.compute(symmetric);
  const ColumnMatrix& packed = factorization.matrixLLT();
  if (factorization.info() != Eigen::Success) {
    /* Eigen reports a numerical issue but not LAPACK's leading-minor index.
     * Re-run only the failure path on prefixes so callers retain the positive
     * DPotrf ``info`` contract without replacing the numerical algorithm. */
    return first_non_positive_leading_minor<Triangle>(symmetric);
  }

  for (lapack_int column = 0; column < n; ++column) {
    for (lapack_int row = 0; row < n; ++row) {
      if constexpr (Triangle == Eigen::Lower) {
        if (row >= column) {
          set_matrix_value(matrix, layout, leading_dimension, row, column, packed(row, column));
        }
      } else if (row <= column) {
        set_matrix_value(matrix, layout, leading_dimension, row, column, packed(row, column));
      }
    }
  }
  return 0;
}

ColumnMatrix copy_triangular_factor(int layout, char uplo, lapack_int n, const double* matrix,
                                    lapack_int leading_dimension) {
  ColumnMatrix factor = ColumnMatrix::Zero(n, n);
  const bool lower = is_lower(uplo);
  for (lapack_int column = 0; column < n; ++column) {
    for (lapack_int row = 0; row < n; ++row) {
      if ((lower && row >= column) || (!lower && row <= column)) {
        factor(row, column) = matrix_value(matrix, layout, leading_dimension, row, column);
      }
    }
  }
  return factor;
}

template <typename Output, typename Left, typename Right>
void assign_gemm(Output& output, const Left& left, const Right& right, double alpha, double beta) {
  if (alpha == 0.0) {
    if (beta == 0.0) {
      output.setZero();
    } else if (beta != 1.0) {
      output *= beta;
    }
    return;
  }
  if (beta == 0.0) {
    output.noalias() = alpha * (left * right);
  } else {
    output *= beta;
    output.noalias() += alpha * (left * right);
  }
}

template <typename Matrix, bool TransposeLeft, bool TransposeRight>
void gemm_mapped(int m, int n, int k, double alpha, const double* left, int left_leading_dimension,
                 const double* right, int right_leading_dimension, double beta, double* output,
                 int output_leading_dimension) {
  const Eigen::Index left_rows = TransposeLeft ? k : m;
  const Eigen::Index left_columns = TransposeLeft ? m : k;
  const Eigen::Index right_rows = TransposeRight ? n : k;
  const Eigen::Index right_columns = TransposeRight ? k : n;
  ConstStridedMap<Matrix> left_map(left, left_rows, left_columns,
                                   Eigen::OuterStride<Eigen::Dynamic>(left_leading_dimension));
  ConstStridedMap<Matrix> right_map(right, right_rows, right_columns,
                                    Eigen::OuterStride<Eigen::Dynamic>(right_leading_dimension));
  StridedMap<Matrix> output_map(output, m, n,
                                Eigen::OuterStride<Eigen::Dynamic>(output_leading_dimension));

  if constexpr (TransposeLeft && TransposeRight) {
    assign_gemm(output_map, left_map.transpose(), right_map.transpose(), alpha, beta);
  } else if constexpr (TransposeLeft) {
    assign_gemm(output_map, left_map.transpose(), right_map, alpha, beta);
  } else if constexpr (TransposeRight) {
    assign_gemm(output_map, left_map, right_map.transpose(), alpha, beta);
  } else {
    assign_gemm(output_map, left_map, right_map, alpha, beta);
  }
}

template <typename Matrix>
void dispatch_gemm_transposes(int transpose_left, int transpose_right, int m, int n, int k,
                              double alpha, const double* left, int left_leading_dimension,
                              const double* right, int right_leading_dimension, double beta,
                              double* output, int output_leading_dimension) {
  const bool left_is_transposed = transpose_left != kCblasNoTrans;
  const bool right_is_transposed = transpose_right != kCblasNoTrans;
  if (left_is_transposed && right_is_transposed) {
    gemm_mapped<Matrix, true, true>(m, n, k, alpha, left, left_leading_dimension, right,
                                    right_leading_dimension, beta, output,
                                    output_leading_dimension);
  } else if (left_is_transposed) {
    gemm_mapped<Matrix, true, false>(m, n, k, alpha, left, left_leading_dimension, right,
                                     right_leading_dimension, beta, output,
                                     output_leading_dimension);
  } else if (right_is_transposed) {
    gemm_mapped<Matrix, false, true>(m, n, k, alpha, left, left_leading_dimension, right,
                                     right_leading_dimension, beta, output,
                                     output_leading_dimension);
  } else {
    gemm_mapped<Matrix, false, false>(m, n, k, alpha, left, left_leading_dimension, right,
                                      right_leading_dimension, beta, output,
                                      output_leading_dimension);
  }
}

template <int Side, int Triangle, typename Triangular, typename RightHandSide>
void solve_triangular(const Triangular& triangular, RightHandSide& right_hand_side) {
  triangular.template triangularView<Triangle>().template solveInPlace<Side>(right_hand_side);
}

template <typename Matrix, int Side, bool Lower, bool Transpose, bool UnitDiagonal>
void trsm_mapped(int m, int n, double alpha, const double* triangular,
                 int triangular_leading_dimension, double* right_hand_side,
                 int right_hand_side_leading_dimension) {
  const Eigen::Index dimension = Side == Eigen::OnTheLeft ? m : n;
  ConstStridedMap<Matrix> triangular_map(
      triangular, dimension, dimension,
      Eigen::OuterStride<Eigen::Dynamic>(triangular_leading_dimension));
  StridedMap<Matrix> right_hand_side_map(
      right_hand_side, m, n, Eigen::OuterStride<Eigen::Dynamic>(right_hand_side_leading_dimension));
  if (alpha != 1.0) {
    right_hand_side_map *= alpha;
  }

  constexpr bool effective_lower = Lower != Transpose;
  constexpr int triangle = effective_lower ? (UnitDiagonal ? Eigen::UnitLower : Eigen::Lower)
                                           : (UnitDiagonal ? Eigen::UnitUpper : Eigen::Upper);
  if constexpr (Transpose) {
    solve_triangular<Side, triangle>(triangular_map.transpose(), right_hand_side_map);
  } else {
    solve_triangular<Side, triangle>(triangular_map, right_hand_side_map);
  }
}

template <typename Matrix, int Side, bool Lower, bool Transpose>
void dispatch_trsm_diagonal(int diagonal, int m, int n, double alpha, const double* triangular,
                            int triangular_leading_dimension, double* right_hand_side,
                            int right_hand_side_leading_dimension) {
  if (diagonal == kCblasUnit) {
    trsm_mapped<Matrix, Side, Lower, Transpose, true>(m, n, alpha, triangular,
                                                      triangular_leading_dimension, right_hand_side,
                                                      right_hand_side_leading_dimension);
  } else {
    trsm_mapped<Matrix, Side, Lower, Transpose, false>(
        m, n, alpha, triangular, triangular_leading_dimension, right_hand_side,
        right_hand_side_leading_dimension);
  }
}

template <typename Matrix, int Side, bool Lower>
void dispatch_trsm_transpose(int transpose, int diagonal, int m, int n, double alpha,
                             const double* triangular, int triangular_leading_dimension,
                             double* right_hand_side, int right_hand_side_leading_dimension) {
  if (transpose == kCblasNoTrans) {
    dispatch_trsm_diagonal<Matrix, Side, Lower, false>(
        diagonal, m, n, alpha, triangular, triangular_leading_dimension, right_hand_side,
        right_hand_side_leading_dimension);
  } else {
    dispatch_trsm_diagonal<Matrix, Side, Lower, true>(diagonal, m, n, alpha, triangular,
                                                      triangular_leading_dimension, right_hand_side,
                                                      right_hand_side_leading_dimension);
  }
}

template <typename Matrix, int Side>
void dispatch_trsm_triangle(int uplo, int transpose, int diagonal, int m, int n, double alpha,
                            const double* triangular, int triangular_leading_dimension,
                            double* right_hand_side, int right_hand_side_leading_dimension) {
  if (uplo == kCblasLower) {
    dispatch_trsm_transpose<Matrix, Side, true>(transpose, diagonal, m, n, alpha, triangular,
                                                triangular_leading_dimension, right_hand_side,
                                                right_hand_side_leading_dimension);
  } else {
    dispatch_trsm_transpose<Matrix, Side, false>(transpose, diagonal, m, n, alpha, triangular,
                                                 triangular_leading_dimension, right_hand_side,
                                                 right_hand_side_leading_dimension);
  }
}

template <typename Matrix>
void dispatch_trsm_side(int side, int uplo, int transpose, int diagonal, int m, int n, double alpha,
                        const double* triangular, int triangular_leading_dimension,
                        double* right_hand_side, int right_hand_side_leading_dimension) {
  if (side == kCblasLeft) {
    dispatch_trsm_triangle<Matrix, Eigen::OnTheLeft>(
        uplo, transpose, diagonal, m, n, alpha, triangular, triangular_leading_dimension,
        right_hand_side, right_hand_side_leading_dimension);
  } else {
    dispatch_trsm_triangle<Matrix, Eigen::OnTheRight>(
        uplo, transpose, diagonal, m, n, alpha, triangular, triangular_leading_dimension,
        right_hand_side, right_hand_side_leading_dimension);
  }
}

void zero_matrix(int layout, int rows, int columns, double* matrix, int leading_dimension) {
  for (int column = 0; column < columns; ++column) {
    for (int row = 0; row < rows; ++row) {
      if (layout == kCblasColMajor) {
        matrix[static_cast<std::size_t>(row) +
               static_cast<std::size_t>(column) * leading_dimension] = 0.0;
      } else {
        matrix[static_cast<std::size_t>(row) * leading_dimension + column] = 0.0;
      }
    }
  }
}

}  // namespace

extern "C" {

const char* openblas_get_config() {
  /* The runtime factory requires a non-null LP64 provider identity and rejects
   * OpenBLAS-style ``USE64BITINT`` markers. This compatibility string states
   * the actual implementation without pretending to be an OpenBLAS build. */
  return "Eigen 5.0.1 WebAssembly LP64 compatibility provider (single threaded)";
}

int openblas_set_num_threads_local(int /*num_threads*/) {
  /* The browser build has no pthreads. Report the only effective setting so
   * xTBloom's scoped one-thread guard restores a stable previous value. */
  return 1;
}

void cblas_dgemm(int order, int transpose_left, int transpose_right, int m, int n, int k,
                 double alpha, const double* left, int left_leading_dimension, const double* right,
                 int right_leading_dimension, double beta, double* output,
                 int output_leading_dimension) {
  if (!valid_layout(order) || !valid_transpose(transpose_left) ||
      !valid_transpose(transpose_right) || m < 0 || n < 0 || k < 0 || output == nullptr ||
      (k > 0 && (left == nullptr || right == nullptr))) {
    return;
  }
  const int left_rows = transpose_left == kCblasNoTrans ? m : k;
  const int left_columns = transpose_left == kCblasNoTrans ? k : m;
  const int right_rows = transpose_right == kCblasNoTrans ? k : n;
  const int right_columns = transpose_right == kCblasNoTrans ? n : k;
  const int minimum_left_leading =
      order == kCblasColMajor ? std::max(1, left_rows) : std::max(1, left_columns);
  const int minimum_right_leading =
      order == kCblasColMajor ? std::max(1, right_rows) : std::max(1, right_columns);
  const int minimum_output_leading = order == kCblasColMajor ? std::max(1, m) : std::max(1, n);
  if (left_leading_dimension < minimum_left_leading ||
      right_leading_dimension < minimum_right_leading ||
      output_leading_dimension < minimum_output_leading || m == 0 || n == 0) {
    return;
  }
  if (k == 0) {
    if (beta == 0.0) {
      zero_matrix(order, m, n, output, output_leading_dimension);
    } else if (beta != 1.0) {
      if (order == kCblasColMajor) {
        StridedMap<ColumnMatrix> output_map(
            output, m, n, Eigen::OuterStride<Eigen::Dynamic>(output_leading_dimension));
        output_map *= beta;
      } else {
        StridedMap<RowMatrix> output_map(
            output, m, n, Eigen::OuterStride<Eigen::Dynamic>(output_leading_dimension));
        output_map *= beta;
      }
    }
    return;
  }
  if (order == kCblasColMajor) {
    dispatch_gemm_transposes<ColumnMatrix>(transpose_left, transpose_right, m, n, k, alpha, left,
                                           left_leading_dimension, right, right_leading_dimension,
                                           beta, output, output_leading_dimension);
  } else {
    dispatch_gemm_transposes<RowMatrix>(transpose_left, transpose_right, m, n, k, alpha, left,
                                        left_leading_dimension, right, right_leading_dimension,
                                        beta, output, output_leading_dimension);
  }
}

void cblas_dtrsm(int order, int side, int uplo, int transpose, int diagonal, int m, int n,
                 double alpha, const double* triangular, int triangular_leading_dimension,
                 double* right_hand_side, int right_hand_side_leading_dimension) {
  if (!valid_layout(order) || (side != kCblasLeft && side != kCblasRight) ||
      (uplo != kCblasLower && uplo != kCblasUpper) || !valid_transpose(transpose) ||
      (diagonal != kCblasNonUnit && diagonal != kCblasUnit) || m < 0 || n < 0 ||
      right_hand_side == nullptr || m == 0 || n == 0) {
    return;
  }
  const int dimension = side == kCblasLeft ? m : n;
  const int minimum_triangular_leading = std::max(1, dimension);
  const int minimum_rhs_leading = order == kCblasColMajor ? std::max(1, m) : std::max(1, n);
  if (triangular_leading_dimension < minimum_triangular_leading ||
      right_hand_side_leading_dimension < minimum_rhs_leading) {
    return;
  }
  if (alpha == 0.0) {
    zero_matrix(order, m, n, right_hand_side, right_hand_side_leading_dimension);
    return;
  }
  if (triangular == nullptr) {
    return;
  }
  if (order == kCblasColMajor) {
    dispatch_trsm_side<ColumnMatrix>(side, uplo, transpose, diagonal, m, n, alpha, triangular,
                                     triangular_leading_dimension, right_hand_side,
                                     right_hand_side_leading_dimension);
  } else {
    dispatch_trsm_side<RowMatrix>(side, uplo, transpose, diagonal, m, n, alpha, triangular,
                                  triangular_leading_dimension, right_hand_side,
                                  right_hand_side_leading_dimension);
  }
}

lapack_int LAPACKE_dpotrf_work(int matrix_layout, char uplo, lapack_int n, double* matrix,
                               lapack_int leading_dimension) {
  if (!valid_layout(matrix_layout)) return -1;
  if (!valid_uplo(uplo)) return -2;
  if (n < 0) return -3;
  if (matrix == nullptr && n > 0) return -4;
  if (leading_dimension < std::max<lapack_int>(1, n)) return -5;
  if (n == 0) return 0;

  const ColumnMatrix symmetric =
      copy_symmetric_triangle(matrix_layout, uplo, n, matrix, leading_dimension);
  return is_lower(uplo)
             ? factor_cholesky<Eigen::Lower>(matrix_layout, n, matrix, leading_dimension, symmetric)
             : factor_cholesky<Eigen::Upper>(matrix_layout, n, matrix, leading_dimension,
                                             symmetric);
}

lapack_int LAPACKE_dpocon_work(int matrix_layout, char uplo, lapack_int n,
                               const double* factor_data, lapack_int leading_dimension,
                               double matrix_one_norm, double* reciprocal_condition, double* work,
                               lapack_int* integer_work) {
  static_cast<void>(work);
  static_cast<void>(integer_work);
  if (!valid_layout(matrix_layout)) return -1;
  if (!valid_uplo(uplo)) return -2;
  if (n < 0) return -3;
  if (factor_data == nullptr && n > 0) return -4;
  if (leading_dimension < std::max<lapack_int>(1, n)) return -5;
  if (!(matrix_one_norm >= 0.0) || !std::isfinite(matrix_one_norm)) return -6;
  if (reciprocal_condition == nullptr) return -7;
  if (n == 0) {
    *reciprocal_condition = 1.0;
    return 0;
  }

  const ColumnMatrix factor =
      copy_triangular_factor(matrix_layout, uplo, n, factor_data, leading_dimension);
  ColumnMatrix inverse = ColumnMatrix::Identity(n, n);
  if (is_lower(uplo)) {
    factor.template triangularView<Eigen::Lower>().solveInPlace(inverse);
    factor.transpose().template triangularView<Eigen::Upper>().solveInPlace(inverse);
  } else {
    factor.transpose().template triangularView<Eigen::Lower>().solveInPlace(inverse);
    factor.template triangularView<Eigen::Upper>().solveInPlace(inverse);
  }
  if (matrix_one_norm == 0.0 || !inverse.allFinite()) {
    *reciprocal_condition = 0.0;
    return 0;
  }
  const double inverse_one_norm = inverse.cwiseAbs().colwise().sum().maxCoeff();
  const double denominator = matrix_one_norm * inverse_one_norm;
  *reciprocal_condition = denominator > 0.0 && std::isfinite(denominator) ? 1.0 / denominator : 0.0;
  return 0;
}

lapack_int LAPACKE_dsyevd_work(int matrix_layout, char jobz, char uplo, lapack_int n,
                               double* matrix, lapack_int leading_dimension, double* eigenvalues,
                               double* work, lapack_int work_count, lapack_int* integer_work,
                               lapack_int integer_work_count) {
  if (!valid_layout(matrix_layout)) return -1;
  const bool vectors = jobz == 'V' || jobz == 'v';
  const bool values_only = jobz == 'N' || jobz == 'n';
  if (!vectors && !values_only) return -2;
  if (!valid_uplo(uplo)) return -3;
  if (n < 0) return -4;
  if (matrix == nullptr && n > 0) return -5;
  if (leading_dimension < std::max<lapack_int>(1, n)) return -6;

  const std::int64_t dimension = n;
  const std::int64_t minimum_work =
      n <= 1 ? 1 : (vectors ? 1 + 6 * dimension + 2 * dimension * dimension : 2 * dimension + 1);
  const std::int64_t minimum_integer_work = n <= 1 ? 1 : (vectors ? 3 + 5 * dimension : 1);
  const bool query = work_count == -1 || integer_work_count == -1;
  if (query) {
    /* LAPACK treats either -1 count as one combined workspace query and
     * returns both recommendations when the output pointers are present. */
    if (work != nullptr) {
      work[0] = static_cast<double>(minimum_work);
    }
    if (integer_work != nullptr) {
      integer_work[0] = static_cast<lapack_int>(minimum_integer_work);
    }
    return 0;
  }
  if (eigenvalues == nullptr && n > 0) return -7;
  if (work == nullptr || work_count < minimum_work) return -9;
  if (integer_work == nullptr || integer_work_count < minimum_integer_work) return -11;
  if (n == 0) return 0;

  const ColumnMatrix symmetric =
      copy_symmetric_triangle(matrix_layout, uplo, n, matrix, leading_dimension);
  Eigen::SelfAdjointEigenSolver<ColumnMatrix> solver;
  solver.compute(symmetric, vectors ? Eigen::ComputeEigenvectors : Eigen::EigenvaluesOnly);
  if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) {
    return 1;
  }
  for (lapack_int index = 0; index < n; ++index) {
    eigenvalues[index] = solver.eigenvalues()[index];
  }
  if (vectors) {
    if (!solver.eigenvectors().allFinite()) return 1;
    for (lapack_int column = 0; column < n; ++column) {
      for (lapack_int row = 0; row < n; ++row) {
        set_matrix_value(matrix, matrix_layout, leading_dimension, row, column,
                         solver.eigenvectors()(row, column));
      }
    }
  }
  return 0;
}

}  // extern "C"
