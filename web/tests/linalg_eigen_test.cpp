/* Focused ABI and numerical tests for the Eigen WebAssembly provider.
 *
 * The same translation units are compiled natively during development and by
 * Emscripten in the Web CI matrix. Reference operations below are deliberately
 * small scalar loops so provider mistakes are not compared against Eigen itself.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string_view>
#include <vector>

using lapack_int = std::int32_t;

extern "C" {
const char* openblas_get_config();
int openblas_set_num_threads_local(int num_threads);
void cblas_dgemm(int order, int transpose_left, int transpose_right, int m, int n, int k,
                 double alpha, const double* left, int left_leading_dimension, const double* right,
                 int right_leading_dimension, double beta, double* output,
                 int output_leading_dimension);
void cblas_dtrsm(int order, int side, int uplo, int transpose, int diagonal, int m, int n,
                 double alpha, const double* triangular, int triangular_leading_dimension,
                 double* right_hand_side, int right_hand_side_leading_dimension);
lapack_int LAPACKE_dpotrf_work(int matrix_layout, char uplo, lapack_int n, double* matrix,
                               lapack_int leading_dimension);
lapack_int LAPACKE_dpocon_work(int matrix_layout, char uplo, lapack_int n, const double* factor,
                               lapack_int leading_dimension, double matrix_one_norm,
                               double* reciprocal_condition, double* work,
                               lapack_int* integer_work);
lapack_int LAPACKE_dsyevd_work(int matrix_layout, char jobz, char uplo, lapack_int n,
                               double* matrix, lapack_int leading_dimension, double* eigenvalues,
                               double* work, lapack_int work_count, lapack_int* integer_work,
                               lapack_int integer_work_count);
}

namespace {

enum : int {
  kRowMajor = 101,
  kColMajor = 102,
  kNoTrans = 111,
  kTrans = 112,
  kConjTrans = 113,
  kUpper = 121,
  kLower = 122,
  kNonUnit = 131,
  kUnit = 132,
  kLeft = 141,
  kRight = 142,
};

#define CHECK(condition)                                                                   \
  do {                                                                                     \
    if (!(condition)) {                                                                    \
      std::fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
      return false;                                                                        \
    }                                                                                      \
  } while (false)

bool close(double left, double right, double tolerance = 2.0e-12) {
  return std::abs(left - right) <= tolerance * std::max({1.0, std::abs(left), std::abs(right)});
}

std::size_t index(int order, int leading_dimension, int row, int column) {
  return order == kColMajor
             ? static_cast<std::size_t>(row) + static_cast<std::size_t>(column) * leading_dimension
             : static_cast<std::size_t>(row) * leading_dimension + column;
}

double get(const std::vector<double>& matrix, int order, int leading_dimension, int row,
           int column) {
  return matrix[index(order, leading_dimension, row, column)];
}

void set(std::vector<double>& matrix, int order, int leading_dimension, int row, int column,
         double value) {
  matrix[index(order, leading_dimension, row, column)] = value;
}

std::vector<double> storage(int order, int rows, int columns, int& leading_dimension) {
  leading_dimension = (order == kColMajor ? rows : columns) + 2;
  const std::size_t elements = order == kColMajor
                                   ? static_cast<std::size_t>(leading_dimension) * columns
                                   : static_cast<std::size_t>(leading_dimension) * rows;
  return std::vector<double>(elements, std::numeric_limits<double>::quiet_NaN());
}

bool test_provider_identity() {
  CHECK(openblas_get_config() != nullptr);
  CHECK(std::string_view(openblas_get_config()).find("Eigen 5.0.1") != std::string_view::npos);
  CHECK(openblas_set_num_threads_local(1) == 1);
  CHECK(openblas_set_num_threads_local(8) == 1);
  return true;
}

bool test_cholesky_and_condition() {
  constexpr double original[3][3] = {
      {4.0, 1.0, 1.0},
      {1.0, 3.0, 0.5},
      {1.0, 0.5, 2.0},
  };
  for (const int order : {kRowMajor, kColMajor}) {
    for (const char uplo : {'L', 'U'}) {
      int leading_dimension = 0;
      std::vector<double> factor = storage(order, 3, 3, leading_dimension);
      for (int column = 0; column < 3; ++column) {
        for (int row = 0; row < 3; ++row) {
          if ((uplo == 'L' && row >= column) || (uplo == 'U' && row <= column)) {
            set(factor, order, leading_dimension, row, column, original[row][column]);
          }
        }
      }
      CHECK(LAPACKE_dpotrf_work(order, uplo, 3, factor.data(), leading_dimension) == 0);
      for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
          double reconstructed = 0.0;
          for (int inner = 0; inner < 3; ++inner) {
            if (uplo == 'L') {
              const double left =
                  inner <= row ? get(factor, order, leading_dimension, row, inner) : 0.0;
              const double right =
                  inner <= column ? get(factor, order, leading_dimension, column, inner) : 0.0;
              reconstructed += left * right;
            } else {
              const double left =
                  inner <= row ? get(factor, order, leading_dimension, inner, row) : 0.0;
              const double right =
                  inner <= column ? get(factor, order, leading_dimension, inner, column) : 0.0;
              reconstructed += left * right;
            }
          }
          CHECK(close(reconstructed, original[row][column]));
        }
      }
    }
  }

  double indefinite[] = {4.0, 0.0, 0.0, -1.0};
  CHECK(LAPACKE_dpotrf_work(kColMajor, 'L', 2, indefinite, 2) == 2);

  double matrix[] = {4.0, 1.0, 1.0, 3.0};
  CHECK(LAPACKE_dpotrf_work(kColMajor, 'L', 2, matrix, 2) == 0);
  double reciprocal_condition = 0.0;
  double work[6]{};
  lapack_int integer_work[2]{};
  CHECK(LAPACKE_dpocon_work(kColMajor, 'L', 2, matrix, 2, 5.0, &reciprocal_condition, work,
                            integer_work) == 0);
  CHECK(close(reciprocal_condition, 11.0 / 25.0));
  CHECK(LAPACKE_dpocon_work(kColMajor, 'L', 0, matrix, 1, 0.0, &reciprocal_condition, work,
                            integer_work) == 0);
  CHECK(reciprocal_condition == 1.0);
  return true;
}

bool check_eigensystem(int order, int leading_dimension, const std::vector<double>& vectors,
                       const double* eigenvalues) {
  constexpr double original[3][3] = {
      {2.0, 1.0, 0.0},
      {1.0, 2.0, 1.0},
      {0.0, 1.0, 2.0},
  };
  const double root_two = std::sqrt(2.0);
  CHECK(close(eigenvalues[0], 2.0 - root_two));
  CHECK(close(eigenvalues[1], 2.0));
  CHECK(close(eigenvalues[2], 2.0 + root_two));
  for (int first = 0; first < 3; ++first) {
    for (int second = 0; second < 3; ++second) {
      double dot = 0.0;
      for (int row = 0; row < 3; ++row) {
        dot += get(vectors, order, leading_dimension, row, first) *
               get(vectors, order, leading_dimension, row, second);
      }
      CHECK(close(dot, first == second ? 1.0 : 0.0));
    }
    for (int row = 0; row < 3; ++row) {
      double product = 0.0;
      for (int column = 0; column < 3; ++column) {
        product += original[row][column] * get(vectors, order, leading_dimension, column, first);
      }
      CHECK(
          close(product, eigenvalues[first] * get(vectors, order, leading_dimension, row, first)));
    }
  }
  return true;
}

bool test_self_adjoint_eigensolver() {
  for (const int order : {kRowMajor, kColMajor}) {
    for (const char uplo : {'L', 'U'}) {
      int leading_dimension = 0;
      std::vector<double> matrix = storage(order, 3, 3, leading_dimension);
      constexpr double original[3][3] = {
          {2.0, 1.0, 0.0},
          {1.0, 2.0, 1.0},
          {0.0, 1.0, 2.0},
      };
      for (int column = 0; column < 3; ++column) {
        for (int row = 0; row < 3; ++row) {
          if ((uplo == 'L' && row >= column) || (uplo == 'U' && row <= column)) {
            set(matrix, order, leading_dimension, row, column, original[row][column]);
          }
        }
      }
      double query = 0.0;
      lapack_int integer_query = 0;
      CHECK(LAPACKE_dsyevd_work(order, 'V', uplo, 3, matrix.data(), leading_dimension, nullptr,
                                &query, -1, &integer_query, -1) == 0);
      CHECK(query == 37.0);
      CHECK(integer_query == 18);
      query = -1.0;
      integer_query = -1;
      CHECK(LAPACKE_dsyevd_work(order, 'V', uplo, 3, matrix.data(), leading_dimension, nullptr,
                                &query, -1, &integer_query, 18) == 0);
      CHECK(query == 37.0);
      CHECK(integer_query == 18);
      query = -1.0;
      integer_query = -1;
      CHECK(LAPACKE_dsyevd_work(order, 'V', uplo, 3, matrix.data(), leading_dimension, nullptr,
                                &query, 37, &integer_query, -1) == 0);
      CHECK(query == 37.0);
      CHECK(integer_query == 18);
      std::vector<double> work(static_cast<std::size_t>(query));
      std::vector<lapack_int> integer_work(static_cast<std::size_t>(integer_query));
      double eigenvalues[3]{};
      CHECK(LAPACKE_dsyevd_work(order, 'V', uplo, 3, matrix.data(), leading_dimension, eigenvalues,
                                work.data(), static_cast<lapack_int>(work.size()),
                                integer_work.data(),
                                static_cast<lapack_int>(integer_work.size())) == 0);
      CHECK(check_eigensystem(order, leading_dimension, matrix, eigenvalues));
    }
  }

  double diagonal[] = {3.0, 0.0, 0.0, 1.0};
  double eigenvalues[2]{};
  double work[5]{};
  lapack_int integer_work[1]{};
  CHECK(LAPACKE_dsyevd_work(kColMajor, 'N', 'L', 2, diagonal, 2, eigenvalues, work, 5, integer_work,
                            1) == 0);
  CHECK(eigenvalues[0] == 1.0);
  CHECK(eigenvalues[1] == 3.0);
  CHECK(LAPACKE_dsyevd_work(kColMajor, 'V', 'L', 2, diagonal, 2, eigenvalues, work, 1, integer_work,
                            13) == -9);
  return true;
}

double logical_value(const std::vector<double>& matrix, int order, int leading_dimension,
                     int transpose, int row, int column) {
  return transpose == kNoTrans ? get(matrix, order, leading_dimension, row, column)
                               : get(matrix, order, leading_dimension, column, row);
}

bool test_gemm() {
  constexpr int m = 2;
  constexpr int n = 3;
  constexpr int k = 4;
  for (const int order : {kRowMajor, kColMajor}) {
    for (const int transpose_left : {kNoTrans, kTrans, kConjTrans}) {
      for (const int transpose_right : {kNoTrans, kTrans, kConjTrans}) {
        const int left_rows = transpose_left == kNoTrans ? m : k;
        const int left_columns = transpose_left == kNoTrans ? k : m;
        const int right_rows = transpose_right == kNoTrans ? k : n;
        const int right_columns = transpose_right == kNoTrans ? n : k;
        int left_leading = 0;
        int right_leading = 0;
        int output_leading = 0;
        std::vector<double> left = storage(order, left_rows, left_columns, left_leading);
        std::vector<double> right = storage(order, right_rows, right_columns, right_leading);
        std::vector<double> output = storage(order, m, n, output_leading);
        for (int column = 0; column < left_columns; ++column) {
          for (int row = 0; row < left_rows; ++row) {
            set(left, order, left_leading, row, column, 0.2 + 0.3 * row - 0.15 * column);
          }
        }
        for (int column = 0; column < right_columns; ++column) {
          for (int row = 0; row < right_rows; ++row) {
            set(right, order, right_leading, row, column, -0.4 + 0.12 * row + 0.25 * column);
          }
        }
        for (int column = 0; column < n; ++column) {
          for (int row = 0; row < m; ++row) {
            set(output, order, output_leading, row, column, 0.1 * (1 + row + column));
          }
        }
        std::vector<double> expected = output;
        for (int column = 0; column < n; ++column) {
          for (int row = 0; row < m; ++row) {
            double product = 0.0;
            for (int inner = 0; inner < k; ++inner) {
              product += logical_value(left, order, left_leading, transpose_left, row, inner) *
                         logical_value(right, order, right_leading, transpose_right, inner, column);
            }
            set(expected, order, output_leading, row, column,
                1.5 * product - 0.25 * get(output, order, output_leading, row, column));
          }
        }
        cblas_dgemm(order, transpose_left, transpose_right, m, n, k, 1.5, left.data(), left_leading,
                    right.data(), right_leading, -0.25, output.data(), output_leading);
        for (int column = 0; column < n; ++column) {
          for (int row = 0; row < m; ++row) {
            CHECK(close(get(output, order, output_leading, row, column),
                        get(expected, order, output_leading, row, column)));
          }
        }
      }
    }
  }
  return true;
}

bool test_trsm() {
  constexpr int m = 3;
  constexpr int n = 2;
  for (const int order : {kRowMajor, kColMajor}) {
    for (const int side : {kLeft, kRight}) {
      for (const int uplo : {kLower, kUpper}) {
        for (const int transpose : {kNoTrans, kTrans}) {
          for (const int diagonal : {kNonUnit, kUnit}) {
            const int dimension = side == kLeft ? m : n;
            int triangular_leading = 0;
            int rhs_leading = 0;
            std::vector<double> triangular =
                storage(order, dimension, dimension, triangular_leading);
            std::vector<double> rhs = storage(order, m, n, rhs_leading);
            std::vector<double> expected_x = storage(order, m, n, rhs_leading);
            for (int column = 0; column < dimension; ++column) {
              for (int row = 0; row < dimension; ++row) {
                if ((uplo == kLower && row >= column) || (uplo == kUpper && row <= column)) {
                  const double value = row == column ? (diagonal == kUnit ? 99.0 : 2.0 + row)
                                                     : 0.1 * (1 + row + column);
                  set(triangular, order, triangular_leading, row, column, value);
                }
              }
            }
            for (int column = 0; column < n; ++column) {
              for (int row = 0; row < m; ++row) {
                set(expected_x, order, rhs_leading, row, column, 0.5 + 0.2 * row - 0.1 * column);
              }
            }
            const auto op_a = [&](int row, int column) {
              const int source_row = transpose == kNoTrans ? row : column;
              const int source_column = transpose == kNoTrans ? column : row;
              if (source_row == source_column && diagonal == kUnit) return 1.0;
              const bool stored =
                  uplo == kLower ? source_row >= source_column : source_row <= source_column;
              return stored ? get(triangular, order, triangular_leading, source_row, source_column)
                            : 0.0;
            };
            for (int column = 0; column < n; ++column) {
              for (int row = 0; row < m; ++row) {
                double value = 0.0;
                if (side == kLeft) {
                  for (int inner = 0; inner < m; ++inner) {
                    value += op_a(row, inner) * get(expected_x, order, rhs_leading, inner, column);
                  }
                } else {
                  for (int inner = 0; inner < n; ++inner) {
                    value += get(expected_x, order, rhs_leading, row, inner) * op_a(inner, column);
                  }
                }
                set(rhs, order, rhs_leading, row, column, value);
              }
            }
            cblas_dtrsm(order, side, uplo, transpose, diagonal, m, n, 0.75, triangular.data(),
                        triangular_leading, rhs.data(), rhs_leading);
            for (int column = 0; column < n; ++column) {
              for (int row = 0; row < m; ++row) {
                CHECK(close(get(rhs, order, rhs_leading, row, column),
                            0.75 * get(expected_x, order, rhs_leading, row, column)));
              }
            }
          }
        }
      }
    }
  }

  double rhs[] = {1.0, 2.0, 3.0, 4.0};
  cblas_dtrsm(kColMajor, kLeft, kLower, kNoTrans, kNonUnit, 2, 2, 0.0, nullptr, 2, rhs, 2);
  CHECK(std::all_of(std::begin(rhs), std::end(rhs), [](double value) { return value == 0.0; }));
  return true;
}

}  // namespace

int main() {
  if (!test_provider_identity() || !test_cholesky_and_condition() ||
      !test_self_adjoint_eigensolver() || !test_gemm() || !test_trsm()) {
    return 1;
  }
  std::puts("Eigen LAPACKE/CBLAS compatibility tests passed");
  return 0;
}
