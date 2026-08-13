// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include "model/common/sto.hpp"

#include <array>
#include <cmath>

#include "data/parameters/tblite_sto.hpp"

namespace xtbloom::detail::common {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoOverPi = 2.0 / kPi;
constexpr std::array<double, 5> kDoubleFactorial{1.0, 1.0, 3.0, 15.0, 105.0};

}  // namespace

bool sto_table(std::uint8_t n, std::uint8_t l, std::uint8_t ng, const double*& alpha,
               const double*& coefficients) noexcept {
  if (n == 6u && ng == 6u) {
    if (l == 0u) {
      alpha = parameters::tblite::kAlpha6s.data();
      coefficients = parameters::tblite::kCoeff6s.data();
      return true;
    }
    if (l == 1u) {
      alpha = parameters::tblite::kAlpha6p.data();
      coefficients = parameters::tblite::kCoeff6p.data();
      return true;
    }
    return false;
  }
  if (n == 0u || n > 5u || l >= n || l > 4u) {
    return false;
  }

  std::size_t type = 0;
  if (l == 0u) {
    type = static_cast<std::size_t>(n - 1u);
  } else if (l == 1u) {
    type = static_cast<std::size_t>(n + 3u);
  } else if (l == 2u) {
    type = static_cast<std::size_t>(n + 6u);
  } else if (l == 3u) {
    type = static_cast<std::size_t>(n + 8u);
  } else {
    type = static_cast<std::size_t>(n + 9u);
  }
  if (type >= parameters::tblite::kAlpha3.size()) {
    return false;
  }
  if (ng == 3u) {
    alpha = parameters::tblite::kAlpha3[type].data();
    coefficients = parameters::tblite::kCoeff3[type].data();
    return true;
  }
  if (ng == 4u) {
    alpha = parameters::tblite::kAlpha4[type].data();
    coefficients = parameters::tblite::kCoeff4[type].data();
    return true;
  }
  if (ng == 6u) {
    alpha = parameters::tblite::kAlpha6[type].data();
    coefficients = parameters::tblite::kCoeff6[type].data();
    return true;
  }
  return false;
}

void expand_sto_shell(std::uint8_t n, std::uint8_t l, std::uint8_t ng, double slater,
                      double* alpha, double* coefficients) noexcept {
  const double* base_alpha = nullptr;
  const double* base_coefficients = nullptr;
  (void)sto_table(n, l, ng, base_alpha, base_coefficients);

  const double zeta_squared = slater * slater;
  for (std::size_t primitive = 0; primitive < ng; ++primitive) {
    alpha[primitive] = base_alpha[primitive] * zeta_squared;
    const double normalization =
        std::pow(kTwoOverPi * alpha[primitive], 0.75) *
        std::pow(std::sqrt(4.0 * alpha[primitive]), static_cast<double>(l)) /
        std::sqrt(kDoubleFactorial[l]);
    coefficients[primitive] = base_coefficients[primitive] * normalization;
  }
}

void orthogonalize_to_first(const double* first_alpha, const double* first_coefficients,
                            std::size_t first_count, double* alpha, double* coefficients,
                            std::size_t base_count) noexcept {
  double overlap = 0.0;
  for (std::size_t first = 0; first < first_count; ++first) {
    for (std::size_t second = 0; second < base_count; ++second) {
      const double exponent_sum = first_alpha[first] + alpha[second];
      const double primitive_overlap = std::pow(std::sqrt(kPi / exponent_sum), 3.0);
      overlap += first_coefficients[first] * coefficients[second] * primitive_overlap;
    }
  }
  for (std::size_t primitive = 0; primitive < first_count; ++primitive) {
    alpha[base_count + primitive] = first_alpha[primitive];
    coefficients[base_count + primitive] = -overlap * first_coefficients[primitive];
  }

  const std::size_t count = base_count + first_count;
  double norm_squared = 0.0;
  for (std::size_t first = 0; first < count; ++first) {
    for (std::size_t second = 0; second < count; ++second) {
      const double exponent_sum = alpha[first] + alpha[second];
      const double primitive_overlap = std::pow(std::sqrt(kPi / exponent_sum), 3.0);
      norm_squared += coefficients[first] * coefficients[second] * primitive_overlap;
    }
  }
  const double inverse_norm = 1.0 / std::sqrt(norm_squared);
  for (std::size_t primitive = 0; primitive < count; ++primitive) {
    coefficients[primitive] *= inverse_norm;
  }
}

}  // namespace xtbloom::detail::common
