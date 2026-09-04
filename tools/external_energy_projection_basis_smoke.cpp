// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

/*
 * Focused source-tree checks for the auxiliary projection basis. This
 * executable deliberately uses internal headers, is built only through an
 * explicit option, and is never installed or included in source distributions.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

#include "model/common/integrals.hpp"
#include "model/common/projection_basis.hpp"
#include "model/gfn2/basis.hpp"

namespace {

bool close_enough(double value, double tolerance) { return std::abs(value) <= tolerance; }

int fail(const char* message) {
  std::fprintf(stderr, "projection smoke: %s\n", message);
  return 1;
}

}  // namespace

int main() {
  using xtbloom::detail::common::BasisPlan;
  using xtbloom::detail::common::IntegralPlan;
  using xtbloom::detail::common::IntegralWorkspace;

  BasisPlan projection;
  std::string error;
  BasisPlan rejected_projection;
  if (xtbloom::detail::common::make_external_projection_basis(0, rejected_projection, error) !=
          XTBLOOM_STATUS_INVALID_ARGUMENT ||
      xtbloom::detail::common::make_external_projection_basis(
          std::numeric_limits<std::int64_t>::max(), rejected_projection, error) !=
          XTBLOOM_STATUS_INVALID_ARGUMENT) {
    return fail("projection basis accepted an invalid atom count");
  }
  if (xtbloom::detail::common::make_external_projection_basis(2, projection, error) !=
      XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "projection smoke: basis construction failed: %s\n", error.c_str());
    return 1;
  }
  if (projection.batch_size != 1 || projection.total_atoms != 2 || projection.total_shells != 72 ||
      projection.total_orbitals != 216 || projection.total_cartesian_orbitals != 240) {
    return fail("unexpected fixed-basis dimensions");
  }
  if (projection.atom_shell_offsets.size() != 3 || projection.atom_orbital_offsets.size() != 3 ||
      projection.shell_orbital_offsets.size() != 73) {
    return fail("projection offset vectors have unexpected sizes");
  }
  for (std::int64_t atom = 0; atom < 2; ++atom) {
    const std::size_t atom_index = static_cast<std::size_t>(atom);
    if (projection.atom_shell_offsets[atom_index + 1u] -
                projection.atom_shell_offsets[atom_index] !=
            36 ||
        projection.atom_orbital_offsets[atom_index + 1u] -
                projection.atom_orbital_offsets[atom_index] !=
            108) {
      return fail("each atom must contain 36 shells and 108 orbitals");
    }
    const std::int64_t shell_begin = projection.atom_shell_offsets[atom_index];
    for (std::int64_t shell = 0; shell < 36; ++shell) {
      const std::size_t index = static_cast<std::size_t>(shell_begin + shell);
      const std::uint8_t angular_momentum = projection.angular_momenta[index];
      const std::int64_t width =
          projection.shell_orbital_offsets[index + 1u] - projection.shell_orbital_offsets[index];
      const std::int64_t expected_width = 2 * static_cast<std::int64_t>(angular_momentum) + 1;
      if (angular_momentum > 2u || width != expected_width ||
          projection.shell_to_atom[index] != atom) {
        return fail("shell metadata does not match the fixed s/p/d layout");
      }
      const std::int64_t primitive_begin = projection.shell_primitive_offsets[index];
      const std::int64_t primitive_end = projection.shell_primitive_offsets[index + 1u];
      if (primitive_end - primitive_begin != 12) {
        return fail("every projection shell must contain twelve primitives");
      }
      for (std::int64_t primitive = primitive_begin; primitive < primitive_end; ++primitive) {
        const std::size_t primitive_index = static_cast<std::size_t>(primitive);
        if (!(projection.primitive_exponents[primitive_index] > 0.0) ||
            !std::isfinite(projection.primitive_exponents[primitive_index]) ||
            !std::isfinite(projection.primitive_coefficients[primitive_index])) {
          return fail("projection primitive data are not finite and positive");
        }
      }
    }
  }

  const std::int64_t atom_offsets[] = {0, 2};
  const std::int32_t atomic_numbers[] = {1, 8};
  BasisPlan native;
  if (xtbloom::detail::gfn2::make_basis_plan(1, 2, atom_offsets, atomic_numbers, native, error) !=
      XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "projection smoke: native basis construction failed: %s\n", error.c_str());
    return 1;
  }
  IntegralPlan native_integrals;
  if (xtbloom::detail::common::make_integral_plan(native, native_integrals, error) !=
      XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "projection smoke: native integral plan failed: %s\n", error.c_str());
    return 1;
  }
  IntegralWorkspace workspace{};
  const double projection_positions[] = {0.0, 0.0, 0.0, 1.4, 0.1, -0.2};
  const std::size_t projection_matrix_elements =
      static_cast<std::size_t>(projection.total_orbitals) *
      static_cast<std::size_t>(projection.total_orbitals);
  std::vector<double> projection_self_overlap(projection_matrix_elements);
  if (xtbloom::detail::common::evaluate_cross_overlap_system_cpu(
          projection, projection, projection_positions, projection_positions,
          projection_self_overlap.data(), &workspace, sizeof(workspace),
          error) != XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "projection smoke: self-overlap evaluation failed: %s\n", error.c_str());
    return 1;
  }
  double maximum_diagonal_error = 0.0;
  for (std::size_t diagonal = 0; diagonal < static_cast<std::size_t>(projection.total_orbitals);
       ++diagonal) {
    maximum_diagonal_error = std::max(
        maximum_diagonal_error,
        std::abs(
            projection_self_overlap[diagonal * static_cast<std::size_t>(projection.total_orbitals) +
                                    diagonal] -
            1.0));
  }
  if (!close_enough(maximum_diagonal_error, 2.0e-10)) {
    std::fprintf(stderr, "projection smoke: normalized diagonal error %.3e\n",
                 maximum_diagonal_error);
    return 1;
  }
  const std::size_t matrix_elements = static_cast<std::size_t>(native.total_orbitals) *
                                      static_cast<std::size_t>(projection.total_orbitals);
  std::vector<double> overlap(matrix_elements);
  std::vector<double> shifted_overlap(matrix_elements);
  std::vector<double> gradient(3u * 2u * matrix_elements);
  const double positions[] = {0.0, 0.0, 0.0, 1.4, 0.1, -0.2};
  const double shifted_positions[] = {3.0, -2.0, 0.5, 4.4, -1.9, 0.3};
  const double invalid_positions[] = {
      std::numeric_limits<double>::infinity(), 0.0, 0.0, 1.4, 0.1, -0.2};
  void* const misaligned_workspace =
      static_cast<void*>(reinterpret_cast<unsigned char*>(static_cast<void*>(&workspace)) + 1u);
  if (xtbloom::detail::common::evaluate_cross_overlap_system_cpu(
          native, projection, nullptr, positions, overlap.data(), &workspace, sizeof(workspace),
          error) != XTBLOOM_STATUS_INVALID_ARGUMENT ||
      xtbloom::detail::common::evaluate_cross_overlap_system_cpu(
          native, projection, positions, positions, nullptr, &workspace, sizeof(workspace),
          error) != XTBLOOM_STATUS_INVALID_ARGUMENT ||
      xtbloom::detail::common::evaluate_cross_overlap_system_cpu(
          native, projection, positions, positions, overlap.data(), misaligned_workspace,
          sizeof(workspace), error) != XTBLOOM_STATUS_INVALID_ARGUMENT ||
      xtbloom::detail::common::evaluate_cross_overlap_system_cpu(
          native, projection, invalid_positions, positions, overlap.data(), &workspace,
          sizeof(workspace), error) != XTBLOOM_STATUS_INVALID_ARGUMENT ||
      xtbloom::detail::common::evaluate_cross_overlap_gradient_system_cpu(
          native, projection, positions, positions, nullptr, &workspace, sizeof(workspace),
          error) != XTBLOOM_STATUS_INVALID_ARGUMENT ||
      xtbloom::detail::common::evaluate_overlap_gradient_system_cpu(
          native, native_integrals, positions, nullptr, &workspace, sizeof(workspace), error) !=
          XTBLOOM_STATUS_INVALID_ARGUMENT) {
    return fail("integral exports accepted invalid source-tree inputs");
  }
  if (xtbloom::detail::common::evaluate_cross_overlap_system_cpu(
          native, projection, positions, positions, overlap.data(), &workspace, sizeof(workspace),
          error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::common::evaluate_cross_overlap_system_cpu(
          native, projection, shifted_positions, shifted_positions, shifted_overlap.data(),
          &workspace, sizeof(workspace), error) != XTBLOOM_STATUS_SUCCESS ||
      xtbloom::detail::common::evaluate_cross_overlap_gradient_system_cpu(
          native, projection, positions, positions, gradient.data(), &workspace, sizeof(workspace),
          error) != XTBLOOM_STATUS_SUCCESS) {
    std::fprintf(stderr, "projection smoke: cross-overlap evaluation failed: %s\n", error.c_str());
    return 1;
  }
  double maximum_overlap_difference = 0.0;
  for (std::size_t element = 0; element < matrix_elements; ++element) {
    if (!std::isfinite(overlap[element])) {
      return fail("cross overlap or gradient contains NaN/infinity");
    }
    maximum_overlap_difference =
        std::max(maximum_overlap_difference, std::abs(overlap[element] - shifted_overlap[element]));
  }
  for (double value : gradient) {
    if (!std::isfinite(value)) {
      return fail("cross overlap or gradient contains NaN/infinity");
    }
  }
  double maximum_translation_residual = 0.0;
  for (std::size_t coordinate = 0; coordinate < 3u; ++coordinate) {
    for (std::size_t element = 0; element < matrix_elements; ++element) {
      const std::size_t first = (coordinate * matrix_elements) + element;
      const std::size_t second = (3u * matrix_elements) + first;
      maximum_translation_residual =
          std::max(maximum_translation_residual, std::abs(gradient[first] + gradient[second]));
    }
  }
  double maximum_gradient_finite_difference_error = 0.0;
  constexpr double kFiniteDifferenceStep = 1.0e-5;
  for (std::size_t atom = 0; atom < 2u; ++atom) {
    for (std::size_t coordinate = 0; coordinate < 3u; ++coordinate) {
      std::vector<double> plus_positions(positions, positions + 6);
      std::vector<double> minus_positions(positions, positions + 6);
      const std::size_t position_index = 3u * atom + coordinate;
      plus_positions[position_index] += kFiniteDifferenceStep;
      minus_positions[position_index] -= kFiniteDifferenceStep;
      std::vector<double> plus_overlap(matrix_elements);
      std::vector<double> minus_overlap(matrix_elements);
      if (xtbloom::detail::common::evaluate_cross_overlap_system_cpu(
              native, projection, plus_positions.data(), plus_positions.data(), plus_overlap.data(),
              &workspace, sizeof(workspace), error) != XTBLOOM_STATUS_SUCCESS ||
          xtbloom::detail::common::evaluate_cross_overlap_system_cpu(
              native, projection, minus_positions.data(), minus_positions.data(),
              minus_overlap.data(), &workspace, sizeof(workspace),
              error) != XTBLOOM_STATUS_SUCCESS) {
        std::fprintf(stderr, "projection smoke: finite-difference overlap failed: %s\n",
                     error.c_str());
        return 1;
      }
      for (std::size_t element = 0; element < matrix_elements; ++element) {
        const double finite_difference =
            (plus_overlap[element] - minus_overlap[element]) / (2.0 * kFiniteDifferenceStep);
        const std::size_t gradient_index = ((3u * atom + coordinate) * matrix_elements) + element;
        maximum_gradient_finite_difference_error =
            std::max(maximum_gradient_finite_difference_error,
                     std::abs(finite_difference - gradient[gradient_index]));
      }
    }
  }
  if (!close_enough(maximum_overlap_difference, 2.0e-12) ||
      !close_enough(maximum_translation_residual, 2.0e-10) ||
      !close_enough(maximum_gradient_finite_difference_error, 2.0e-8)) {
    std::fprintf(stderr,
                 "projection smoke: derivative checks failed (overlap %.3e, translation %.3e, "
                 "finite-difference %.3e)\n",
                 maximum_overlap_difference, maximum_translation_residual,
                 maximum_gradient_finite_difference_error);
    return 1;
  }
  std::printf(
      "status=success shells=%lld orbitals=%lld max_diagonal_error=%.3e "
      "max_overlap_translation=%.3e max_gradient_translation=%.3e "
      "max_gradient_finite_difference=%.3e\n",
      static_cast<long long>(projection.total_shells),
      static_cast<long long>(projection.total_orbitals), maximum_diagonal_error,
      maximum_overlap_difference, maximum_translation_residual,
      maximum_gradient_finite_difference_error);
  return 0;
}
