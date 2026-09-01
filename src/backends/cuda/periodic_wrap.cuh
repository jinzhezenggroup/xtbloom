// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#ifndef XTBLOOM_BACKENDS_CUDA_PERIODIC_WRAP_CUH
#define XTBLOOM_BACKENDS_CUDA_PERIODIC_WRAP_CUH

#include <cmath>

#include "backends/cuda/periodic_topology.cuh"

namespace xtbloom::detail::cuda {
namespace periodic_wrap_detail {

/*
 * Keep the periodic image-filter arithmetic rounded to binary64 at each
 * operation.  The host evaluator uses the same barrier between subtraction,
 * addition, and multiplication; without it, nvcc may contract a boundary
 * expression into an FMA and make the forward and reverse image domains
 * disagree by one ulp.
 */
__device__ inline double rounded_add(double lhs, double rhs) {
  volatile double result = lhs + rhs;
  return result;
}

__device__ inline double rounded_subtract(double lhs, double rhs) {
  volatile double result = lhs - rhs;
  return result;
}

__device__ inline double rounded_multiply(double lhs, double rhs) {
  volatile double result = lhs * rhs;
  return result;
}

/*
 * CUDA's long double is binary64, whereas the host lattice transform is
 * evaluated in extended precision before narrowing to binary64.  This small
 * double-double implementation keeps exact integer lattice preimages stable
 * on both backends without introducing a per-call allocation or host round
 * trip.  It is intentionally private to the device wrapping helper below.
 */
struct DoubleDouble {
  double high = 0.0;
  double low = 0.0;
};

__device__ inline DoubleDouble two_sum(double first, double second) {
  const double high = first + second;
  const double virtual_second = high - first;
  return {high, (first - (high - virtual_second)) + (second - virtual_second)};
}

__device__ inline DoubleDouble quick_two_sum(double first, double second) {
  const double high = first + second;
  return {high, second - (high - first)};
}

__device__ inline DoubleDouble add(DoubleDouble first, DoubleDouble second) {
  DoubleDouble result = two_sum(first.high, second.high);
  result.low += first.low + second.low;
  return quick_two_sum(result.high, result.low);
}

__device__ inline DoubleDouble subtract(DoubleDouble first, DoubleDouble second) {
  return add(first, {-second.high, -second.low});
}

__device__ inline DoubleDouble multiply(double first, double second) {
  const double high = first * second;
  return {high, fma(first, second, -high)};
}

__device__ inline DoubleDouble multiply(DoubleDouble value, double factor) {
  DoubleDouble result = multiply(value.high, factor);
  result.low += value.low * factor;
  return quick_two_sum(result.high, result.low);
}

__device__ inline DoubleDouble dot(const DoubleDouble first[3], const double second[3]) {
  DoubleDouble result = multiply(first[0].high, second[0]);
  result.low += first[0].low * second[0];
  result = add(result, multiply(first[1].high, second[1]));
  result.low += first[1].low * second[1];
  result = add(result, multiply(first[2].high, second[2]));
  result.low += first[2].low * second[2];
  return quick_two_sum(result.high, result.low);
}

__device__ inline void cross(const double first[3], const double second[3],
                             DoubleDouble result[3]) {
  result[0] = subtract(multiply(first[1], second[2]), multiply(first[2], second[1]));
  result[1] = subtract(multiply(first[2], second[0]), multiply(first[0], second[2]));
  result[2] = subtract(multiply(first[0], second[1]), multiply(first[1], second[0]));
}

__device__ inline DoubleDouble divide(DoubleDouble numerator, DoubleDouble denominator) {
  const double first = numerator.high / denominator.high;
  DoubleDouble remainder = subtract(numerator, multiply(denominator, first));
  const double second = (remainder.high + remainder.low) / denominator.high;
  const DoubleDouble leading = two_sum(first, second);
  remainder = subtract(remainder, multiply(denominator, second));
  const double third = (remainder.high + remainder.low) / denominator.high;
  return quick_two_sum(leading.high, leading.low + third);
}

__device__ inline double to_double(DoubleDouble value) { return value.high + value.low; }

__device__ inline double forward_component(const double fractional[3], const double* cell,
                                           int component) {
  DoubleDouble result = multiply(fractional[0], cell[component]);
  result = add(result, multiply(fractional[1], cell[3 + component]));
  result = add(result, multiply(fractional[2], cell[6 + component]));
  return to_double(result);
}

}  // namespace periodic_wrap_detail

/*
 * Wrap one atom into the host-defined half-open central cell.  The exact
 * preimage certificate is deliberately strict: only an integer triplet whose
 * compensated forward transform reproduces all three input Cartesian bits is
 * snapped.  Genuine coordinates immediately adjacent to a cell boundary are
 * therefore not altered by an epsilon heuristic.
 */
__device__ inline bool wrap_periodic_position(const Gfn2CudaPeriodicTopologyView& topology,
                                              std::int64_t system, const double* position,
                                              double* wrapped) {
  using namespace periodic_wrap_detail;
  if (topology.periodic_axes[system] == XTBLOOM_PERIODIC_AXES_NONE) {
    wrapped[0] = position[0];
    wrapped[1] = position[1];
    wrapped[2] = position[2];
    return isfinite(wrapped[0]) && isfinite(wrapped[1]) && isfinite(wrapped[2]);
  }

  const double* const cell = topology.cell_matrices + system * 9;
  const double determinant = cell[0] * (cell[4] * cell[8] - cell[5] * cell[7]) -
                             cell[1] * (cell[3] * cell[8] - cell[5] * cell[6]) +
                             cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]);
  if (!(determinant > 0.0) || !isfinite(determinant)) return false;

  const double inverse[9] = {
      (cell[4] * cell[8] - cell[5] * cell[7]) / determinant,
      (cell[2] * cell[7] - cell[1] * cell[8]) / determinant,
      (cell[1] * cell[5] - cell[2] * cell[4]) / determinant,
      (cell[5] * cell[6] - cell[3] * cell[8]) / determinant,
      (cell[0] * cell[8] - cell[2] * cell[6]) / determinant,
      (cell[2] * cell[3] - cell[0] * cell[5]) / determinant,
      (cell[3] * cell[7] - cell[4] * cell[6]) / determinant,
      (cell[1] * cell[6] - cell[0] * cell[7]) / determinant,
      (cell[0] * cell[4] - cell[1] * cell[3]) / determinant,
  };

  double fractional[3]{};
  for (int component = 0; component < 3; ++component) {
    fractional[component] = position[0] * inverse[component] +
                            position[1] * inverse[3 + component] +
                            position[2] * inverse[6 + component];
    if (!isfinite(fractional[component])) return false;
  }

  const double raw[3] = {fractional[0], fractional[1], fractional[2]};
  bool certified = false;
  for (int first = 0; first < 2 && !certified; ++first) {
    for (int second = 0; second < 2 && !certified; ++second) {
      for (int third = 0; third < 2; ++third) {
        const double candidate[3] = {first == 0 ? floor(raw[0]) : ceil(raw[0]),
                                     second == 0 ? floor(raw[1]) : ceil(raw[1]),
                                     third == 0 ? floor(raw[2]) : ceil(raw[2])};
        if (forward_component(candidate, cell, 0) == position[0] &&
            forward_component(candidate, cell, 1) == position[1] &&
            forward_component(candidate, cell, 2) == position[2]) {
          fractional[0] = candidate[0];
          fractional[1] = candidate[1];
          fractional[2] = candidate[2];
          certified = true;
          break;
        }
      }
    }
  }
  (void)certified;
  for (int component = 0; component < 3; ++component) {
    fractional[component] -= floor(fractional[component]);
    if (fractional[component] >= 1.0) fractional[component] = 0.0;
    if (fractional[component] == 0.0) fractional[component] = 0.0;
  }
  for (int component = 0; component < 3; ++component)
    wrapped[component] = forward_component(fractional, cell, component);
  return isfinite(wrapped[0]) && isfinite(wrapped[1]) && isfinite(wrapped[2]);
}

}  // namespace xtbloom::detail::cuda

#endif  // XTBLOOM_BACKENDS_CUDA_PERIODIC_WRAP_CUH
