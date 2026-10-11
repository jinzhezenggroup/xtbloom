// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_native_periodic_ewald.cuh"
#include "backends/cuda/periodic_wrap.cuh"

namespace xtbloom::detail::cuda {
namespace {

using DeviceError = Gfn2NativePeriodicEwaldDeviceError;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kSqrtPi = 1.7724538509055160272981674833411451828;
constexpr double kBinary64Epsilon = 2.220446049250313080847263336181640625e-16;
constexpr double kWignerToleranceSquared = 0.3;
constexpr double kWignerThreshold = 1.4901161193847656e-8;
constexpr std::int64_t kMaxInt64 = (std::numeric_limits<std::int64_t>::max)();

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
  bool empty = true;
};

template <typename T>
bool make_range(const T* pointer, std::int64_t elements, AddressRange* range) noexcept {
  if (range == nullptr || elements < 0) return false;
  if (elements == 0) {
    *range = {};
    return pointer == nullptr;
  }
  if (pointer == nullptr || reinterpret_cast<std::uintptr_t>(pointer) % alignof(T) != 0u ||
      static_cast<std::uint64_t>(elements) >
          static_cast<std::uint64_t>((std::numeric_limits<std::size_t>::max)() / sizeof(T))) {
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(elements) * sizeof(T);
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > (std::numeric_limits<std::uintptr_t>::max)() - bytes) return false;
  *range = {begin, begin + bytes, false};
  return true;
}

template <typename T>
bool make_optional_range(const T* pointer, std::int64_t elements, AddressRange* range) noexcept {
  if (pointer == nullptr) {
    if (range == nullptr || elements < 0) return false;
    *range = {};
    return true;
  }
  return make_range(pointer, elements, range);
}

bool overlaps(const AddressRange& first, const AddressRange& second) noexcept {
  return !first.empty && !second.empty && first.begin < second.end && second.begin < first.end;
}

template <std::size_t N>
bool all_disjoint(const std::array<AddressRange, N>& ranges) noexcept {
  for (std::size_t first = 0u; first < ranges.size(); ++first) {
    for (std::size_t second = first + 1u; second < ranges.size(); ++second) {
      if (overlaps(ranges[first], ranges[second])) return false;
    }
  }
  return true;
}

bool checked_multiply(std::int64_t value, std::int64_t factor, std::int64_t* result) noexcept {
  if (result == nullptr || value < 0 || factor < 0 || (factor != 0 && value > kMaxInt64 / factor)) {
    return false;
  }
  *result = value * factor;
  return true;
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    DeviceError error) {
  atomicCAS(system_errors + system, static_cast<std::uint32_t>(DeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ void record_device_error(std::uint32_t* device_error, DeviceError error) {
  atomicCAS(device_error, static_cast<std::uint32_t>(DeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ bool fail_system(std::uint32_t* system_errors, std::int64_t system, DeviceError error) {
  record_system_error(system_errors, system, error);
  return false;
}

__device__ bool is_origin(const Gfn2CudaPeriodicTranslation& translation) {
  /* The CPU plan's origin predicate is integer-coordinate based. */
  return translation.index[0] == 0 && translation.index[1] == 0 && translation.index[2] == 0;
}

__device__ double squared_norm(const double vector[3]) {
  return vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
}

__device__ double dot(const double lhs[3], const double rhs[3]) {
  return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

__device__ void outer(const double lhs[3], const double rhs[3], double result[9]) {
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column) result[row * 3 + column] = lhs[row] * rhs[column];
}

__device__ bool candidate_vector(const Gfn2NativePeriodicEwaldDeviceBatch& batch,
                                 std::int64_t system, const double rij[3], bool self,
                                 std::int64_t translation_index, double vector[3],
                                 double& squared) {
  const auto& translation = batch.direct_translations[translation_index];
  if (self && is_origin(translation)) return false;
  for (int component = 0; component < 3; ++component) {
    vector[component] = rij[component] - translation.cartesian[component];
    if (!isfinite(vector[component])) return false;
  }
  squared = squared_norm(vector);
  return isfinite(squared) && squared >= kBinary64Epsilon;
}

__device__ double smooth_shape(double delta) {
  double x = fmin(1.0, fmax(0.0, delta) / kWignerToleranceSquared);
  return fmax(0.0, 1.0 - 10.0 * x * x * x + 15.0 * x * x * x * x - 6.0 * x * x * x * x * x);
}

__device__ double smooth_shape_derivative(double delta) {
  const double x = fmin(1.0, fmax(0.0, delta) / kWignerToleranceSquared);
  return -30.0 * x * x * (1.0 - x) * (1.0 - x) / kWignerToleranceSquared;
}

/*
 * Summaries are enough to reproduce CPU build_wsc_images without a dynamic
 * per-pair list.  The direct metadata is a conservative superset; entries
 * outside the smoothing window contribute exactly zero and therefore do not
 * alter the normalized weights.
 */
__device__ bool wsc_summary(const Gfn2NativePeriodicEwaldDeviceBatch& batch, std::int64_t system,
                            const double rij[3], bool self, double& minimum, double& shape_sum,
                            double reference[3], double sum_gradient[3], double sum_strain[9],
                            std::uint32_t* system_errors) {
  const std::int64_t begin = batch.direct_translation_offsets[system];
  const std::int64_t end = batch.direct_translation_offsets[system + 1];
  minimum = 1.79769313486231570814527423731704357e308;
  for (std::int64_t translation = begin; translation < end; ++translation) {
    double vector[3]{};
    double squared = 0.0;
    if (!candidate_vector(batch, system, rij, self, translation, vector, squared)) continue;
    if (squared < minimum) minimum = squared;
  }
  if (!isfinite(minimum)) return fail_system(system_errors, system, DeviceError::kCoincidentImage);

  shape_sum = 0.0;
  for (std::int64_t translation = begin; translation < end; ++translation) {
    double vector[3]{};
    double squared = 0.0;
    if (!candidate_vector(batch, system, rij, self, translation, vector, squared)) continue;
    shape_sum += smooth_shape(fmax(0.0, squared - minimum));
  }
  if (!(shape_sum > 0.0) || !isfinite(shape_sum))
    return fail_system(system_errors, system, DeviceError::kNonfiniteArithmetic);

  bool found_reference = false;
  for (std::int64_t translation = begin; translation < end; ++translation) {
    double vector[3]{};
    double squared = 0.0;
    if (!candidate_vector(batch, system, rij, self, translation, vector, squared)) continue;
    if (squared == minimum) {
      reference[0] = vector[0];
      reference[1] = vector[1];
      reference[2] = vector[2];
      found_reference = true;
      break;
    }
  }
  if (!found_reference)
    return fail_system(system_errors, system, DeviceError::kNonfiniteArithmetic);
  for (int axis = 0; axis < 3; ++axis) sum_gradient[axis] = 0.0;
  for (int index = 0; index < 9; ++index) sum_strain[index] = 0.0;
  for (std::int64_t translation = begin; translation < end; ++translation) {
    double vector[3]{};
    double squared = 0.0;
    if (!candidate_vector(batch, system, rij, self, translation, vector, squared)) continue;
    const double delta = fmax(0.0, squared - minimum);
    const double derivative = smooth_shape_derivative(delta);
    const double difference[3] = {vector[0] - reference[0], vector[1] - reference[1],
                                  vector[2] - reference[2]};
    for (int axis = 0; axis < 3; ++axis) sum_gradient[axis] += derivative * 2.0 * difference[axis];
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        sum_strain[row * 3 + column] +=
            derivative * 2.0 * (vector[row] * vector[column] - reference[row] * reference[column]);
      }
    }
  }
  return isfinite(shape_sum) && isfinite(sum_gradient[0]) && isfinite(sum_gradient[1]) &&
         isfinite(sum_gradient[2]);
}

__device__ bool wsc_image(const Gfn2NativePeriodicEwaldDeviceBatch& batch, std::int64_t system,
                          const double rij[3], bool self, std::int64_t translation_index,
                          double minimum, double shape_sum, const double reference[3],
                          const double sum_gradient[3], const double sum_strain[9],
                          double vector[3], double& weight, double weight_gradient[3],
                          double weight_strain[9]) {
  double squared = 0.0;
  if (!candidate_vector(batch, system, rij, self, translation_index, vector, squared)) return false;
  const double delta = fmax(0.0, squared - minimum);
  const double shape = smooth_shape(delta);
  weight = shape / shape_sum;
  const double derivative = smooth_shape_derivative(delta);
  const double difference[3] = {vector[0] - reference[0], vector[1] - reference[1],
                                vector[2] - reference[2]};
  for (int axis = 0; axis < 3; ++axis) {
    const double shape_gradient = derivative * 2.0 * difference[axis];
    weight_gradient[axis] = (shape_gradient - weight * sum_gradient[axis]) / shape_sum;
  }
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      const int index = row * 3 + column;
      const double shape_strain =
          derivative * 2.0 * (vector[row] * vector[column] - reference[row] * reference[column]);
      weight_strain[index] = (shape_strain - weight * sum_strain[index]) / shape_sum;
    }
  }
  return isfinite(weight) && isfinite(weight_gradient[0]) && isfinite(weight_gradient[1]) &&
         isfinite(weight_gradient[2]);
}

__device__ double direct_term(const double vector[3], double alpha) {
  const double distance = sqrt(squared_norm(vector));
  if (distance < kWignerThreshold) return 0.0;
  return erfc(alpha * distance) / distance;
}

__device__ void direct_derivative(const double vector[3], double alpha, double result[3]) {
  const double squared = squared_norm(vector);
  const double distance = sqrt(squared);
  if (distance < kWignerThreshold) {
    result[0] = result[1] = result[2] = 0.0;
    return;
  }
  const double coefficient = -erfc(alpha * distance) / (squared * distance) -
                             2.0 * alpha * exp(-squared * alpha * alpha) / (kSqrtPi * squared);
  for (int axis = 0; axis < 3; ++axis) result[axis] = coefficient * vector[axis];
}

__device__ void direct_strain_term(const double vector[3], double alpha, double result[9]) {
  double derivative[3]{};
  direct_derivative(vector, alpha, derivative);
  outer(derivative, vector, result);
}

__device__ double direct_value(const double rij[3], const Gfn2NativePeriodicEwaldDeviceBatch& batch,
                               std::int64_t system, double alpha) {
  double result = 0.0;
  const std::int64_t begin = batch.direct_translation_offsets[system];
  const std::int64_t end = batch.direct_translation_offsets[system + 1];
  for (std::int64_t translation = begin; translation < end; ++translation) {
    const auto& value = batch.direct_translations[translation];
    const double vector[3] = {rij[0] - value.cartesian[0], rij[1] - value.cartesian[1],
                              rij[2] - value.cartesian[2]};
    result += direct_term(vector, alpha);
  }
  return result;
}

__device__ void direct_derivative_sum(const double rij[3],
                                      const Gfn2NativePeriodicEwaldDeviceBatch& batch,
                                      std::int64_t system, double alpha, double result[3]) {
  result[0] = result[1] = result[2] = 0.0;
  const std::int64_t begin = batch.direct_translation_offsets[system];
  const std::int64_t end = batch.direct_translation_offsets[system + 1];
  for (std::int64_t translation = begin; translation < end; ++translation) {
    const auto& value = batch.direct_translations[translation];
    const double vector[3] = {rij[0] - value.cartesian[0], rij[1] - value.cartesian[1],
                              rij[2] - value.cartesian[2]};
    double term[3]{};
    direct_derivative(vector, alpha, term);
    for (int axis = 0; axis < 3; ++axis) result[axis] += term[axis];
  }
}

__device__ void direct_strain_sum(const double rij[3],
                                  const Gfn2NativePeriodicEwaldDeviceBatch& batch,
                                  std::int64_t system, double alpha, double result[9]) {
  for (int index = 0; index < 9; ++index) result[index] = 0.0;
  const std::int64_t begin = batch.direct_translation_offsets[system];
  const std::int64_t end = batch.direct_translation_offsets[system + 1];
  for (std::int64_t translation = begin; translation < end; ++translation) {
    const auto& value = batch.direct_translations[translation];
    const double vector[3] = {rij[0] - value.cartesian[0], rij[1] - value.cartesian[1],
                              rij[2] - value.cartesian[2]};
    double term[9]{};
    direct_strain_term(vector, alpha, term);
    for (int index = 0; index < 9; ++index) result[index] += term[index];
  }
}

__device__ double reciprocal_value(const double rij[3],
                                   const Gfn2NativePeriodicEwaldDeviceBatch& batch,
                                   std::int64_t system, double alpha, double volume) {
  double result = 0.0;
  const double alpha_squared = alpha * alpha;
  const std::int64_t begin = batch.reciprocal_translation_offsets[system];
  const std::int64_t end = batch.reciprocal_translation_offsets[system + 1];
  for (std::int64_t translation = begin; translation < end; ++translation) {
    const auto& value = batch.reciprocal_translations[translation];
    const double g[3] = {value.cartesian[0], value.cartesian[1], value.cartesian[2]};
    const double g2 = squared_norm(g);
    if (g2 < kBinary64Epsilon) continue;
    result += 4.0 * kPi / volume * exp(-0.25 * g2 / alpha_squared) * cos(dot(rij, g)) / g2;
  }
  return result;
}

__device__ void reciprocal_derivative(const double rij[3],
                                      const Gfn2NativePeriodicEwaldDeviceBatch& batch,
                                      std::int64_t system, double alpha, double volume,
                                      double result[3]) {
  result[0] = result[1] = result[2] = 0.0;
  const double alpha_squared = alpha * alpha;
  const std::int64_t begin = batch.reciprocal_translation_offsets[system];
  const std::int64_t end = batch.reciprocal_translation_offsets[system + 1];
  for (std::int64_t translation = begin; translation < end; ++translation) {
    const auto& value = batch.reciprocal_translations[translation];
    const double g[3] = {value.cartesian[0], value.cartesian[1], value.cartesian[2]};
    const double g2 = squared_norm(g);
    if (g2 < kBinary64Epsilon) continue;
    const double coefficient = 4.0 * kPi / volume * exp(-0.25 * g2 / alpha_squared) / g2;
    const double factor = -sin(dot(rij, g)) * coefficient;
    for (int axis = 0; axis < 3; ++axis) result[axis] += factor * g[axis];
  }
}

__device__ void reciprocal_strain(const double rij[3],
                                  const Gfn2NativePeriodicEwaldDeviceBatch& batch,
                                  std::int64_t system, double alpha, double volume,
                                  double result[9]) {
  for (int index = 0; index < 9; ++index) result[index] = 0.0;
  const double alpha_squared = alpha * alpha;
  const std::int64_t begin = batch.reciprocal_translation_offsets[system];
  const std::int64_t end = batch.reciprocal_translation_offsets[system + 1];
  for (std::int64_t translation = begin; translation < end; ++translation) {
    const auto& value = batch.reciprocal_translations[translation];
    const double g[3] = {value.cartesian[0], value.cartesian[1], value.cartesian[2]};
    const double g2 = squared_norm(g);
    if (g2 < kBinary64Epsilon) continue;
    const double exponential = 4.0 * kPi / volume * exp(-0.25 * g2 / alpha_squared) / g2;
    const double cosine = cos(dot(rij, g)) * exponential;
    for (int row = 0; row < 3; ++row) {
      for (int column = 0; column < 3; ++column) {
        const int index = row * 3 + column;
        result[index] += cosine * g[row] * g[column] * (2.0 / g2 + 0.5 / alpha_squared);
        if (index == 0 || index == 4 || index == 8) result[index] -= cosine;
      }
    }
  }
}

__device__ double correction_value(const double vector[3], double gamma) {
  const double squared = squared_norm(vector);
  const double distance = sqrt(squared);
  if (distance < kWignerThreshold) return 0.0;
  return 1.0 / sqrt(squared + 1.0 / (gamma * gamma)) - 1.0 / distance;
}

__device__ void correction_derivative(const double vector[3], double gamma, double result[3]) {
  const double squared = squared_norm(vector);
  const double distance = sqrt(squared);
  if (distance < kWignerThreshold) {
    result[0] = result[1] = result[2] = 0.0;
    return;
  }
  const double inverse_gamma_squared = 1.0 / (gamma * gamma);
  const double coefficient =
      -1.0 / pow(squared + inverse_gamma_squared, 1.5) + 1.0 / (squared * distance);
  for (int axis = 0; axis < 3; ++axis) result[axis] = coefficient * vector[axis];
}

__device__ bool finite_vector(const double* values, std::int64_t count) {
  for (std::int64_t index = 0; index < count; ++index)
    if (!isfinite(values[index])) return false;
  return true;
}

__device__ bool add_pair(const Gfn2NativePeriodicEwaldDeviceBatch& batch,
                         const Gfn2NativePeriodicEwaldDeviceWorkspace& workspace,
                         std::int64_t system, std::int64_t atom_begin, std::int64_t shell_begin,
                         std::int64_t center, std::int64_t image, const double rij[3], double alpha,
                         double volume, bool self, double* matrix, double* gradients,
                         double* strain, std::uint32_t* system_errors) {
  double minimum = 0.0;
  double shape_sum = 0.0;
  double reference[3]{};
  double sum_gradient[3]{};
  double sum_strain[9]{};
  if (!wsc_summary(batch, system, rij, self, minimum, shape_sum, reference, sum_gradient,
                   sum_strain, system_errors)) {
    return false;
  }

  const double base = direct_value(rij, batch, system, alpha) +
                      reciprocal_value(rij, batch, system, alpha, volume) -
                      (self ? 2.0 * alpha / kSqrtPi : 0.0);
  double d_direct[3]{};
  double d_reciprocal[3]{};
  double d_strain_direct[9]{};
  double d_strain_reciprocal[9]{};
  direct_derivative_sum(rij, batch, system, alpha, d_direct);
  reciprocal_derivative(rij, batch, system, alpha, volume, d_reciprocal);
  direct_strain_sum(rij, batch, system, alpha, d_strain_direct);
  reciprocal_strain(rij, batch, system, alpha, volume, d_strain_reciprocal);

  const std::int64_t center_shell_begin = batch.atom_shell_offsets[atom_begin + center];
  const std::int64_t center_shell_end = batch.atom_shell_offsets[atom_begin + center + 1];
  const std::int64_t image_shell_begin = batch.atom_shell_offsets[atom_begin + image];
  const std::int64_t image_shell_end = batch.atom_shell_offsets[atom_begin + image + 1];
  const std::int64_t local_shell_count =
      batch.batch_shell_offsets[system + 1] - batch.batch_shell_offsets[system];
  const std::int64_t matrix_begin = batch.matrix_offsets[system];

  const std::int64_t translation_begin = batch.direct_translation_offsets[system];
  const std::int64_t translation_end = batch.direct_translation_offsets[system + 1];
  for (std::int64_t translation = translation_begin; translation < translation_end; ++translation) {
    double vector[3]{};
    double weight = 0.0;
    double weight_gradient[3]{};
    double weight_strain[9]{};
    if (!wsc_image(batch, system, rij, self, translation, minimum, shape_sum, reference,
                   sum_gradient, sum_strain, vector, weight, weight_gradient, weight_strain)) {
      continue;
    }
    if (!(weight > 0.0)) continue;
    double image_correction_derivative[3]{};
    for (std::int64_t shell_i = center_shell_begin; shell_i < center_shell_end; ++shell_i) {
      const std::int64_t shell_j_begin = self ? center_shell_begin : image_shell_begin;
      const std::int64_t shell_j_end = self ? shell_i + 1 : image_shell_end;
      for (std::int64_t shell_j = shell_j_begin; shell_j < shell_j_end; ++shell_j) {
        const double gamma = 0.5 * (batch.shell_hardness[shell_i] + batch.shell_hardness[shell_j]);
        const double correction = correction_value(vector, gamma);
        const double value = self ? (base + correction + gamma) : (base + correction);
        const std::int64_t local_i = shell_i - batch.batch_shell_offsets[system];
        const std::int64_t local_j = shell_j - batch.batch_shell_offsets[system];
        const std::int64_t ij = matrix_begin + local_i * local_shell_count + local_j;
        matrix[ij] += self ? weight * value : weight * value;
        if (self && local_i != local_j) {
          const std::int64_t ji = matrix_begin + local_j * local_shell_count + local_i;
          matrix[ji] += weight * value;
        } else if (!self) {
          const std::int64_t ji = matrix_begin + local_j * local_shell_count + local_i;
          matrix[ji] += weight * value;
        }

        if (!self) {
          correction_derivative(vector, gamma, image_correction_derivative);
          double derivative[3]{};
          double derivative_strain[9]{};
          for (int axis = 0; axis < 3; ++axis) {
            derivative[axis] =
                weight * (d_direct[axis] + d_reciprocal[axis] + image_correction_derivative[axis]) +
                weight_gradient[axis] * correction;
          }
          double correction_strain[9]{};
          outer(image_correction_derivative, vector, correction_strain);
          for (int index = 0; index < 9; ++index) {
            derivative_strain[index] =
                weight * (d_strain_direct[index] + d_strain_reciprocal[index] +
                          correction_strain[index]) +
                weight_strain[index] * correction;
          }
          const double factor = batch.shell_charges[shell_i] * batch.shell_charges[shell_j];
          for (int axis = 0; axis < 3; ++axis) {
            gradients[(atom_begin + center) * 3 + axis] += factor * derivative[axis];
            gradients[(atom_begin + image) * 3 + axis] -= factor * derivative[axis];
          }
          for (int index = 0; index < 9; ++index)
            strain[system * 9 + index] += factor * derivative_strain[index];
        } else {
          double correction_derivative_self[3]{};
          correction_derivative(vector, gamma, correction_derivative_self);
          double correction_strain[9]{};
          outer(correction_derivative_self, vector, correction_strain);
          const double factor =
              shell_i == shell_j ? 0.5 * batch.shell_charges[shell_i] * batch.shell_charges[shell_i]
                                 : batch.shell_charges[shell_i] * batch.shell_charges[shell_j];
          for (int index = 0; index < 9; ++index) {
            strain[system * 9 + index] +=
                factor * (weight * (d_strain_direct[index] + d_strain_reciprocal[index] +
                                    correction_strain[index]) +
                          weight_strain[index] * correction);
          }
        }
      }
    }
  }
  return true;
}

__global__ void native_periodic_ewald_kernel(Gfn2NativePeriodicEwaldDeviceBatch batch,
                                             Gfn2NativePeriodicEwaldDeviceWorkspace workspace,
                                             double* coulomb_matrix, double* shell_potentials,
                                             double* energies, double* gradients, double* strain,
                                             std::uint32_t* system_errors,
                                             std::uint32_t* device_error) {
  if (threadIdx.x != 0) return;
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= batch.topology.batch_size) return;
  if (batch.active_mask != nullptr && batch.active_mask[system] == 0u) return;

  const std::int64_t atom_begin = batch.topology.atom_offsets[system];
  const std::int64_t atom_end = batch.topology.atom_offsets[system + 1];
  const std::int64_t shell_begin = batch.batch_shell_offsets[system];
  const std::int64_t shell_end = batch.batch_shell_offsets[system + 1];
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t matrix_end = batch.matrix_offsets[system + 1];
  const std::int64_t shell_count = shell_end - shell_begin;
  const std::int64_t direct_begin = batch.direct_translation_offsets[system];
  const std::int64_t direct_end = batch.direct_translation_offsets[system + 1];
  const std::int64_t reciprocal_begin = batch.reciprocal_translation_offsets[system];
  const std::int64_t reciprocal_end = batch.reciprocal_translation_offsets[system + 1];
  const std::int64_t total_shells = batch.shell_hardness_elements;
  const std::int64_t total_matrix_elements = batch.matrix_elements;
  if (atom_begin < 0 || atom_begin > atom_end || atom_end > batch.topology.total_atoms ||
      batch.topology.atom_offsets[0] != 0 ||
      batch.topology.atom_offsets[batch.topology.batch_size] != batch.topology.total_atoms ||
      batch.batch_shell_offsets[0] != 0 ||
      batch.batch_shell_offsets[batch.topology.batch_size] != total_shells ||
      batch.matrix_offsets[0] != 0 ||
      batch.matrix_offsets[batch.topology.batch_size] != total_matrix_elements ||
      batch.direct_translation_offsets[0] != 0 ||
      batch.direct_translation_offsets[batch.topology.batch_size] !=
          batch.direct_translation_elements ||
      batch.reciprocal_translation_offsets[0] != 0 ||
      batch.reciprocal_translation_offsets[batch.topology.batch_size] !=
          batch.reciprocal_translation_elements ||
      shell_begin < 0 || shell_begin > shell_end || shell_end > batch.shell_hardness_elements ||
      matrix_begin < 0 || matrix_begin > matrix_end ||
      matrix_end > batch.matrix_offsets[batch.topology.batch_size] || direct_begin < 0 ||
      direct_begin > direct_end || direct_end > batch.direct_translation_elements ||
      reciprocal_begin < 0 || reciprocal_begin > reciprocal_end ||
      reciprocal_end > batch.reciprocal_translation_elements || shell_count <= 0 ||
      shell_count > kMaxInt64 / shell_count ||
      matrix_end != matrix_begin + shell_count * shell_count ||
      batch.topology.periodic_axes[system] != XTBLOOM_PERIODIC_AXES_XYZ) {
    record_system_error(system_errors, system, DeviceError::kInvalidOffsets);
    record_device_error(device_error, DeviceError::kInvalidTopology);
    return;
  }
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    const std::int64_t first = batch.atom_shell_offsets[atom];
    const std::int64_t last = batch.atom_shell_offsets[atom + 1];
    if (first < shell_begin || first > last || last > shell_end) {
      record_system_error(system_errors, system, DeviceError::kInvalidOffsets);
      record_device_error(device_error, DeviceError::kInvalidTopology);
      return;
    }
  }
  if (batch.atom_shell_offsets[atom_begin] != shell_begin ||
      batch.atom_shell_offsets[atom_end] != shell_end) {
    record_system_error(system_errors, system, DeviceError::kInvalidOffsets);
    record_device_error(device_error, DeviceError::kInvalidTopology);
    return;
  }

  const double alpha = batch.alphas[system];
  if (!(alpha > 0.0) || !isfinite(alpha)) {
    record_system_error(system_errors, system, DeviceError::kInvalidAlpha);
    return;
  }
  const double* const cell = batch.topology.cell_matrices + system * 9;
  const double volume = cell[0] * (cell[4] * cell[8] - cell[5] * cell[7]) -
                        cell[1] * (cell[3] * cell[8] - cell[5] * cell[6]) +
                        cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]);
  if (!(volume > 0.0) || !isfinite(volume)) {
    record_system_error(system_errors, system, DeviceError::kInvalidTopology);
    record_device_error(device_error, DeviceError::kInvalidTopology);
    return;
  }

  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    const double* position = batch.positions + atom * 3;
    double* wrapped = workspace.wrapped_positions + atom * 3;
    if (!finite_vector(position, 3) ||
        !wrap_periodic_position(batch.topology, system, position, wrapped)) {
      record_system_error(system_errors, system, DeviceError::kNonfinitePosition);
      return;
    }
  }
  for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) {
    if (!(batch.shell_hardness[shell] > 0.0) || !isfinite(batch.shell_hardness[shell]) ||
        !isfinite(batch.shell_charges[shell])) {
      record_system_error(system_errors, system,
                          !isfinite(batch.shell_charges[shell]) ? DeviceError::kNonfiniteCharge
                                                                : DeviceError::kInvalidHardness);
      return;
    }
  }

  double* const matrix = workspace.matrix;
  double* const potential = workspace.shell_potentials;
  double* const energy = workspace.energies;
  double* const gradient = workspace.gradients;
  double* const system_strain = workspace.strain + system * 9;
  for (std::int64_t index = matrix_begin; index < matrix_end; ++index) matrix[index] = 0.0;
  for (std::int64_t shell = shell_begin; shell < shell_end; ++shell) potential[shell] = 0.0;
  energy[system] = 0.0;
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom)
    for (int axis = 0; axis < 3; ++axis) gradient[atom * 3 + axis] = 0.0;
  for (int index = 0; index < 9; ++index) system_strain[index] = 0.0;

  for (std::int64_t center = 0; center < atom_end - atom_begin; ++center) {
    for (std::int64_t image = 0; image < center; ++image) {
      const double* center_position = workspace.wrapped_positions + (atom_begin + center) * 3;
      const double* image_position = workspace.wrapped_positions + (atom_begin + image) * 3;
      const double rij[3] = {center_position[0] - image_position[0],
                             center_position[1] - image_position[1],
                             center_position[2] - image_position[2]};
      if (!add_pair(batch, workspace, system, atom_begin, shell_begin, center, image, rij, alpha,
                    volume, false, matrix, gradient, workspace.strain, system_errors))
        return;
    }
    const double zero[3] = {0.0, 0.0, 0.0};
    if (!add_pair(batch, workspace, system, atom_begin, shell_begin, center, center, zero, alpha,
                  volume, true, matrix, gradient, workspace.strain, system_errors))
      return;
  }

  double total_charge = 0.0;
  for (std::int64_t shell = shell_begin; shell < shell_end; ++shell)
    total_charge += batch.shell_charges[shell];
  const double alpha_squared = alpha * alpha;
  const double background_factor = -kPi / (alpha_squared * volume);
  if (!isfinite(total_charge) || !(alpha_squared > 0.0) || !isfinite(background_factor)) {
    record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
    return;
  }
  for (std::int64_t row = 0; row < shell_count; ++row) {
    double value = 0.0;
    for (std::int64_t column = 0; column < shell_count; ++column) {
      value += matrix[matrix_begin + row * shell_count + column] *
               batch.shell_charges[shell_begin + column];
    }
    potential[shell_begin + row] = value + background_factor * total_charge;
    energy[system] += 0.5 * batch.shell_charges[shell_begin + row] * value;
  }
  energy[system] += 0.5 * background_factor * total_charge * total_charge;
  const double background_strain = -0.5 * background_factor * total_charge * total_charge;
  system_strain[0] += background_strain;
  system_strain[4] += background_strain;
  system_strain[8] += background_strain;

  if (!finite_vector(matrix + matrix_begin, matrix_end - matrix_begin) ||
      !finite_vector(potential + shell_begin, shell_end - shell_begin) ||
      !finite_vector(energy + system, 1) ||
      !finite_vector(gradient + atom_begin * 3, (atom_end - atom_begin) * 3) ||
      !finite_vector(system_strain, 9)) {
    record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
    return;
  }

  /* Commit this peer only after every matrix/potential/energy/derivative value
   * is finite.  Other ragged peers can publish independently. */
  if (coulomb_matrix != nullptr) {
    for (std::int64_t index = matrix_begin; index < matrix_end; ++index)
      coulomb_matrix[index] = matrix[index];
  }
  if (shell_potentials != nullptr) {
    for (std::int64_t shell = shell_begin; shell < shell_end; ++shell)
      shell_potentials[shell] = potential[shell];
  }
  if (energies != nullptr) energies[system] = energy[system];
  if (gradients != nullptr) {
    for (std::int64_t atom = atom_begin; atom < atom_end; ++atom)
      for (int axis = 0; axis < 3; ++axis) gradients[atom * 3 + axis] = gradient[atom * 3 + axis];
  }
  if (strain != nullptr) {
    for (int index = 0; index < 9; ++index) strain[system * 9 + index] = system_strain[index];
  }
}

}  // namespace

cudaError_t reset_gfn2_native_periodic_ewald_errors_cuda(std::int64_t batch_size,
                                                         std::uint32_t* system_errors,
                                                         std::uint32_t* device_error,
                                                         cudaStream_t stream) noexcept {
  if (batch_size <= 0 || system_errors == nullptr || device_error == nullptr ||
      static_cast<std::uint64_t>(batch_size) >
          static_cast<std::uint64_t>((std::numeric_limits<std::size_t>::max)() /
                                     sizeof(*system_errors))) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(
      system_errors, 0, static_cast<std::size_t>(batch_size) * sizeof(*system_errors), stream);
  if (status != cudaSuccess) return status;
  return cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream);
}

cudaError_t evaluate_gfn2_native_periodic_ewald_cuda(
    const Gfn2NativePeriodicEwaldDeviceBatch& batch,
    const Gfn2NativePeriodicEwaldDeviceWorkspace& workspace, double* coulomb_matrix,
    double* shell_potentials, double* energies, double* gradients, double* strain,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  const auto& topology = batch.topology;
  if (topology.batch_size <= 0 || topology.total_atoms < 0 || topology.total_translations <= 0 ||
      topology.batch_size > static_cast<std::int64_t>((std::numeric_limits<unsigned int>::max)()) ||
      topology.atom_offset_count != topology.batch_size + 1 ||
      topology.translation_offset_count != topology.batch_size + 1 ||
      topology.cell_elements != topology.batch_size * 9 ||
      topology.periodic_axes_elements != topology.batch_size || topology.plan_token == 0u ||
      topology.atom_offsets == nullptr || topology.cell_matrices == nullptr ||
      topology.periodic_axes == nullptr || topology.translation_offsets == nullptr ||
      topology.translations == nullptr ||
      batch.batch_shell_offset_elements != topology.batch_size + 1 ||
      batch.atom_shell_offset_elements != topology.total_atoms + 1 ||
      batch.matrix_offset_elements != topology.batch_size + 1 || batch.matrix_elements <= 0 ||
      batch.shell_hardness_elements < 0 || batch.alpha_elements != topology.batch_size ||
      batch.direct_translation_offset_elements != topology.batch_size + 1 ||
      batch.reciprocal_translation_offset_elements != topology.batch_size + 1 ||
      batch.position_elements < 0 || batch.shell_charge_elements < 0 ||
      workspace.plan_token != topology.plan_token || workspace.wrapped_position_elements < 0 ||
      workspace.matrix_elements < 0 || workspace.shell_potential_elements < 0 ||
      workspace.energy_elements != topology.batch_size || workspace.gradient_elements < 0 ||
      workspace.strain_elements < 0 || system_errors == nullptr || device_error == nullptr) {
    return cudaErrorInvalidValue;
  }
  if ((batch.active_mask == nullptr) != (batch.active_mask_elements == 0) ||
      (batch.active_mask != nullptr && batch.active_mask_elements != topology.batch_size)) {
    return cudaErrorInvalidValue;
  }

  std::int64_t atom_coordinate_elements = 0;
  std::int64_t strain_elements = 0;
  if (!checked_multiply(topology.total_atoms, 3, &atom_coordinate_elements) ||
      !checked_multiply(topology.batch_size, 9, &strain_elements) ||
      batch.position_elements != atom_coordinate_elements ||
      workspace.wrapped_position_elements != atom_coordinate_elements ||
      workspace.gradient_elements != atom_coordinate_elements ||
      workspace.strain_elements != strain_elements ||
      workspace.matrix_elements != batch.matrix_elements ||
      workspace.shell_potential_elements != batch.shell_hardness_elements ||
      batch.shell_charge_elements != batch.shell_hardness_elements ||
      batch.batch_shell_offsets == nullptr || batch.atom_shell_offsets == nullptr ||
      batch.matrix_offsets == nullptr || batch.shell_hardness == nullptr ||
      batch.alphas == nullptr || batch.direct_translation_offsets == nullptr ||
      batch.direct_translations == nullptr || batch.reciprocal_translation_offsets == nullptr ||
      batch.reciprocal_translations == nullptr || batch.positions == nullptr ||
      batch.shell_charges == nullptr || workspace.wrapped_positions == nullptr ||
      workspace.matrix == nullptr || workspace.shell_potentials == nullptr ||
      workspace.energies == nullptr || workspace.gradients == nullptr ||
      workspace.strain == nullptr) {
    return cudaErrorInvalidValue;
  }

  std::array<AddressRange, 26> ranges{};
  if (!make_range(topology.atom_offsets, topology.atom_offset_count, &ranges[0]) ||
      !make_range(topology.cell_matrices, topology.cell_elements, &ranges[1]) ||
      !make_range(topology.periodic_axes, topology.periodic_axes_elements, &ranges[2]) ||
      !make_range(topology.translation_offsets, topology.translation_offset_count, &ranges[3]) ||
      !make_range(topology.translations, topology.total_translations, &ranges[4]) ||
      !make_range(batch.batch_shell_offsets, batch.batch_shell_offset_elements, &ranges[5]) ||
      !make_range(batch.atom_shell_offsets, batch.atom_shell_offset_elements, &ranges[6]) ||
      !make_range(batch.matrix_offsets, batch.matrix_offset_elements, &ranges[7]) ||
      !make_range(batch.shell_hardness, batch.shell_hardness_elements, &ranges[8]) ||
      !make_range(batch.alphas, batch.alpha_elements, &ranges[9]) ||
      !make_range(batch.direct_translation_offsets, batch.direct_translation_offset_elements,
                  &ranges[10]) ||
      !make_range(batch.direct_translations, batch.direct_translation_elements, &ranges[11]) ||
      !make_range(batch.reciprocal_translation_offsets,
                  batch.reciprocal_translation_offset_elements, &ranges[12]) ||
      !make_range(batch.reciprocal_translations, batch.reciprocal_translation_elements,
                  &ranges[13]) ||
      !make_range(batch.positions, batch.position_elements, &ranges[14]) ||
      !make_range(batch.shell_charges, batch.shell_charge_elements, &ranges[15]) ||
      !make_range(batch.active_mask, batch.active_mask_elements, &ranges[25]) ||
      !make_range(workspace.wrapped_positions, workspace.wrapped_position_elements, &ranges[16]) ||
      !make_range(workspace.matrix, workspace.matrix_elements, &ranges[17]) ||
      !make_range(workspace.shell_potentials, workspace.shell_potential_elements, &ranges[18]) ||
      !make_range(workspace.energies, workspace.energy_elements, &ranges[19]) ||
      !make_range(workspace.gradients, workspace.gradient_elements, &ranges[20]) ||
      !make_range(workspace.strain, workspace.strain_elements, &ranges[21]) ||
      !make_optional_range(coulomb_matrix, batch.matrix_elements, &ranges[22]) ||
      !make_optional_range(shell_potentials, batch.shell_hardness_elements, &ranges[23]) ||
      !make_optional_range(energies, topology.batch_size, &ranges[24])) {
    return cudaErrorInvalidValue;
  }
  AddressRange gradients_range{};
  AddressRange strain_range{};
  AddressRange errors_range{};
  AddressRange device_error_range{};
  if (!make_optional_range(gradients, atom_coordinate_elements, &gradients_range) ||
      !make_optional_range(strain, strain_elements, &strain_range) ||
      !make_range(system_errors, topology.batch_size, &errors_range) ||
      !make_range(device_error, 1, &device_error_range) ||
      overlaps(errors_range, device_error_range) || !all_disjoint(ranges)) {
    return cudaErrorInvalidValue;
  }
  const std::array<AddressRange, 6> writes{ranges[16], ranges[17], ranges[18],
                                           ranges[19], ranges[20], ranges[21]};
  const std::array<AddressRange, 5> outputs{ranges[22], ranges[23], ranges[24], gradients_range,
                                            strain_range};
  if (!all_disjoint(outputs)) return cudaErrorInvalidValue;
  for (const auto& write : writes) {
    for (const auto& output : outputs)
      if (overlaps(write, output)) return cudaErrorInvalidValue;
    if (overlaps(write, errors_range) || overlaps(write, device_error_range))
      return cudaErrorInvalidValue;
  }
  for (const auto& output : outputs) {
    if (overlaps(output, errors_range) || overlaps(output, device_error_range))
      return cudaErrorInvalidValue;
  }
  for (std::size_t read = 0u; read < 17u; ++read) {
    for (const auto& output : outputs)
      if (overlaps(ranges[read], output)) return cudaErrorInvalidValue;
  }

  native_periodic_ewald_kernel<<<static_cast<unsigned int>(topology.batch_size), 1, 0, stream>>>(
      batch, workspace, coulomb_matrix, shell_potentials, energies, gradients, strain,
      system_errors, device_error);
  return cudaGetLastError();
}

}  // namespace xtbloom::detail::cuda
