// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdio.h>

#include "backends/cuda/gfn2_native_periodic_multipole.cuh"
#include "backends/cuda/gfn2_parameters.cuh"
#include "backends/cuda/periodic_wrap.cuh"

namespace xtbloom::detail::cuda {
namespace {

using DeviceError = Gfn2NativePeriodicMultipoleDeviceError;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kSqrtPi = 1.7724538509055160272981674833411451828;
constexpr double kBinary64Epsilon = 2.220446049250313080847263336181640625e-16;
constexpr double kWignerToleranceSquared = 0.3;
/* CPU build_wsc_images compares squared distance against sqrt(epsilon). */
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

__host__ __device__ bool checked_multiply(std::int64_t value, std::int64_t factor,
                                          std::int64_t* result) noexcept {
  if (result == nullptr || value < 0 || factor < 0 || (factor != 0 && value > kMaxInt64 / factor)) {
    return false;
  }
  *result = value * factor;
  return true;
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    DeviceError error) {
  const auto code = static_cast<std::uint32_t>(error);
  if (atomicCAS(system_errors + system, static_cast<std::uint32_t>(DeviceError::kSuccess), code) ==
      static_cast<std::uint32_t>(DeviceError::kSuccess)) {
  }
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
  return translation.index[0] == 0 && translation.index[1] == 0 && translation.index[2] == 0;
}

__device__ double dot3(const double first[3], const double second[3]) {
  return first[0] * second[0] + first[1] * second[1] + first[2] * second[2];
}

__device__ double norm3_squared(const double value[3]) { return dot3(value, value); }

__device__ void outer3(const double first[3], const double second[3], double result[9]) {
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      result[row * 3 + column] = first[row] * second[column];
}

__device__ void add_scaled3(double target[3], const double value[3], double factor = 1.0) {
  for (int axis = 0; axis < 3; ++axis) target[axis] += factor * value[axis];
}

__device__ void add_scaled9(double target[9], const double value[9], double factor = 1.0) {
  for (int component = 0; component < 9; ++component)
    target[component] += factor * value[component];
}

__device__ void add_scaled6(double target[6], const double value[6], double factor = 1.0) {
  for (int component = 0; component < 6; ++component)
    target[component] += factor * value[component];
}

__device__ void packed_outer3(const double vector[3], double factor, double result[6]) {
  result[0] = factor * vector[0] * vector[0];
  result[1] = factor * 2.0 * vector[0] * vector[1];
  result[2] = factor * vector[1] * vector[1];
  result[3] = factor * 2.0 * vector[0] * vector[2];
  result[4] = factor * 2.0 * vector[1] * vector[2];
  result[5] = factor * vector[2] * vector[2];
}

__device__ double packed_dot6(const double tensor[6], const double quadrupole[6]) {
  double result = 0.0;
  for (int component = 0; component < 6; ++component)
    result += tensor[component] * quadrupole[component];
  return result;
}

__device__ void quadrupole_vector(const double quadrupole[6], const double vector[3],
                                  double result[3]) {
  result[0] = quadrupole[0] * vector[0] + quadrupole[1] * vector[1] + quadrupole[3] * vector[2];
  result[1] = quadrupole[1] * vector[0] + quadrupole[2] * vector[1] + quadrupole[4] * vector[2];
  result[2] = quadrupole[3] * vector[0] + quadrupole[4] * vector[1] + quadrupole[5] * vector[2];
}

__device__ double quadrupole_scalar(const double quadrupole[6], const double vector[3]) {
  double product[3]{};
  quadrupole_vector(quadrupole, vector, product);
  return dot3(vector, product);
}

__device__ double logistic(double argument) {
  if (argument >= 0.0) {
    const double exponential = exp(-argument);
    return 1.0 / (1.0 + exponential);
  }
  const double exponential = exp(argument);
  return exponential / (1.0 + exponential);
}

__device__ double multipole_radius_value(const Gfn2NativePeriodicMultipoleDeviceBatch& batch,
                                         std::int64_t atom, double coordination) {
  const double argument =
      g_gfn2_global.multipole_kexp *
      (coordination - batch.multipole_valence_cn[atom] - g_gfn2_global.multipole_shift);
  const double fraction = logistic(argument);
  return batch.multipole_radius[atom] +
         (g_gfn2_global.multipole_rmax - batch.multipole_radius[atom]) * fraction;
}

__device__ double multipole_radius_cn_derivative(
    const Gfn2NativePeriodicMultipoleDeviceBatch& batch, std::int64_t atom, double coordination) {
  const double argument =
      g_gfn2_global.multipole_kexp *
      (coordination - batch.multipole_valence_cn[atom] - g_gfn2_global.multipole_shift);
  const double fraction = logistic(argument);
  /* The periodic CPU path intentionally publishes tblite's signed radius
   * adjoint, which is the negative logistic derivative. */
  return -(g_gfn2_global.multipole_rmax - batch.multipole_radius[atom]) *
         g_gfn2_global.multipole_kexp * fraction * (1.0 - fraction);
}

struct MultipoleTerms {
  double sd[3]{};
  double dd[9]{};
  double sq[6]{};
};

struct MultipoleDerivatives {
  double radius = 0.0;
  double gradient[3]{};
  double strain[9]{};
};

__device__ bool candidate_vector(const Gfn2NativePeriodicMultipoleDeviceBatch& batch,
                                 std::int64_t system, const double rij[3], bool self,
                                 std::int64_t translation_index, double vector[3],
                                 double& squared) {
  const auto& translation = batch.direct_translations[translation_index];
  if (self && is_origin(translation)) return false;
  for (int component = 0; component < 3; ++component) {
    vector[component] = rij[component] - translation.cartesian[component];
    if (!isfinite(vector[component])) return false;
  }
  squared = norm3_squared(vector);
  return isfinite(squared) && squared >= kWignerThreshold;
}

__device__ double smooth_shape(double delta) {
  const double x = fmin(1.0, fmax(0.0, delta) / kWignerToleranceSquared);
  return fmax(0.0, 1.0 - 10.0 * x * x * x + 15.0 * x * x * x * x - 6.0 * x * x * x * x * x);
}

__device__ double smooth_shape_derivative(double delta) {
  const double x = fmin(1.0, fmax(0.0, delta) / kWignerToleranceSquared);
  return -30.0 * x * x * (1.0 - x) * (1.0 - x) / kWignerToleranceSquared;
}

__device__ bool wsc_summary(const Gfn2NativePeriodicMultipoleDeviceBatch& batch,
                            std::int64_t system, const double rij[3], bool self, double& minimum,
                            double& shape_sum, double reference[3], double sum_gradient[3],
                            double sum_strain[9], std::uint32_t* system_errors) {
  const std::int64_t begin = batch.direct_translation_offsets[system];
  const std::int64_t end = batch.direct_translation_offsets[system + 1];
  /* Keep the device code independent of the optional CUDART_INF macro. */
  minimum = __longlong_as_double(0x7ff0000000000000ULL);
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
  for (int component = 0; component < 9; ++component) sum_strain[component] = 0.0;
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

__device__ bool wsc_image(const Gfn2NativePeriodicMultipoleDeviceBatch& batch, std::int64_t system,
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

__device__ bool matrix_terms(const Gfn2NativePeriodicMultipoleDeviceBatch& batch,
                             std::int64_t system, const double vector[3], double radius,
                             double alpha, double volume, MultipoleTerms& result) {
  result = {};
  const double alpha_squared = alpha * alpha;
  const std::int64_t direct_begin = batch.direct_translation_offsets[system];
  const std::int64_t direct_end = batch.direct_translation_offsets[system + 1];
  for (std::int64_t translation_index = direct_begin; translation_index < direct_end;
       ++translation_index) {
    const auto& translation = batch.direct_translations[translation_index];
    const double value[3] = {vector[0] + translation.cartesian[0],
                             vector[1] + translation.cartesian[1],
                             vector[2] + translation.cartesian[2]};
    const double distance = sqrt(norm3_squared(value));
    if (!(distance >= kWignerThreshold)) continue;
    const double inverse = 1.0 / distance;
    const double inverse_squared = inverse * inverse;
    const double inverse_cubed = inverse_squared * inverse;
    const double inverse_fifth = inverse_cubed * inverse_squared;
    const double scaled = radius * inverse;
    const double scaled_squared = scaled * scaled;
    const double damping3 = 1.0 / (1.0 + 6.0 * scaled_squared * scaled);
    const double damping5 = 1.0 / (1.0 + 6.0 * scaled_squared * scaled_squared);
    const double argument = alpha * distance;
    const double argument_squared = argument * argument;
    const double exponential = exp(-argument_squared) / kSqrtPi;
    const double erf_term = -erf(argument) * inverse;
    const double e1 = inverse_squared * (erf_term + 2.0 * exponential * alpha);
    const double e2 = inverse_squared * (e1 + 4.0 * exponential * alpha_squared * alpha / 3.0);
    const double sd_kernel = damping3 * inverse_cubed + e1;
    const double dd_diagonal = damping5 * inverse_cubed + e1;
    const double dd_tensor = damping5 * inverse_fifth + e2;
    for (int axis = 0; axis < 3; ++axis) result.sd[axis] += value[axis] * sd_kernel;
    for (int axis = 0; axis < 3; ++axis) result.dd[axis * 3 + axis] += dd_diagonal;
    for (int row = 0; row < 3; ++row)
      for (int column = 0; column < 3; ++column)
        result.dd[row * 3 + column] -= 3.0 * value[row] * value[column] * dd_tensor;
    double packed[6]{};
    packed_outer3(value, dd_tensor, packed);
    add_scaled6(result.sq, packed);
    const double trace = dd_diagonal / 3.0;
    result.sq[0] -= trace;
    result.sq[2] -= trace;
    result.sq[5] -= trace;
  }

  const double factor = 4.0 * kPi / volume;
  const std::int64_t reciprocal_begin = batch.reciprocal_translation_offsets[system];
  const std::int64_t reciprocal_end = batch.reciprocal_translation_offsets[system + 1];
  for (std::int64_t translation_index = reciprocal_begin; translation_index < reciprocal_end;
       ++translation_index) {
    const auto& translation = batch.reciprocal_translations[translation_index];
    const double g[3] = {translation.cartesian[0], translation.cartesian[1],
                         translation.cartesian[2]};
    const double g2 = norm3_squared(g);
    if (!(g2 >= kBinary64Epsilon)) continue;
    const double exponential = factor * exp(-0.25 * g2 / alpha_squared) / g2;
    const double phase = dot3(vector, g);
    const double sine = sin(phase) * exponential;
    const double cosine = cos(phase) * exponential;
    for (int axis = 0; axis < 3; ++axis) result.sd[axis] += g[axis] * sine;
    for (int row = 0; row < 3; ++row)
      for (int column = 0; column < 3; ++column)
        result.dd[row * 3 + column] += g[row] * g[column] * cosine;
    double packed[6]{};
    packed_outer3(g, -cosine / 3.0, packed);
    add_scaled6(result.sq, packed);
  }
  for (int component = 0; component < 3; ++component)
    if (!isfinite(result.sd[component])) return false;
  for (int component = 0; component < 9; ++component)
    if (!isfinite(result.dd[component])) return false;
  for (int component = 0; component < 6; ++component)
    if (!isfinite(result.sq[component])) return false;
  return true;
}

__device__ bool derivative_terms(const Gfn2NativePeriodicMultipoleDeviceBatch& batch,
                                 std::int64_t system, const double rij[3], double qi, double qj,
                                 const double mi[3], const double mj[3], const double ti[6],
                                 const double tj[6], double radius, double alpha, double volume,
                                 MultipoleDerivatives& result) {
  result = {};
  const double alpha_squared = alpha * alpha;
  double outer_value[9]{};
  const std::int64_t direct_begin = batch.direct_translation_offsets[system];
  const std::int64_t direct_end = batch.direct_translation_offsets[system + 1];
  for (std::int64_t translation_index = direct_begin; translation_index < direct_end;
       ++translation_index) {
    const auto& translation = batch.direct_translations[translation_index];
    const double vector[3] = {rij[0] + translation.cartesian[0], rij[1] + translation.cartesian[1],
                              rij[2] + translation.cartesian[2]};
    const double distance = sqrt(norm3_squared(vector));
    if (!(distance >= kWignerThreshold)) continue;
    const double squared = distance * distance;
    const double inverse = 1.0 / distance;
    const double inverse_squared = inverse * inverse;
    const double g3 = inverse_squared * inverse;
    const double g5 = g3 * inverse_squared;
    const double g7 = g5 * inverse_squared;
    const double argument = alpha * distance;
    const double argument_squared = argument * argument;
    const double exponential = exp(-argument_squared) / kSqrtPi;
    const double erf_term = -erf(argument) * inverse;
    const double e1 = inverse_squared * (erf_term + exponential * (2.0 * alpha_squared) / alpha);
    const double e2 = inverse_squared * (e1 + exponential * (2.0 * alpha_squared) *
                                                  (2.0 * alpha_squared) / (3.0 * alpha));
    const double e3 =
        inverse_squared * (e2 + exponential * (2.0 * alpha_squared) * (2.0 * alpha_squared) *
                                    (2.0 * alpha_squared) / (15.0 * alpha));
    const double scaled = radius * inverse;
    const double scaled_squared = scaled * scaled;
    const double damping3 = 1.0 / (1.0 + 6.0 * scaled_squared * scaled);
    const double damping5 = 1.0 / (1.0 + 6.0 * scaled_squared * scaled_squared);
    const double ddamping3 = -3.0 * damping3 - 3.0 * damping3 * (damping3 - 1.0);
    const double ddamping5 = -5.0 * damping5 - 4.0 * (damping5 * damping5 - damping5);

    const double dpiqj = dot3(vector, mi) * qj;
    const double qidpj = dot3(vector, mj) * qi;
    const double charge_dipole_difference = dpiqj - qidpj;
    double g_sd[3] = {};
    for (int axis = 0; axis < 3; ++axis)
      g_sd[axis] = vector[axis] * (-(ddamping3 * g5) * charge_dipole_difference) +
                   (mj[axis] * qi - mi[axis] * qj) * damping3 * g3;

    double tab[9]{};
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) tab[a * 3 + b] = 3.0 * vector[a] * vector[b] * e2;
      tab[a * 3 + a] -= e1;
    }
    for (int k = 0; k < 3; ++k)
      for (int a = 0; a < 3; ++a)
        g_sd[k] += qj * tab[a * 3 + k] * mi[a] - qi * tab[a * 3 + k] * mj[a];
    result.radius += 3.0 * charge_dipole_difference * g_gfn2_global.multipole_dmp3 * damping3 * g3 *
                     (damping3 / radius) * (radius * inverse) * (radius * inverse) *
                     (radius * inverse);
    add_scaled3(result.gradient, g_sd);
    outer3(vector, g_sd, outer_value);
    add_scaled9(result.strain, outer_value, -0.5);
    outer3(g_sd, vector, outer_value);
    add_scaled9(result.strain, outer_value, -0.5);

    const double dipole_dot = dot3(mj, mi);
    const double dipole_i_projection = dot3(mi, vector);
    const double dipole_j_projection = dot3(mj, vector);
    const double dipole_energy =
        dipole_dot * squared - 3.0 * dipole_j_projection * dipole_i_projection;
    double g_dd[3] = {};
    for (int axis = 0; axis < 3; ++axis) {
      const double dipole_linear = dipole_i_projection * mj[axis] + dipole_j_projection * mi[axis];
      g_dd[axis] = vector[axis] * (-2.0 * damping5 * g5 * dipole_dot) +
                   3.0 * damping5 * g5 * dipole_linear -
                   vector[axis] * dipole_energy * ddamping5 * g7;
    }

    double tabc[3][3][3]{};
    for (int c = 0; c < 3; ++c) {
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) tabc[a][b][c] = -15.0 * vector[a] * vector[b] * vector[c] * e3;
      for (int a = 0; a < 3; ++a) {
        tabc[a][a][c] += 3.0 * e2 * vector[c];
        tabc[c][a][c] += 3.0 * e2 * vector[a];
        tabc[a][c][c] += 3.0 * e2 * vector[a];
      }
    }
    for (int k = 0; k < 3; ++k)
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) g_dd[k] += mi[a] * tabc[b][a][k] * mj[b];
    result.radius += 3.0 * dipole_energy * g_gfn2_global.multipole_dmp5 * damping5 * g5 *
                     (damping5 / radius) * (radius * inverse) * (radius * inverse) *
                     (radius * inverse) * (radius * inverse);
    add_scaled3(result.gradient, g_dd);
    outer3(vector, g_dd, outer_value);
    add_scaled9(result.strain, outer_value, -0.5);
    outer3(g_dd, vector, outer_value);
    add_scaled9(result.strain, outer_value, -0.5);

    double tensor[6]{};
    for (int component = 0; component < 6; ++component)
      tensor[component] = qj * ti[component] + qi * tj[component];
    const double quadrupole_energy =
        tensor[0] * vector[0] * vector[0] + tensor[2] * vector[1] * vector[1] +
        tensor[5] * vector[2] * vector[2] + 2.0 * tensor[1] * vector[0] * vector[1] +
        2.0 * tensor[3] * vector[0] * vector[2] + 2.0 * tensor[4] * vector[1] * vector[2];
    double tj_vector[3]{}, ti_vector[3]{};
    quadrupole_vector(tj, vector, tj_vector);
    quadrupole_vector(ti, vector, ti_vector);
    double g_sq[3]{};
    for (int axis = 0; axis < 3; ++axis)
      g_sq[axis] = vector[axis] * (-quadrupole_energy * ddamping5 * g7) -
                   2.0 * damping5 * g5 * qi * tj_vector[axis] -
                   2.0 * damping5 * g5 * qj * ti_vector[axis];
    for (int k = 0; k < 3; ++k) {
      const double tj_contract = tabc[0][0][k] * tj[0] + 2.0 * tabc[1][0][k] * tj[1] +
                                 2.0 * tabc[2][0][k] * tj[3] + tabc[1][1][k] * tj[2] +
                                 2.0 * tabc[2][1][k] * tj[4] + tabc[2][2][k] * tj[5];
      const double ti_contract = tabc[0][0][k] * ti[0] + 2.0 * tabc[1][0][k] * ti[1] +
                                 2.0 * tabc[2][0][k] * ti[3] + tabc[1][1][k] * ti[2] +
                                 2.0 * tabc[2][1][k] * ti[4] + tabc[2][2][k] * ti[5];
      g_sq[k] += (-qi * tj_contract - qj * ti_contract) / 3.0;
    }
    result.radius += quadrupole_energy * 3.0 * g_gfn2_global.multipole_dmp5 * damping5 * g5 *
                     (damping5 / radius) * (radius * inverse) * (radius * inverse) *
                     (radius * inverse) * (radius * inverse);
    add_scaled3(result.gradient, g_sq);
    outer3(vector, g_sq, outer_value);
    add_scaled9(result.strain, outer_value, -0.5);
    outer3(g_sq, vector, outer_value);
    add_scaled9(result.strain, outer_value, -0.5);
  }

  const double factor = 4.0 * kPi / volume;
  for (std::int64_t translation_index = batch.reciprocal_translation_offsets[system];
       translation_index < batch.reciprocal_translation_offsets[system + 1]; ++translation_index) {
    const auto& translation = batch.reciprocal_translations[translation_index];
    const double g[3] = {translation.cartesian[0], translation.cartesian[1],
                         translation.cartesian[2]};
    const double g2 = norm3_squared(g);
    if (!(g2 >= kBinary64Epsilon)) continue;
    const double exponential = factor * exp(-0.25 * g2 / alpha_squared) / g2;
    const double phase = dot3(rij, g);
    const double cosine = cos(phase) * exponential;
    const double sine = sin(phase) * exponential;
    const double dpiqj = dot3(g, mi) * qj;
    const double qidpj = dot3(g, mj) * qi;
    const double difference = dpiqj - qidpj;
    for (int axis = 0; axis < 3; ++axis) result.gradient[axis] -= g[axis] * cosine * difference;

    double kernel[9]{};
    const double factor_value = 2.0 / g2 + 0.5 / alpha_squared;
    for (int component = 0; component < 9; ++component) {
      const int row = component / 3;
      const int column = component % 3;
      kernel[component] = g[row] * g[column] * factor_value;
    }
    kernel[0] -= 1.0;
    kernel[4] -= 1.0;
    kernel[8] -= 1.0;
    add_scaled9(result.strain, kernel, sine * difference);

    double charge_dipole[3] = {mi[0] * qj - mj[0] * qi, mi[1] * qj - mj[1] * qi,
                               mi[2] * qj - mj[2] * qi};
    double charge_dipole_outer[9]{};
    outer3(g, charge_dipole, charge_dipole_outer);
    outer3(charge_dipole, g, outer_value);
    add_scaled9(charge_dipole_outer, outer_value);
    add_scaled9(result.strain, charge_dipole_outer, -0.5 * sine);

    const double dipole_i_projection = dot3(mi, g);
    const double dipole_j_projection = dot3(mj, g);
    for (int axis = 0; axis < 3; ++axis)
      result.gradient[axis] += g[axis] * sine * dipole_i_projection * dipole_j_projection;
    double dipole_linear[3] = {mi[0] * dipole_j_projection + mj[0] * dipole_i_projection,
                               mi[1] * dipole_j_projection + mj[1] * dipole_i_projection,
                               mi[2] * dipole_j_projection + mj[2] * dipole_i_projection};
    double dipole_outer[9]{};
    outer3(g, dipole_linear, dipole_outer);
    outer3(dipole_linear, g, outer_value);
    add_scaled9(dipole_outer, outer_value);
    add_scaled9(result.strain, kernel, cosine * dipole_i_projection * dipole_j_projection);
    add_scaled9(result.strain, dipole_outer, -0.5 * cosine);

    const double qiqpj = qi * quadrupole_scalar(tj, g);
    const double qpiqj = qj * quadrupole_scalar(ti, g);
    double tj_vector[3]{}, ti_vector[3]{};
    quadrupole_vector(tj, g, tj_vector);
    quadrupole_vector(ti, g, ti_vector);
    const double charge_quadrupole = qiqpj + qpiqj;
    for (int axis = 0; axis < 3; ++axis)
      result.gradient[axis] -= g[axis] * sine * charge_quadrupole / 3.0;
    add_scaled9(result.strain, kernel, -cosine * charge_quadrupole / 3.0);
    const double quad_linear[3] = {qi * tj_vector[0] + qj * ti_vector[0],
                                   qi * tj_vector[1] + qj * ti_vector[1],
                                   qi * tj_vector[2] + qj * ti_vector[2]};
    double quad_outer[9]{};
    outer3(g, quad_linear, quad_outer);
    outer3(quad_linear, g, outer_value);
    add_scaled9(quad_outer, outer_value);
    add_scaled9(result.strain, quad_outer, cosine / 3.0);
  }
  if (!isfinite(result.radius)) return false;
  for (int component = 0; component < 3; ++component)
    if (!isfinite(result.gradient[component])) return false;
  for (int component = 0; component < 9; ++component)
    if (!isfinite(result.strain[component])) return false;
  return true;
}

__device__ double pair_energy(double qi, double qj, const double mi[3], const double mj[3],
                              const double ti[6], const double tj[6], const MultipoleTerms& terms) {
  const double charge_dipole[3] = {mi[0] * qj - mj[0] * qi, mi[1] * qj - mj[1] * qi,
                                   mi[2] * qj - mj[2] * qi};
  const double dd_first[3] = {terms.dd[0] * mj[0] + terms.dd[1] * mj[1] + terms.dd[2] * mj[2],
                              terms.dd[3] * mj[0] + terms.dd[4] * mj[1] + terms.dd[5] * mj[2],
                              terms.dd[6] * mj[0] + terms.dd[7] * mj[1] + terms.dd[8] * mj[2]};
  double charge_quadrupole[6]{};
  for (int component = 0; component < 6; ++component)
    charge_quadrupole[component] = qi * tj[component] + qj * ti[component];
  return dot3(charge_dipole, terms.sd) + dot3(mi, dd_first) +
         packed_dot6(terms.sq, charge_quadrupole);
}

__device__ void add_matrix(double* sd, double* dd, double* sq, std::int64_t pair,
                           const MultipoleTerms& terms, double weight) {
  for (int component = 0; component < 3; ++component)
    sd[3 * pair + component] += weight * terms.sd[component];
  for (int component = 0; component < 9; ++component)
    dd[9 * pair + component] += weight * terms.dd[component];
  for (int component = 0; component < 6; ++component)
    sq[6 * pair + component] += weight * terms.sq[component];
}

__device__ void accumulate_potentials_and_energy(
    const Gfn2NativePeriodicMultipoleDeviceBatch& batch,
    const Gfn2NativePeriodicMultipoleDeviceWorkspace& workspace, std::int64_t system,
    double& energy) {
  const std::int64_t atom_begin = batch.topology.atom_offsets[system];
  const std::int64_t atom_end = batch.topology.atom_offsets[system + 1];
  const std::int64_t atom_count = atom_end - atom_begin;
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const double quadrupole_scale[6] = {1.0, 2.0, 1.0, 2.0, 2.0, 1.0};

  for (std::int64_t local = 0; local < atom_count; ++local) {
    const std::int64_t atom = atom_begin + local;
    workspace.charge_potentials[atom] = 0.0;
    for (int component = 0; component < 3; ++component)
      workspace.dipole_potentials[3 * atom + component] = 0.0;
    for (int component = 0; component < 6; ++component)
      workspace.quadrupole_potentials[6 * atom + component] = 0.0;
  }

  for (std::int64_t local_row = 0; local_row < atom_count; ++local_row) {
    const std::int64_t row_atom = atom_begin + local_row;
    const double* row_dipole = batch.atomic_dipoles + 3 * row_atom;
    const double* row_quadrupole = batch.atomic_quadrupoles + 6 * row_atom;
    for (std::int64_t local_column = 0; local_column < atom_count; ++local_column) {
      const std::int64_t column_atom = atom_begin + local_column;
      const std::int64_t pair = matrix_begin + local_row * atom_count + local_column;
      const double* charge_dipole = workspace.charge_dipole_matrix + 3 * pair;
      const double* dipole_matrix = workspace.dipole_dipole_matrix + 9 * pair;
      const double* charge_quad = workspace.charge_quadrupole_matrix + 6 * pair;

      workspace.charge_potentials[column_atom] +=
          dot3(charge_dipole, row_dipole) + packed_dot6(charge_quad, row_quadrupole);
      for (int row_component = 0; row_component < 3; ++row_component) {
        workspace.dipole_potentials[3 * row_atom + row_component] +=
            charge_dipole[row_component] * batch.atomic_charges[column_atom];
        for (int column_component = 0; column_component < 3; ++column_component) {
          workspace.dipole_potentials[3 * row_atom + row_component] +=
              dipole_matrix[3 * row_component + column_component] *
              batch.atomic_dipoles[3 * column_atom + column_component];
        }
      }
      for (int component = 0; component < 6; ++component)
        workspace.quadrupole_potentials[6 * row_atom + component] +=
            charge_quad[component] * batch.atomic_charges[column_atom];

      energy += dot3(row_dipole, charge_dipole) * batch.atomic_charges[column_atom];
      double dd_energy = 0.0;
      for (int row_component = 0; row_component < 3; ++row_component)
        for (int column_component = 0; column_component < 3; ++column_component)
          dd_energy += batch.atomic_dipoles[3 * row_atom + row_component] *
                       dipole_matrix[3 * row_component + column_component] *
                       batch.atomic_dipoles[3 * column_atom + column_component];
      energy += 0.5 * dd_energy;
      energy += packed_dot6(charge_quad, row_quadrupole) * batch.atomic_charges[column_atom];
    }
  }

  for (std::int64_t local = 0; local < atom_count; ++local) {
    const std::int64_t atom = atom_begin + local;
    const double dipole_kernel = batch.dipole_kernel[atom];
    const double quadrupole_kernel = batch.quadrupole_kernel[atom];
    const double* dipole = batch.atomic_dipoles + 3 * atom;
    const double* quadrupole = batch.atomic_quadrupoles + 6 * atom;
    for (int component = 0; component < 3; ++component)
      workspace.dipole_potentials[3 * atom + component] += 2.0 * dipole_kernel * dipole[component];
    for (int component = 0; component < 6; ++component)
      workspace.quadrupole_potentials[6 * atom + component] +=
          2.0 * quadrupole_kernel * quadrupole_scale[component] * quadrupole[component];
    energy += dipole_kernel * dot3(dipole, dipole);
    double quadrupole_norm = 0.0;
    for (int component = 0; component < 6; ++component)
      quadrupole_norm +=
          quadrupole_scale[component] * quadrupole[component] * quadrupole[component];
    energy += quadrupole_kernel * quadrupole_norm;
  }
}

__device__ bool finite_values(const double* values, std::int64_t count) {
  for (std::int64_t index = 0; index < count; ++index)
    if (!isfinite(values[index])) return false;
  return true;
}

__global__ void native_periodic_multipole_kernel(
    Gfn2NativePeriodicMultipoleDeviceBatch batch,
    Gfn2NativePeriodicMultipoleDeviceWorkspace workspace, double* charge_dipole_matrix,
    double* dipole_dipole_matrix, double* charge_quadrupole_matrix, double* charge_potentials,
    double* dipole_potentials, double* quadrupole_potentials, double* energies, double* gradients,
    double* strain, double* coordination_adjoint, std::uint32_t* system_errors,
    std::uint32_t* device_error) {
  if (threadIdx.x != 0) return;
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (system >= batch.topology.batch_size) return;
  if (batch.active_mask != nullptr && batch.active_mask[system] == 0u) return;
  if (batch.active_mask != nullptr && batch.active_mask[system] == 0u) return;

  const std::int64_t atom_begin = batch.topology.atom_offsets[system];
  const std::int64_t atom_end = batch.topology.atom_offsets[system + 1];
  const std::int64_t atom_count = atom_end - atom_begin;
  const std::int64_t matrix_begin = batch.matrix_offsets[system];
  const std::int64_t matrix_end = batch.matrix_offsets[system + 1];
  const std::int64_t direct_begin = batch.direct_translation_offsets[system];
  const std::int64_t direct_end = batch.direct_translation_offsets[system + 1];
  const std::int64_t reciprocal_begin = batch.reciprocal_translation_offsets[system];
  const std::int64_t reciprocal_end = batch.reciprocal_translation_offsets[system + 1];
  std::int64_t expected_matrix_elements = 0;
  const bool matrix_count_safe =
      atom_count >= 0 && (atom_count == 0 || atom_count <= kMaxInt64 / atom_count);
  if (matrix_count_safe) expected_matrix_elements = atom_count * atom_count;
  if (atom_begin < 0 || atom_begin > atom_end || atom_end > batch.topology.total_atoms ||
      matrix_begin < 0 || matrix_begin > matrix_end || matrix_end > batch.matrix_elements ||
      direct_begin < 0 || direct_begin > direct_end ||
      direct_end > batch.direct_translation_elements || reciprocal_begin < 0 ||
      reciprocal_begin > reciprocal_end || reciprocal_end > batch.reciprocal_translation_elements ||
      !matrix_count_safe || matrix_end - matrix_begin != expected_matrix_elements ||
      batch.topology.periodic_axes[system] != XTBLOOM_PERIODIC_AXES_XYZ) {
    record_system_error(system_errors, system, DeviceError::kInvalidOffsets);
    record_device_error(device_error, DeviceError::kInvalidTopology);
    return;
  }

  const double alpha = batch.alphas[system];
  const double volume = batch.volumes[system];
  if (!(alpha > 0.0) || !isfinite(alpha)) {
    record_system_error(system_errors, system, DeviceError::kInvalidAlpha);
    return;
  }
  if (!(volume > 0.0) || !isfinite(volume)) {
    record_system_error(system_errors, system, DeviceError::kInvalidVolume);
    return;
  }
  for (std::int64_t translation = direct_begin; translation < direct_end; ++translation) {
    const auto& value = batch.direct_translations[translation];
    if (!isfinite(value.cartesian[0]) || !isfinite(value.cartesian[1]) ||
        !isfinite(value.cartesian[2])) {
      record_system_error(system_errors, system, DeviceError::kInvalidTopology);
      record_device_error(device_error, DeviceError::kInvalidTopology);
      return;
    }
  }
  for (std::int64_t translation = reciprocal_begin; translation < reciprocal_end; ++translation) {
    const auto& value = batch.reciprocal_translations[translation];
    if (!isfinite(value.cartesian[0]) || !isfinite(value.cartesian[1]) ||
        !isfinite(value.cartesian[2])) {
      record_system_error(system_errors, system, DeviceError::kInvalidTopology);
      record_device_error(device_error, DeviceError::kInvalidTopology);
      return;
    }
  }

  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    const double* position = batch.positions + 3 * atom;
    if (!finite_values(position, 3) ||
        !wrap_periodic_position(batch.topology, system, position,
                                 workspace.wrapped_positions + 3 * atom)) {
      record_system_error(system_errors, system, DeviceError::kNonfinitePosition);
      return;
    }
    if (!isfinite(batch.coordination_numbers[atom])) {
      record_system_error(system_errors, system, DeviceError::kNonfiniteCoordination);
      return;
    }
    if (!isfinite(batch.atomic_charges[atom])) {
      record_system_error(system_errors, system, DeviceError::kNonfiniteCharge);
      return;
    }
    if (!finite_values(batch.atomic_dipoles + 3 * atom, 3)) {
      record_system_error(system_errors, system, DeviceError::kNonfiniteDipole);
      return;
    }
    if (!finite_values(batch.atomic_quadrupoles + 6 * atom, 6)) {
      record_system_error(system_errors, system, DeviceError::kNonfiniteQuadrupole);
      return;
    }
    if (!(batch.multipole_radius[atom] > 0.0) ||
        batch.multipole_radius[atom] > g_gfn2_global.multipole_rmax ||
        !isfinite(batch.multipole_radius[atom]) || !(batch.multipole_valence_cn[atom] > 0.0) ||
        !isfinite(batch.multipole_valence_cn[atom]) || !isfinite(batch.dipole_kernel[atom]) ||
        !isfinite(batch.quadrupole_kernel[atom])) {
      record_system_error(system_errors, system, DeviceError::kInvalidKernel);
      return;
    }
  }

  double* system_gradient = workspace.gradients + 3 * atom_begin;
  double* system_strain = workspace.strain + 9 * system;
  for (std::int64_t index = matrix_begin; index < matrix_end; ++index) {
    for (int component = 0; component < 3; ++component)
      workspace.charge_dipole_matrix[3 * index + component] = 0.0;
    for (int component = 0; component < 9; ++component)
      workspace.dipole_dipole_matrix[9 * index + component] = 0.0;
    for (int component = 0; component < 6; ++component)
      workspace.charge_quadrupole_matrix[6 * index + component] = 0.0;
  }
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    workspace.charge_potentials[atom] = 0.0;
    for (int component = 0; component < 3; ++component)
      workspace.dipole_potentials[3 * atom + component] = 0.0;
    for (int component = 0; component < 6; ++component)
      workspace.quadrupole_potentials[6 * atom + component] = 0.0;
    for (int component = 0; component < 3; ++component)
      system_gradient[3 * (atom - atom_begin) + component] = 0.0;
    workspace.coordination_adjoint[atom] = 0.0;
  }
  workspace.energies[system] = 0.0;
  for (int component = 0; component < 9; ++component) system_strain[component] = 0.0;

  for (std::int64_t center = 0; center < atom_count; ++center) {
    for (std::int64_t image = 0; image < center; ++image) {
      const double* center_position = workspace.wrapped_positions + 3 * (atom_begin + center);
      const double* image_position = workspace.wrapped_positions + 3 * (atom_begin + image);
      const double rij[3] = {center_position[0] - image_position[0],
                             center_position[1] - image_position[1],
                             center_position[2] - image_position[2]};
      double minimum = 0.0, shape_sum = 0.0, reference[3]{}, sum_gradient[3]{}, sum_strain[9]{};
      if (!wsc_summary(batch, system, rij, false, minimum, shape_sum, reference, sum_gradient,
                       sum_strain, system_errors))
        return;
      const std::int64_t first = atom_begin + center;
      const std::int64_t second = atom_begin + image;
      const double radius =
          0.5 * (multipole_radius_value(batch, first, batch.coordination_numbers[first]) +
                 multipole_radius_value(batch, second, batch.coordination_numbers[second]));
      if (!(radius > 0.0) || !isfinite(radius)) {
        record_system_error(system_errors, system, DeviceError::kInvalidRadius);
        return;
      }
      const double* mi = batch.atomic_dipoles + 3 * first;
      const double* mj = batch.atomic_dipoles + 3 * second;
      const double* ti = batch.atomic_quadrupoles + 6 * first;
      const double* tj = batch.atomic_quadrupoles + 6 * second;
      for (std::int64_t translation = direct_begin; translation < direct_end; ++translation) {
        double vector[3]{}, weight = 0.0, weight_gradient[3]{}, weight_strain[9]{};
        if (!wsc_image(batch, system, rij, false, translation, minimum, shape_sum, reference,
                       sum_gradient, sum_strain, vector, weight, weight_gradient, weight_strain))
          continue;
        if (!(weight > 0.0)) continue;
        double reverse_vector[3] = {-vector[0], -vector[1], -vector[2]};
        MultipoleTerms forward{}, reverse{};
        if (!matrix_terms(batch, system, vector, radius, alpha, volume, forward) ||
            !matrix_terms(batch, system, reverse_vector, radius, alpha, volume, reverse)) {
          record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
          return;
        }
        const std::int64_t forward_pair = matrix_begin + image * atom_count + center;
        const std::int64_t reverse_pair = matrix_begin + center * atom_count + image;
        add_matrix(workspace.charge_dipole_matrix, workspace.dipole_dipole_matrix,
                   workspace.charge_quadrupole_matrix, forward_pair, forward, weight);
        add_matrix(workspace.charge_dipole_matrix, workspace.dipole_dipole_matrix,
                   workspace.charge_quadrupole_matrix, reverse_pair, reverse, weight);

        const double derivative_vector[3] = {-vector[0], -vector[1], -vector[2]};
        MultipoleDerivatives derivative{};
        if (!derivative_terms(batch, system, derivative_vector, batch.atomic_charges[first],
                              batch.atomic_charges[second], mi, mj, ti, tj, radius, alpha, volume,
                              derivative)) {
          record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
          return;
        }
        MultipoleTerms energy_terms{};
        if (!matrix_terms(batch, system, derivative_vector, radius, alpha, volume, energy_terms)) {
          record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
          return;
        }
        const double image_energy =
            pair_energy(batch.atomic_charges[first], batch.atomic_charges[second], mi, mj, ti, tj,
                        energy_terms);
        if (!isfinite(image_energy)) {
          record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
          return;
        }
        double weighted_gradient[3] = {
            weight * derivative.gradient[0] + image_energy * weight_gradient[0],
            weight * derivative.gradient[1] + image_energy * weight_gradient[1],
            weight * derivative.gradient[2] + image_energy * weight_gradient[2]};
        for (int axis = 0; axis < 3; ++axis) {
          workspace.gradients[3 * first + axis] += weighted_gradient[axis];
          workspace.gradients[3 * second + axis] -= weighted_gradient[axis];
        }
        for (int component = 0; component < 9; ++component)
          system_strain[component] +=
              weight * derivative.strain[component] + image_energy * weight_strain[component];
        workspace.coordination_adjoint[first] += weight * derivative.radius;
        workspace.coordination_adjoint[second] += weight * derivative.radius;
      }
    }

    const std::int64_t atom = atom_begin + center;
    const double radius = multipole_radius_value(batch, atom, batch.coordination_numbers[atom]);
    if (!(radius > 0.0) || !isfinite(radius)) {
      record_system_error(system_errors, system, DeviceError::kInvalidRadius);
      return;
    }
    const double* dipole = batch.atomic_dipoles + 3 * atom;
    const double* quadrupole = batch.atomic_quadrupoles + 6 * atom;
    const double zero[3] = {0.0, 0.0, 0.0};
    double minimum = 0.0, shape_sum = 0.0, reference[3]{}, sum_gradient[3]{}, sum_strain[9]{};
    if (!wsc_summary(batch, system, zero, true, minimum, shape_sum, reference, sum_gradient,
                     sum_strain, system_errors))
      return;
    for (std::int64_t translation = direct_begin; translation < direct_end; ++translation) {
      double vector[3]{}, weight = 0.0, weight_gradient[3]{}, weight_strain[9]{};
      if (!wsc_image(batch, system, zero, true, translation, minimum, shape_sum, reference,
                     sum_gradient, sum_strain, vector, weight, weight_gradient, weight_strain))
        continue;
      if (!(weight > 0.0)) continue;
      MultipoleTerms matrix{};
      if (!matrix_terms(batch, system, vector, radius, alpha, volume, matrix)) {
        record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
        return;
      }
      const std::int64_t pair = matrix_begin + center * atom_count + center;
      add_matrix(workspace.charge_dipole_matrix, workspace.dipole_dipole_matrix,
                 workspace.charge_quadrupole_matrix, pair, matrix, weight);
      const double derivative_vector[3] = {-vector[0], -vector[1], -vector[2]};
      MultipoleDerivatives derivative{};
      if (!derivative_terms(batch, system, derivative_vector, batch.atomic_charges[atom],
                            batch.atomic_charges[atom], dipole, dipole, quadrupole, quadrupole,
                            radius, alpha, volume, derivative)) {
        record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
        return;
      }
      MultipoleTerms energy_terms{};
      if (!matrix_terms(batch, system, derivative_vector, radius, alpha, volume, energy_terms)) {
        record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
        return;
      }
      const double image_energy =
          pair_energy(batch.atomic_charges[atom], batch.atomic_charges[atom], dipole, dipole,
                      quadrupole, quadrupole, energy_terms);
      if (!isfinite(image_energy)) {
        record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
        return;
      }
      for (int component = 0; component < 9; ++component)
        system_strain[component] +=
            0.5 * (weight * derivative.strain[component] + image_energy * weight_strain[component]);
      workspace.coordination_adjoint[atom] += weight * derivative.radius;
    }
  }

  const double self_dd = -4.0 * alpha * alpha * alpha / (3.0 * kSqrtPi);
  const double self_sq = 4.0 * alpha * alpha * alpha / (9.0 * kSqrtPi);
  for (std::int64_t local = 0; local < atom_count; ++local) {
    const std::int64_t pair = matrix_begin + local * atom_count + local;
    workspace.dipole_dipole_matrix[9 * pair] += self_dd;
    workspace.dipole_dipole_matrix[9 * pair + 4] += self_dd;
    workspace.dipole_dipole_matrix[9 * pair + 8] += self_dd;
    workspace.charge_quadrupole_matrix[6 * pair] += self_sq;
    workspace.charge_quadrupole_matrix[6 * pair + 2] += self_sq;
    workspace.charge_quadrupole_matrix[6 * pair + 5] += self_sq;
  }

  double energy = 0.0;
  accumulate_potentials_and_energy(batch, workspace, system, energy);
  workspace.energies[system] = energy;
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom)
    workspace.coordination_adjoint[atom] *=
        multipole_radius_cn_derivative(batch, atom, batch.coordination_numbers[atom]);

  const std::int64_t matrix_count = matrix_end - matrix_begin;
  if (!finite_values(workspace.charge_dipole_matrix + 3 * matrix_begin, 3 * matrix_count) ||
      !finite_values(workspace.dipole_dipole_matrix + 9 * matrix_begin, 9 * matrix_count) ||
      !finite_values(workspace.charge_quadrupole_matrix + 6 * matrix_begin, 6 * matrix_count) ||
      !finite_values(workspace.charge_potentials + atom_begin, atom_count) ||
      !finite_values(workspace.dipole_potentials + 3 * atom_begin, 3 * atom_count) ||
      !finite_values(workspace.quadrupole_potentials + 6 * atom_begin, 6 * atom_count) ||
      !finite_values(workspace.gradients + 3 * atom_begin, 3 * atom_count) ||
      !finite_values(workspace.coordination_adjoint + atom_begin, atom_count) ||
      !finite_values(workspace.energies + system, 1) || !finite_values(system_strain, 9)) {
    record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
    return;
  }

  /* Commit only after the entire peer is finite. */
  for (std::int64_t index = matrix_begin; index < matrix_end; ++index) {
    if (charge_dipole_matrix != nullptr) {
      for (int component = 0; component < 3; ++component)
        charge_dipole_matrix[3 * index + component] =
            workspace.charge_dipole_matrix[3 * index + component];
    }
    if (dipole_dipole_matrix != nullptr) {
      for (int component = 0; component < 9; ++component)
        dipole_dipole_matrix[9 * index + component] =
            workspace.dipole_dipole_matrix[9 * index + component];
    }
    if (charge_quadrupole_matrix != nullptr) {
      for (int component = 0; component < 6; ++component)
        charge_quadrupole_matrix[6 * index + component] =
            workspace.charge_quadrupole_matrix[6 * index + component];
    }
  }
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    if (charge_potentials != nullptr) charge_potentials[atom] = workspace.charge_potentials[atom];
    if (dipole_potentials != nullptr) {
      for (int component = 0; component < 3; ++component)
        dipole_potentials[3 * atom + component] = workspace.dipole_potentials[3 * atom + component];
    }
    if (quadrupole_potentials != nullptr) {
      for (int component = 0; component < 6; ++component)
        quadrupole_potentials[6 * atom + component] =
            workspace.quadrupole_potentials[6 * atom + component];
    }
    if (gradients != nullptr) {
      for (int component = 0; component < 3; ++component)
        gradients[3 * atom + component] = workspace.gradients[3 * atom + component];
    }
    if (coordination_adjoint != nullptr) coordination_adjoint[atom] = workspace.coordination_adjoint[atom];
  }
  if (energies != nullptr) energies[system] = workspace.energies[system];
  if (strain != nullptr) {
    for (int component = 0; component < 9; ++component)
      strain[9 * system + component] = system_strain[component];
  }
}

}  // namespace

cudaError_t reset_gfn2_native_periodic_multipole_errors_cuda(std::int64_t batch_size,
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

cudaError_t evaluate_gfn2_native_periodic_multipole_cuda(
    const Gfn2NativePeriodicMultipoleDeviceBatch& batch,
    const Gfn2NativePeriodicMultipoleDeviceWorkspace& workspace, double* charge_dipole_matrix,
    double* dipole_dipole_matrix, double* charge_quadrupole_matrix, double* charge_potentials,
    double* dipole_potentials, double* quadrupole_potentials, double* energies, double* gradients,
    double* strain, double* coordination_adjoint, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  const auto& topology = batch.topology;
  if (topology.batch_size <= 0 || topology.total_atoms < 0 || topology.total_translations <= 0 ||
      topology.batch_size > static_cast<std::int64_t>((std::numeric_limits<unsigned int>::max)()) ||
      topology.atom_offset_count != topology.batch_size + 1 ||
      topology.translation_offset_count != topology.batch_size + 1 ||
      topology.cell_elements != topology.batch_size * 9 ||
      topology.periodic_axes_elements != topology.batch_size || topology.plan_token == 0u ||
      topology.atom_offsets == nullptr || topology.cell_matrices == nullptr ||
      topology.periodic_axes == nullptr || topology.translation_offsets == nullptr ||
      topology.translations == nullptr || batch.matrix_offset_elements != topology.batch_size + 1 ||
      batch.matrix_elements <= 0 || batch.volume_elements != topology.batch_size ||
      batch.alpha_elements != topology.batch_size ||
      batch.direct_translation_offset_elements != topology.batch_size + 1 ||
      batch.reciprocal_translation_offset_elements != topology.batch_size + 1 ||
      batch.direct_translation_elements <= 0 || batch.reciprocal_translation_elements <= 0 ||
      batch.dipole_kernel_elements != topology.total_atoms ||
      batch.quadrupole_kernel_elements != topology.total_atoms ||
      batch.multipole_radius_elements != topology.total_atoms ||
      batch.multipole_valence_cn_elements != topology.total_atoms ||
      workspace.plan_token != topology.plan_token ||
      workspace.energy_elements != topology.batch_size || system_errors == nullptr ||
      device_error == nullptr) {
    return cudaErrorInvalidValue;
  }
  if ((batch.active_mask == nullptr) != (batch.active_mask_elements == 0) ||
      (batch.active_mask != nullptr && batch.active_mask_elements != topology.batch_size)) {
    return cudaErrorInvalidValue;
  }

  std::int64_t coordinate_elements = 0;
  std::int64_t quadrupole_elements = 0;
  std::int64_t strain_elements = 0;
  if (!checked_multiply(topology.total_atoms, 3, &coordinate_elements) ||
      !checked_multiply(topology.total_atoms, 6, &quadrupole_elements) ||
      !checked_multiply(topology.batch_size, 9, &strain_elements) ||
      batch.position_elements != coordinate_elements ||
      batch.coordination_number_elements != topology.total_atoms ||
      batch.atomic_charge_elements != topology.total_atoms ||
      batch.atomic_dipole_elements != coordinate_elements ||
      batch.atomic_quadrupole_elements != quadrupole_elements ||
      workspace.wrapped_position_elements != coordinate_elements ||
      workspace.charge_potential_elements != topology.total_atoms ||
      workspace.dipole_potential_elements != coordinate_elements ||
      workspace.quadrupole_potential_elements != quadrupole_elements ||
      workspace.gradient_elements != coordinate_elements ||
      workspace.strain_elements != strain_elements ||
      workspace.coordination_adjoint_elements != topology.total_atoms ||
      batch.matrix_offsets == nullptr || batch.volumes == nullptr || batch.alphas == nullptr ||
      batch.direct_translation_offsets == nullptr || batch.direct_translations == nullptr ||
      batch.reciprocal_translation_offsets == nullptr || batch.reciprocal_translations == nullptr ||
      batch.dipole_kernel == nullptr || batch.quadrupole_kernel == nullptr ||
      batch.multipole_radius == nullptr || batch.multipole_valence_cn == nullptr ||
      batch.positions == nullptr || batch.coordination_numbers == nullptr ||
      batch.atomic_charges == nullptr || batch.atomic_dipoles == nullptr ||
      batch.atomic_quadrupoles == nullptr || workspace.wrapped_positions == nullptr ||
      workspace.charge_dipole_matrix == nullptr || workspace.dipole_dipole_matrix == nullptr ||
      workspace.charge_quadrupole_matrix == nullptr || workspace.charge_potentials == nullptr ||
      workspace.dipole_potentials == nullptr || workspace.quadrupole_potentials == nullptr ||
      workspace.energies == nullptr || workspace.gradients == nullptr ||
      workspace.strain == nullptr || workspace.coordination_adjoint == nullptr) {
    return cudaErrorInvalidValue;
  }
  std::int64_t matrix_dipole_elements = 0;
  std::int64_t matrix_dd_elements = 0;
  std::int64_t matrix_quadrupole_elements = 0;
  if (!checked_multiply(batch.matrix_elements, 3, &matrix_dipole_elements) ||
      !checked_multiply(batch.matrix_elements, 9, &matrix_dd_elements) ||
      !checked_multiply(batch.matrix_elements, 6, &matrix_quadrupole_elements) ||
      workspace.charge_dipole_matrix_elements != matrix_dipole_elements ||
      workspace.dipole_dipole_matrix_elements != matrix_dd_elements ||
      workspace.charge_quadrupole_matrix_elements != matrix_quadrupole_elements) {
    return cudaErrorInvalidValue;
  }

  std::array<AddressRange, 45> ranges{};
  if (!make_range(topology.atom_offsets, topology.atom_offset_count, &ranges[0]) ||
      !make_range(topology.cell_matrices, topology.cell_elements, &ranges[1]) ||
      !make_range(topology.periodic_axes, topology.periodic_axes_elements, &ranges[2]) ||
      !make_range(topology.translation_offsets, topology.translation_offset_count, &ranges[3]) ||
      !make_range(topology.translations, topology.total_translations, &ranges[4]) ||
      !make_range(batch.matrix_offsets, batch.matrix_offset_elements, &ranges[5]) ||
      !make_range(batch.volumes, batch.volume_elements, &ranges[6]) ||
      !make_range(batch.alphas, batch.alpha_elements, &ranges[7]) ||
      !make_range(batch.direct_translation_offsets, batch.direct_translation_offset_elements,
                  &ranges[8]) ||
      !make_range(batch.direct_translations, batch.direct_translation_elements, &ranges[9]) ||
      !make_range(batch.reciprocal_translation_offsets,
                  batch.reciprocal_translation_offset_elements, &ranges[10]) ||
      !make_range(batch.reciprocal_translations, batch.reciprocal_translation_elements,
                  &ranges[11]) ||
      !make_range(batch.dipole_kernel, batch.dipole_kernel_elements, &ranges[12]) ||
      !make_range(batch.quadrupole_kernel, batch.quadrupole_kernel_elements, &ranges[13]) ||
      !make_range(batch.multipole_radius, batch.multipole_radius_elements, &ranges[14]) ||
      !make_range(batch.multipole_valence_cn, batch.multipole_valence_cn_elements, &ranges[15]) ||
      !make_range(batch.positions, batch.position_elements, &ranges[16]) ||
      !make_range(batch.coordination_numbers, batch.coordination_number_elements, &ranges[17]) ||
      !make_range(batch.atomic_charges, batch.atomic_charge_elements, &ranges[18]) ||
      !make_range(batch.atomic_dipoles, batch.atomic_dipole_elements, &ranges[19]) ||
      !make_range(batch.atomic_quadrupoles, batch.atomic_quadrupole_elements, &ranges[20]) ||
      !make_range(batch.active_mask, batch.active_mask_elements, &ranges[44]) ||
      !make_range(workspace.wrapped_positions, workspace.wrapped_position_elements, &ranges[21]) ||
      !make_range(workspace.charge_dipole_matrix, workspace.charge_dipole_matrix_elements,
                  &ranges[22]) ||
      !make_range(workspace.dipole_dipole_matrix, workspace.dipole_dipole_matrix_elements,
                  &ranges[23]) ||
      !make_range(workspace.charge_quadrupole_matrix, workspace.charge_quadrupole_matrix_elements,
                  &ranges[24]) ||
      !make_range(workspace.charge_potentials, workspace.charge_potential_elements, &ranges[25]) ||
      !make_range(workspace.dipole_potentials, workspace.dipole_potential_elements, &ranges[26]) ||
      !make_range(workspace.quadrupole_potentials, workspace.quadrupole_potential_elements,
                  &ranges[27]) ||
      !make_range(workspace.energies, workspace.energy_elements, &ranges[28]) ||
      !make_range(workspace.gradients, workspace.gradient_elements, &ranges[29]) ||
      !make_range(workspace.strain, workspace.strain_elements, &ranges[30]) ||
      !make_range(workspace.coordination_adjoint, workspace.coordination_adjoint_elements,
                  &ranges[31]) ||
      !make_optional_range(charge_dipole_matrix, matrix_dipole_elements, &ranges[32]) ||
      !make_optional_range(dipole_dipole_matrix, matrix_dd_elements, &ranges[33]) ||
      !make_optional_range(charge_quadrupole_matrix, matrix_quadrupole_elements, &ranges[34]) ||
      !make_optional_range(charge_potentials, topology.total_atoms, &ranges[35]) ||
      !make_optional_range(dipole_potentials, coordinate_elements, &ranges[36]) ||
      !make_optional_range(quadrupole_potentials, quadrupole_elements, &ranges[37]) ||
      !make_optional_range(energies, topology.batch_size, &ranges[38]) ||
      !make_optional_range(gradients, coordinate_elements, &ranges[39]) ||
      !make_optional_range(strain, strain_elements, &ranges[40]) ||
      !make_optional_range(coordination_adjoint, topology.total_atoms, &ranges[41]) ||
      !make_range(system_errors, topology.batch_size, &ranges[42]) ||
      !make_range(device_error, 1, &ranges[43]) || !all_disjoint(ranges)) {
    return cudaErrorInvalidValue;
  }

  native_periodic_multipole_kernel<<<static_cast<unsigned int>(topology.batch_size), 1, 0,
                                     stream>>>(
      batch, workspace, charge_dipole_matrix, dipole_dipole_matrix, charge_quadrupole_matrix,
      charge_potentials, dipole_potentials, quadrupole_potentials, energies, gradients, strain,
      coordination_adjoint, system_errors, device_error);
  return cudaGetLastError();
}

}  // namespace xtbloom::detail::cuda
