// SPDX-License-Identifier: GPL-3.0-or-later
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/cuda_atomics.cuh"
#include "backends/cuda/gfn2_native_periodic_d4.cuh"
#include "backends/cuda/periodic_wrap.cuh"

namespace xtbloom::detail::cuda {
namespace {

using DeviceError = Gfn2NativePeriodicD4DeviceError;

constexpr int kThreadsPerBlock = 1;
constexpr double kCoordinationCutoff = 30.0;
constexpr double kTwoBodyCutoff = 50.0;
constexpr double kAtmCutoff = 25.0;
constexpr double kCutoffSwitchWidth = 0.05;
constexpr double kMinimumDistanceSquared = 1.0e-12;
constexpr double kCoordinationSteepness = 7.5;
constexpr double kEnK4 = 4.10451;
constexpr double kEnK5 = 19.08857;
constexpr double kEnK6 = 2.0 * 11.28174 * 11.28174;
constexpr double kChargeScalingHeight = 3.0;
constexpr double kChargeScalingSteepness = 2.0;
constexpr double kReferenceWeightFactor = 6.0;
constexpr double kMinimumWeightNorm = 1.4916681462400413e-154;
constexpr double kAtmExponent = 16.0;
constexpr double kInverseSqrtPi = 0.5641895835477562869480794515607726;
constexpr double kDispersionS6 = 1.0;
constexpr double kDispersionS8 = 2.7;
constexpr double kDispersionA1 = 0.52;
constexpr double kDispersionA2 = 5.0;
constexpr double kDispersionS9 = 5.0;
constexpr std::int64_t kMaximumInt64 = (std::numeric_limits<std::int64_t>::max)();

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

bool checked_multiply(std::int64_t first, std::int64_t second, std::int64_t* result) noexcept {
  if (result == nullptr || first < 0 || second < 0 ||
      (second != 0 && first > kMaximumInt64 / second)) {
    return false;
  }
  *result = first * second;
  return true;
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    DeviceError error) {
  const auto expected = static_cast<std::uint32_t>(DeviceError::kSuccess);
  atomicCAS(system_errors + system, expected, static_cast<std::uint32_t>(error));
}

__device__ void record_device_error(std::uint32_t* device_error, DeviceError error) {
  atomicCAS(device_error, static_cast<std::uint32_t>(DeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__device__ bool sequence_is_valid(const std::uint32_t* device_error) {
  return atomicAdd(const_cast<std::uint32_t*>(device_error), 0u) ==
         static_cast<std::uint32_t>(DeviceError::kSuccess);
}

__device__ bool origin(const Gfn2CudaPeriodicTranslation& translation) {
  return translation.index[0] == 0 && translation.index[1] == 0 && translation.index[2] == 0;
}

__device__ bool finite_vector(const double* values, int count) {
  for (int index = 0; index < count; ++index) {
    if (!isfinite(values[index])) return false;
  }
  return true;
}

__device__ bool prepare_system(const Gfn2NativePeriodicD4DeviceBatch& batch,
                               const Gfn2NativePeriodicD4DeviceWorkspace& workspace,
                               std::int64_t system, std::uint32_t* system_errors,
                               std::uint32_t* device_error, bool require_coordination,
                               bool require_charges) {
  if (!sequence_is_valid(device_error)) return false;
  if (batch.active_mask != nullptr) {
    const std::uint8_t active = batch.active_mask[system];
    if (active > 1u) {
      record_system_error(system_errors, system, DeviceError::kInvalidActivity);
      return false;
    }
    if (active == 0u) return false;
  }
  if (atomicAdd(system_errors + system, 0u) !=
      static_cast<std::uint32_t>(DeviceError::kSuccess)) {
    return false;
  }

  const std::int64_t atom_begin = batch.topology.atom_offsets[system];
  const std::int64_t atom_end = batch.topology.atom_offsets[system + 1];
  const std::int64_t translation_begin = batch.topology.translation_offsets[system];
  const std::int64_t translation_end = batch.topology.translation_offsets[system + 1];
  if (atom_begin < 0 || atom_begin > atom_end || atom_end > batch.topology.total_atoms ||
      translation_begin < 0 || translation_begin >= translation_end ||
      translation_end > batch.topology.total_translations ||
      batch.topology.periodic_axes[system] != XTBLOOM_PERIODIC_AXES_XYZ ||
      batch.image_cutoff < kTwoBodyCutoff || !isfinite(batch.image_cutoff)) {
    record_system_error(system_errors, system, DeviceError::kInvalidOffsets);
    record_device_error(device_error, DeviceError::kInvalidTopology);
    return false;
  }
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    const std::int32_t atomic_number = batch.atomic_numbers[atom];
    if (atomic_number <= 0 || atomic_number > batch.parameters.element_count) {
      record_system_error(system_errors, system, DeviceError::kInvalidAtomicNumber);
      return false;
    }
    const Gfn2D4DeviceElementData element = batch.parameters.elements[atomic_number - 1];
    if (element.reference_count == 0 || element.reference_count > kGfn2D4MaximumReferences ||
        static_cast<std::int64_t>(element.reference_offset) + element.reference_count >
            batch.parameters.reference_count ||
        !(element.covalent_radius > 0.0) || !isfinite(element.covalent_radius) ||
        !isfinite(element.electronegativity) || !(element.r4r2 > 0.0) ||
        !isfinite(element.r4r2) || !(element.effective_charge > 0.0) ||
        !isfinite(element.effective_charge) || !(element.hardness > 0.0) ||
        !isfinite(element.hardness)) {
      record_device_error(device_error, DeviceError::kInvalidParameterData);
      record_system_error(system_errors, system, DeviceError::kInvalidParameterData);
      return false;
    }
    const double* position = batch.positions + atom * 3;
    double* wrapped = workspace.wrapped_positions + atom * 3;
    if (!finite_vector(position, 3) ||
        !wrap_periodic_position(batch.topology, system, position, wrapped)) {
      record_system_error(system_errors, system, DeviceError::kNonfinitePosition);
      return false;
    }
    if (require_coordination &&
        (!(batch.coordination_numbers[atom] >= 0.0) ||
         !isfinite(batch.coordination_numbers[atom]))) {
      record_system_error(system_errors, system, DeviceError::kNonfiniteCoordination);
      return false;
    }
    if (require_charges && !isfinite(batch.atomic_charges[atom])) {
      record_system_error(system_errors, system, DeviceError::kNonfiniteCharge);
      return false;
    }
  }
  return true;
}

struct PairParameters {
  double coordination_radius = 0.0;
  double electronegativity_factor = 0.0;
  double rrij = 0.0;
  double damping_radius = 0.0;
};

__device__ bool pair_parameters(const Gfn2NativePeriodicD4DeviceBatch& batch, std::int64_t first,
                                std::int64_t second, PairParameters& output) {
  const Gfn2D4DeviceElementData first_element =
      batch.parameters.elements[batch.atomic_numbers[first] - 1];
  const Gfn2D4DeviceElementData second_element =
      batch.parameters.elements[batch.atomic_numbers[second] - 1];
  output.coordination_radius = first_element.covalent_radius + second_element.covalent_radius;
  const double en_delta = fabs(first_element.electronegativity - second_element.electronegativity);
  output.electronegativity_factor =
      kEnK4 * exp(-((en_delta + kEnK5) * (en_delta + kEnK5)) / kEnK6);
  output.rrij = 3.0 * first_element.r4r2 * second_element.r4r2;
  output.damping_radius = kDispersionA1 * sqrt(output.rrij) + kDispersionA2;
  return output.coordination_radius > 0.0 && isfinite(output.coordination_radius) &&
         isfinite(output.electronegativity_factor) && output.rrij > 0.0 &&
         isfinite(output.rrij) && output.damping_radius > 0.0 &&
         isfinite(output.damping_radius);
}

__device__ bool image_geometry(const double* center, const double* image,
                               const Gfn2CudaPeriodicTranslation& translation, double cutoff,
                               double vector[3], double& distance_squared) {
  for (int component = 0; component < 3; ++component) {
    vector[component] = image[component] + translation.cartesian[component] - center[component];
    if (!isfinite(vector[component]) || fabs(vector[component]) > cutoff) return false;
  }
  distance_squared = fma(vector[0], vector[0],
                         fma(vector[1], vector[1], vector[2] * vector[2]));
  return isfinite(distance_squared) && distance_squared <= cutoff * cutoff;
}

__device__ double coordination_count(const PairParameters& parameters, double distance_squared) {
  const double distance = sqrt(distance_squared);
  const double exponent = kCoordinationSteepness *
                          (distance - parameters.coordination_radius) /
                          parameters.coordination_radius;
  return 0.5 * parameters.electronegativity_factor * (1.0 + erf(-exponent));
}

struct CutoffSwitch {
  double value = 1.0;
  double distance_derivative = 0.0;
};

__device__ CutoffSwitch cutoff_switch(double distance, double cutoff) {
  const double inner = cutoff - kCutoffSwitchWidth;
  if (distance <= inner) return {1.0, 0.0};
  if (distance >= cutoff) return {0.0, 0.0};
  const double x = (cutoff - distance) / kCutoffSwitchWidth;
  const double x_squared = x * x;
  const double one_minus_x = 1.0 - x;
  return {x_squared * x * (10.0 + x * (-15.0 + 6.0 * x)),
          -30.0 * x_squared * one_minus_x * one_minus_x / kCutoffSwitchWidth};
}

__device__ void d4_damping(const PairParameters& parameters, double distance_squared,
                           double cutoff, double& damping, double& derivative) {
  const double r2_squared = distance_squared * distance_squared;
  const double r2_cubed = r2_squared * distance_squared;
  const double radius_squared = parameters.damping_radius * parameters.damping_radius;
  const double radius_fourth = radius_squared * radius_squared;
  const double radius_sixth = radius_fourth * radius_squared;
  const double t6 = 1.0 / (r2_cubed + radius_sixth);
  const double t8 = 1.0 / (r2_squared * r2_squared + radius_fourth * radius_fourth);
  const double base = kDispersionS6 * t6 + kDispersionS8 * parameters.rrij * t8;
  const double base_derivative = kDispersionS6 * (-6.0 * r2_squared * t6 * t6) +
                                 kDispersionS8 * parameters.rrij *
                                     (-8.0 * r2_cubed * t8 * t8);
  const double distance = sqrt(distance_squared);
  const CutoffSwitch outer = cutoff_switch(distance, cutoff);
  damping = outer.value * base;
  derivative = outer.value * base_derivative + outer.distance_derivative * base / distance;
}

__device__ double charge_scale(double a, double c, double qref, double qmod) {
  if (qmod < 0.0) return exp(a);
  return exp(a * (1.0 - exp(c * (1.0 - qref / qmod))));
}

__device__ double charge_scale_derivative(double a, double c, double qref, double qmod) {
  if (qmod < 0.0) return 0.0;
  const double inner = exp(c * (1.0 - qref / qmod));
  return -a * c * inner * charge_scale(a, c, qref, qmod) * qref / (qmod * qmod);
}

__device__ bool prepare_weights(const Gfn2NativePeriodicD4DeviceBatch& batch,
                                const Gfn2NativePeriodicD4DeviceWorkspace& workspace,
                                std::int64_t system, bool use_charges,
                                std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t begin = batch.topology.atom_offsets[system];
  const std::int64_t end = batch.topology.atom_offsets[system + 1];
  for (std::int64_t atom = begin; atom < end; ++atom) {
    const auto element = batch.parameters.elements[batch.atomic_numbers[atom] - 1];
    const double coordination = batch.coordination_numbers[atom];
    const double charge = use_charges ? batch.atomic_charges[atom] : 0.0;
    const std::int64_t offset = atom * kGfn2D4MaximumReferences;
    for (std::int64_t local = 0; local < kGfn2D4MaximumReferences; ++local) {
      workspace.weights[offset + local] = 0.0;
      workspace.weight_cn_derivatives[offset + local] = 0.0;
      workspace.weight_charge_derivatives[offset + local] = 0.0;
    }
    double normalization = 0.0;
    double normalization_derivative = 0.0;
    double maximum_reference_cn = -1.7976931348623157e308;
    for (std::int64_t local = 0; local < element.reference_count; ++local) {
      const auto reference = batch.parameters.references[element.reference_offset + local];
      if (reference.gaussian_count == 0 || !isfinite(reference.coordination_number) ||
          !isfinite(reference.charge)) {
        record_device_error(device_error, DeviceError::kInvalidParameterData);
        record_system_error(system_errors, system, DeviceError::kInvalidParameterData);
        return false;
      }
      maximum_reference_cn = fmax(maximum_reference_cn, reference.coordination_number);
      for (std::int64_t gaussian = 1; gaussian <= reference.gaussian_count; ++gaussian) {
        const double factor = static_cast<double>(gaussian) * kReferenceWeightFactor;
        const double delta = coordination - reference.coordination_number;
        const double value = exp(-factor * delta * delta);
        normalization += value;
        normalization_derivative +=
            2.0 * factor * (reference.coordination_number - coordination) * value;
      }
    }
    const double inverse_normalization =
        fabs(normalization) > kMinimumWeightNorm ? 1.0 / normalization : 0.0;
    const double qmod = charge + element.effective_charge;
    const double charge_steepness = element.hardness * kChargeScalingSteepness;
    for (std::int64_t local = 0; local < element.reference_count; ++local) {
      const auto reference = batch.parameters.references[element.reference_offset + local];
      double numerator = 0.0;
      double numerator_derivative = 0.0;
      for (std::int64_t gaussian = 1; gaussian <= reference.gaussian_count; ++gaussian) {
        const double factor = static_cast<double>(gaussian) * kReferenceWeightFactor;
        const double delta = coordination - reference.coordination_number;
        const double value = exp(-factor * delta * delta);
        numerator += value;
        numerator_derivative +=
            2.0 * factor * (reference.coordination_number - coordination) * value;
      }
      double cn_weight = numerator * inverse_normalization;
      if (!isfinite(cn_weight) || inverse_normalization == 0.0) {
        cn_weight = fabs(maximum_reference_cn - reference.coordination_number) < 1.0e-12
                        ? 1.0
                        : 0.0;
      }
      double cn_derivative = inverse_normalization *
                             (numerator_derivative -
                              numerator * normalization_derivative * inverse_normalization);
      if (!isfinite(cn_derivative) || inverse_normalization == 0.0) cn_derivative = 0.0;
      const double qref = reference.charge + element.effective_charge;
      const double scaling =
          charge_scale(kChargeScalingHeight, charge_steepness, qref, qmod);
      const double charge_derivative =
          use_charges ? cn_weight * charge_scale_derivative(kChargeScalingHeight,
                                                              charge_steepness, qref, qmod)
                      : 0.0;
      const double weight = cn_weight * scaling;
      const double cn_scaled_derivative = cn_derivative * scaling;
      if (!isfinite(weight) || !isfinite(cn_scaled_derivative) ||
          !isfinite(charge_derivative)) {
        record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
        return false;
      }
      workspace.weights[offset + local] = weight;
      workspace.weight_cn_derivatives[offset + local] = cn_scaled_derivative;
      workspace.weight_charge_derivatives[offset + local] = charge_derivative;
    }
  }
  return true;
}

struct PairCoefficient {
  double c6 = 0.0;
  double first_cn = 0.0;
  double second_cn = 0.0;
  double first_charge = 0.0;
  double second_charge = 0.0;
};

__device__ PairCoefficient pair_coefficient(const Gfn2NativePeriodicD4DeviceBatch& batch,
                                            const Gfn2NativePeriodicD4DeviceWorkspace& workspace,
                                            std::int64_t first, std::int64_t second,
                                            bool derivatives) {
  const auto first_element = batch.parameters.elements[batch.atomic_numbers[first] - 1];
  const auto second_element = batch.parameters.elements[batch.atomic_numbers[second] - 1];
  const std::int64_t first_offset = first * kGfn2D4MaximumReferences;
  const std::int64_t second_offset = second * kGfn2D4MaximumReferences;
  PairCoefficient result{};
  for (std::int64_t first_ref = 0; first_ref < first_element.reference_count; ++first_ref) {
    const std::int64_t global_first = first_element.reference_offset + first_ref;
    const double first_value = workspace.weights[first_offset + first_ref];
    for (std::int64_t second_ref = 0; second_ref < second_element.reference_count; ++second_ref) {
      const std::int64_t global_second = second_element.reference_offset + second_ref;
      const double reference_c6 = batch.parameters.reference_c6[
          global_first * batch.parameters.reference_count + global_second];
      const double second_value = workspace.weights[second_offset + second_ref];
      result.c6 += first_value * second_value * reference_c6;
      if (derivatives) {
        result.first_cn += workspace.weight_cn_derivatives[first_offset + first_ref] *
                           second_value * reference_c6;
        result.second_cn += first_value *
                            workspace.weight_cn_derivatives[second_offset + second_ref] *
                            reference_c6;
        result.first_charge += workspace.weight_charge_derivatives[first_offset + first_ref] *
                               second_value * reference_c6;
        result.second_charge += first_value *
                                workspace.weight_charge_derivatives[second_offset + second_ref] *
                                reference_c6;
      }
    }
  }
  return result;
}

__device__ bool finite_pair_coefficient(const PairCoefficient& coefficient, bool derivatives) {
  return isfinite(coefficient.c6) && (!derivatives ||
                                      (isfinite(coefficient.first_cn) &&
                                       isfinite(coefficient.second_cn) &&
                                       isfinite(coefficient.first_charge) &&
                                       isfinite(coefficient.second_charge)));
}

__device__ double triple_scale(std::int64_t first, std::int64_t second, std::int64_t third) {
  if (first == second) return first == third ? 1.0 / 6.0 : 0.5;
  return first != third && second != third ? 1.0 : 0.5;
}

__device__ bool pair_image(const Gfn2NativePeriodicD4DeviceBatch& batch,
                           const Gfn2NativePeriodicD4DeviceWorkspace& workspace,
                           std::int64_t first, std::int64_t second,
                           const Gfn2CudaPeriodicTranslation& translation, double cutoff,
                           double vector[3], double& distance_squared) {
  const double* center = workspace.wrapped_positions + first * 3;
  const double* image = workspace.wrapped_positions + second * 3;
  return image_geometry(center, image, translation, cutoff, vector, distance_squared);
}

__device__ bool finite_array(const double* values, std::int64_t elements) {
  for (std::int64_t index = 0; index < elements; ++index) {
    if (!isfinite(values[index])) return false;
  }
  return true;
}

__device__ bool finish_system(std::int64_t system, std::uint32_t* system_errors,
                              std::uint32_t* device_error) {
  return sequence_is_valid(device_error) &&
         atomicAdd(system_errors + system, 0u) == static_cast<std::uint32_t>(DeviceError::kSuccess);
}


__global__ void native_d4_coordination_kernel(
    Gfn2NativePeriodicD4DeviceBatch batch, Gfn2NativePeriodicD4DeviceWorkspace workspace,
    double* coordination_numbers, std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (threadIdx.x != 0 || system >= batch.topology.batch_size ||
      !prepare_system(batch, workspace, system, system_errors, device_error, false, false)) {
    return;
  }
  const std::int64_t atom_begin = batch.topology.atom_offsets[system];
  const std::int64_t atom_end = batch.topology.atom_offsets[system + 1];
  const std::int64_t translation_begin = batch.topology.translation_offsets[system];
  const std::int64_t translation_end = batch.topology.translation_offsets[system + 1];
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    workspace.coordination[atom] = 0.0;
  }
  for (std::int64_t first = atom_begin; first < atom_end; ++first) {
    const double* center = workspace.wrapped_positions + first * 3;
    for (std::int64_t second = atom_begin; second <= first; ++second) {
      PairParameters parameters{};
      if (!pair_parameters(batch, first, second, parameters)) {
        record_system_error(system_errors, system, DeviceError::kInvalidParameterData);
        return;
      }
      const double* image = workspace.wrapped_positions + second * 3;
      for (std::int64_t translation = translation_begin; translation < translation_end;
           ++translation) {
        const auto& value = batch.topology.translations[translation];
        if (first == second && origin(value)) continue;
        double vector[3]{};
        double distance_squared = 0.0;
        if (!image_geometry(center, image, value, kCoordinationCutoff, vector, distance_squared)) {
          continue;
        }
        if (distance_squared < kMinimumDistanceSquared) {
          record_system_error(system_errors, system, DeviceError::kCoincidentImage);
          return;
        }
        const double count = coordination_count(parameters, distance_squared);
        if (!isfinite(count) || count < 0.0) {
          record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
          return;
        }
        workspace.coordination[first] += count;
        if (first != second) workspace.coordination[second] += count;
      }
    }
  }
  if (!finite_array(workspace.coordination + atom_begin, atom_end - atom_begin)) {
    record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
    return;
  }
  if (!finish_system(system, system_errors, device_error)) return;
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    coordination_numbers[atom] = workspace.coordination[atom];
  }
}

__global__ void native_d4_two_body_kernel(
    Gfn2NativePeriodicD4DeviceBatch batch, Gfn2NativePeriodicD4DeviceWorkspace workspace,
    double* per_atom_energies, double* atomic_potentials, std::uint32_t* system_errors,
    std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (threadIdx.x != 0 || system >= batch.topology.batch_size ||
      !prepare_system(batch, workspace, system, system_errors, device_error, true, true)) {
    return;
  }
  if (!prepare_weights(batch, workspace, system, true, system_errors, device_error)) return;
  const std::int64_t atom_begin = batch.topology.atom_offsets[system];
  const std::int64_t atom_end = batch.topology.atom_offsets[system + 1];
  const std::int64_t translation_begin = batch.topology.translation_offsets[system];
  const std::int64_t translation_end = batch.topology.translation_offsets[system + 1];
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    workspace.atom_energy[atom] = 0.0;
    workspace.atom_potential[atom] = 0.0;
  }
  for (std::int64_t first = atom_begin; first < atom_end; ++first) {
    const double* center = workspace.wrapped_positions + first * 3;
    for (std::int64_t second = atom_begin; second <= first; ++second) {
      PairParameters parameters{};
      if (!pair_parameters(batch, first, second, parameters)) {
        record_system_error(system_errors, system, DeviceError::kInvalidParameterData);
        return;
      }
      const double* image = workspace.wrapped_positions + second * 3;
      for (std::int64_t translation = translation_begin; translation < translation_end;
           ++translation) {
        const auto& value = batch.topology.translations[translation];
        if (first == second && origin(value)) continue;
        double vector[3]{};
        double distance_squared = 0.0;
        if (!image_geometry(center, image, value, kTwoBodyCutoff, vector, distance_squared)) {
          continue;
        }
        if (distance_squared < kMinimumDistanceSquared) {
          record_system_error(system_errors, system, DeviceError::kCoincidentImage);
          return;
        }
        double damping = 0.0;
        double derivative = 0.0;
        d4_damping(parameters, distance_squared, kTwoBodyCutoff, damping, derivative);
        const PairCoefficient coefficient =
            pair_coefficient(batch, workspace, first, second, true);
        if (!finite_pair_coefficient(coefficient, true) || !isfinite(damping) ||
            !isfinite(derivative) || damping < 0.0) {
          record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
          return;
        }
        const double pair_energy = -coefficient.c6 * damping;
        workspace.atom_energy[first] += 0.5 * pair_energy;
        if (first != second) workspace.atom_energy[second] += 0.5 * pair_energy;
        const double accounting = first == second ? 0.5 : 1.0;
        workspace.atom_potential[first] -= accounting * coefficient.first_charge * damping;
        if (first != second) {
          workspace.atom_potential[second] -= coefficient.second_charge * damping;
        } else {
          workspace.atom_potential[first] -= accounting * coefficient.second_charge * damping;
        }
      }
    }
  }
  if (!finite_array(workspace.atom_energy + atom_begin, atom_end - atom_begin) ||
      !finite_array(workspace.atom_potential + atom_begin, atom_end - atom_begin)) {
    record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
    return;
  }
  if (!finish_system(system, system_errors, device_error)) return;
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    per_atom_energies[atom] = workspace.atom_energy[atom];
    atomic_potentials[atom] = workspace.atom_potential[atom];
  }
}

__device__ bool atm_pair_geometry(const Gfn2NativePeriodicD4DeviceBatch& batch,
                                  const Gfn2NativePeriodicD4DeviceWorkspace& workspace,
                                  std::int64_t first, std::int64_t second,
                                  const Gfn2CudaPeriodicTranslation& translation, double cutoff,
                                  double vector[3], double& distance_squared) {
  return pair_image(batch, workspace, first, second, translation, cutoff, vector, distance_squared);
}

/*
 * Differentiate one ATM radial leg with respect to its image displacement.
 * Keep this as a regular device helper instead of an extended CUDA lambda so
 * the native primitive builds with the repository's strict C++17/NVCC setup.
 */
__device__ void atm_distance_gradient(double target, double other_first, double other_second,
                                      double r5_product, double switch_product, double c9,
                                      double angle, double damping, double damping_derivative,
                                      double switch_derivative, double other_switches,
                                      const double vector[3], double output[3]) {
  const double angle_derivative =
      -0.375 *
      (target * target * target + target * target * (other_first + other_second) +
       target * (3.0 * other_first * other_first + 2.0 * other_first * other_second +
                 3.0 * other_second * other_second) -
       5.0 * (other_first - other_second) * (other_first - other_second) *
           (other_first + other_second)) /
      r5_product;
  const double energy_without_switch = angle * damping * c9;
  const double radial_scale =
      switch_product * c9 * (-angle_derivative * damping + angle * damping_derivative) / target -
      energy_without_switch * switch_derivative * other_switches / sqrt(target);
  for (int axis = 0; axis < 3; ++axis) output[axis] = radial_scale * vector[axis];
}

__device__ bool atm_term_values(const Gfn2NativePeriodicD4DeviceBatch& batch,
                                const Gfn2NativePeriodicD4DeviceWorkspace& workspace,
                                std::int64_t first, std::int64_t second, std::int64_t third,
                                const double vij[3], const double vik[3], const double vjk[3],
                                double r2ij, double r2ik, double r2jk, double scale,
                                bool derivatives, double& energy, double dgij[3],
                                double dgik[3], double dgjk[3], double& first_adjoint,
                                double& second_adjoint, double& third_adjoint) {
  PairParameters pij{};
  PairParameters pik{};
  PairParameters pjk{};
  if (!pair_parameters(batch, first, second, pij) || !pair_parameters(batch, first, third, pik) ||
      !pair_parameters(batch, second, third, pjk)) {
    return false;
  }
  const PairCoefficient cij = pair_coefficient(batch, workspace, first, second, derivatives);
  const PairCoefficient cik = pair_coefficient(batch, workspace, first, third, derivatives);
  const PairCoefficient cjk = pair_coefficient(batch, workspace, second, third, derivatives);
  if (!finite_pair_coefficient(cij, derivatives) || !finite_pair_coefficient(cik, derivatives) ||
      !finite_pair_coefficient(cjk, derivatives) || !(cij.c6 > 0.0) || !(cik.c6 > 0.0) ||
      !(cjk.c6 > 0.0)) {
    return false;
  }
  const double r2_product = r2ij * r2ik * r2jk;
  const double r1_product = sqrt(r2_product);
  const double r3_product = r2_product * r1_product;
  const double r5_product = r3_product * r2_product;
  const double ratio = (pij.damping_radius * pik.damping_radius * pjk.damping_radius) /
                       r1_product;
  const double ratio_power = pow(ratio, kAtmExponent / 3.0);
  const double damping = 1.0 / (1.0 + 6.0 * ratio_power);
  const double angle =
      0.375 * (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) *
          (-r2ij + r2jk + r2ik) / r5_product +
      1.0 / r3_product;
  const double c9 = -kDispersionS9 * sqrt(cij.c6 * cik.c6 * cjk.c6);
  const CutoffSwitch switch_ij = cutoff_switch(sqrt(r2ij), kAtmCutoff);
  const CutoffSwitch switch_ik = cutoff_switch(sqrt(r2ik), kAtmCutoff);
  const CutoffSwitch switch_jk = cutoff_switch(sqrt(r2jk), kAtmCutoff);
  const double switch_product = switch_ij.value * switch_ik.value * switch_jk.value;
  const double rr = angle * damping;
  energy = rr * c9 * scale * switch_product;
  if (!isfinite(energy)) return false;
  first_adjoint = second_adjoint = third_adjoint = 0.0;
  if (!derivatives) return true;

  const double damping_derivative = -2.0 * kAtmExponent * ratio_power * damping * damping;
  atm_distance_gradient(r2ij, r2jk, r2ik, r5_product, switch_product, c9, angle, damping,
                        damping_derivative, switch_ij.distance_derivative,
                        switch_ik.value * switch_jk.value, vij, dgij);
  atm_distance_gradient(r2ik, r2jk, r2ij, r5_product, switch_product, c9, angle, damping,
                        damping_derivative, switch_ik.distance_derivative,
                        switch_ij.value * switch_jk.value, vik, dgik);
  atm_distance_gradient(r2jk, r2ik, r2ij, r5_product, switch_product, c9, angle, damping,
                        damping_derivative, switch_jk.distance_derivative,
                        switch_ij.value * switch_ik.value, vjk, dgjk);
  /* The CN VJP follows the same fully switched ATM energy that was just
   * evaluated.  Omitting switch_product here would leave a discontinuous
   * coordinate derivative in the final 0.05-bohr cutoff shell. */
  const double switched_energy = rr * c9 * scale * switch_product;
  first_adjoint = -0.5 * switched_energy *
                  (cij.first_cn / cij.c6 + cik.first_cn / cik.c6);
  second_adjoint = -0.5 * switched_energy *
                   (cij.second_cn / cij.c6 + cjk.first_cn / cjk.c6);
  third_adjoint = -0.5 * switched_energy *
                  (cik.second_cn / cik.c6 + cjk.second_cn / cjk.c6);
  return isfinite(dgij[0]) && isfinite(dgij[1]) && isfinite(dgij[2]) &&
         isfinite(dgik[0]) && isfinite(dgik[1]) && isfinite(dgik[2]) &&
         isfinite(dgjk[0]) && isfinite(dgjk[1]) && isfinite(dgjk[2]) &&
         isfinite(first_adjoint) && isfinite(second_adjoint) && isfinite(third_adjoint);
}

__global__ void native_d4_atm_kernel(
    Gfn2NativePeriodicD4DeviceBatch batch, Gfn2NativePeriodicD4DeviceWorkspace workspace,
    double* per_atom_energies, std::uint32_t* system_errors, std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (threadIdx.x != 0 || system >= batch.topology.batch_size ||
      !prepare_system(batch, workspace, system, system_errors, device_error, true, false)) {
    return;
  }
  if (!prepare_weights(batch, workspace, system, false, system_errors, device_error)) return;
  const std::int64_t atom_begin = batch.topology.atom_offsets[system];
  const std::int64_t atom_end = batch.topology.atom_offsets[system + 1];
  const std::int64_t translation_begin = batch.topology.translation_offsets[system];
  const std::int64_t translation_end = batch.topology.translation_offsets[system + 1];
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    workspace.atom_energy[atom] = 0.0;
  }
  for (std::int64_t first = atom_begin; first < atom_end; ++first) {
    for (std::int64_t second = atom_begin; second <= first; ++second) {
      for (std::int64_t third = atom_begin; third <= second; ++third) {
        PairParameters unused{};
        if (!pair_parameters(batch, first, second, unused) ||
            !pair_parameters(batch, first, third, unused) ||
            !pair_parameters(batch, second, third, unused)) {
          record_system_error(system_errors, system, DeviceError::kInvalidParameterData);
          return;
        }
        const PairCoefficient cij = pair_coefficient(batch, workspace, first, second, false);
        const PairCoefficient cik = pair_coefficient(batch, workspace, first, third, false);
        const PairCoefficient cjk = pair_coefficient(batch, workspace, second, third, false);
        if (!finite_pair_coefficient(cij, false) || !finite_pair_coefficient(cik, false) ||
            !finite_pair_coefficient(cjk, false) || !(cij.c6 > 0.0) || !(cik.c6 > 0.0) ||
            !(cjk.c6 > 0.0)) {
          record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
          return;
        }
        const double scale = triple_scale(first, second, third);
        for (std::int64_t first_translation = translation_begin;
             first_translation < translation_end; ++first_translation) {
          const auto& translation_ij = batch.topology.translations[first_translation];
          if (first == second && origin(translation_ij)) continue;
          double vij[3]{};
          double r2ij = 0.0;
          if (!atm_pair_geometry(batch, workspace, first, second, translation_ij, kAtmCutoff,
                                 vij, r2ij)) {
            continue;
          }
          if (r2ij < kMinimumDistanceSquared) {
            record_system_error(system_errors, system, DeviceError::kCoincidentImage);
            return;
          }
          for (std::int64_t second_translation = translation_begin;
               second_translation < translation_end; ++second_translation) {
            const auto& translation_ik = batch.topology.translations[second_translation];
            if (first == third && origin(translation_ik)) continue;
            double vik[3]{};
            double r2ik = 0.0;
            if (!atm_pair_geometry(batch, workspace, first, third, translation_ik, kAtmCutoff,
                                   vik, r2ik)) {
              continue;
            }
            if (r2ik < kMinimumDistanceSquared) {
              record_system_error(system_errors, system, DeviceError::kCoincidentImage);
              return;
            }
            double vjk[3]{};
            double r2jk = 0.0;
            for (int axis = 0; axis < 3; ++axis) vjk[axis] = vik[axis] - vij[axis];
            r2jk = fma(vjk[0], vjk[0], fma(vjk[1], vjk[1], vjk[2] * vjk[2]));
            if (!isfinite(r2jk) || r2jk > kAtmCutoff * kAtmCutoff) continue;
            if (r2jk < kMinimumDistanceSquared) {
              if (second == third &&
                  translation_ij.index[0] == translation_ik.index[0] &&
                  translation_ij.index[1] == translation_ik.index[1] &&
                  translation_ij.index[2] == translation_ik.index[2]) {
                continue;
              }
              record_system_error(system_errors, system, DeviceError::kCoincidentImage);
              return;
            }
            double energy = 0.0;
            double dgij[3]{}, dgik[3]{}, dgjk[3]{};
            double first_adjoint = 0.0, second_adjoint = 0.0, third_adjoint = 0.0;
            if (!atm_term_values(batch, workspace, first, second, third, vij, vik, vjk, r2ij,
                                 r2ik, r2jk, scale, false, energy, dgij, dgik, dgjk,
                                 first_adjoint, second_adjoint, third_adjoint)) {
              record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
              return;
            }
            const double per_atom = energy / 3.0;
            workspace.atom_energy[first] -= per_atom;
            workspace.atom_energy[second] -= per_atom;
            workspace.atom_energy[third] -= per_atom;
          }
        }
      }
    }
  }
  if (!finite_array(workspace.atom_energy + atom_begin, atom_end - atom_begin)) {
    record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
    return;
  }
  if (!finish_system(system, system_errors, device_error)) return;
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    per_atom_energies[atom] = workspace.atom_energy[atom];
  }
}

/* Add all periodic D4 coordinate/cell derivatives in one deterministic pass.
 * The scratch tuple is complete before either caller-owned accumulator is
 * touched, preserving peer-local publication when a later image or ATM term
 * turns out to be invalid. */
__global__ void native_d4_gradient_kernel(
    Gfn2NativePeriodicD4DeviceBatch batch, Gfn2NativePeriodicD4DeviceWorkspace workspace,
    double* gradients, double* strain_derivatives, std::uint32_t* system_errors,
    std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (threadIdx.x != 0 || system >= batch.topology.batch_size ||
      !prepare_system(batch, workspace, system, system_errors, device_error, true, true)) {
    return;
  }
  if (!prepare_weights(batch, workspace, system, true, system_errors, device_error)) return;

  const std::int64_t atom_begin = batch.topology.atom_offsets[system];
  const std::int64_t atom_end = batch.topology.atom_offsets[system + 1];
  const std::int64_t translation_begin = batch.topology.translation_offsets[system];
  const std::int64_t translation_end = batch.topology.translation_offsets[system + 1];
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    for (int axis = 0; axis < 3; ++axis) workspace.gradient[atom * 3 + axis] = 0.0;
    workspace.coordination_adjoint[atom] = 0.0;
  }
  double* const system_strain = workspace.strain + system * 9;
  for (int component = 0; component < 9; ++component) system_strain[component] = 0.0;

  /* Charge-dependent two-body term, including its charge-to-CN adjoint. */
  for (std::int64_t first = atom_begin; first < atom_end; ++first) {
    const double* center = workspace.wrapped_positions + first * 3;
    for (std::int64_t second = atom_begin; second <= first; ++second) {
      PairParameters parameters{};
      if (!pair_parameters(batch, first, second, parameters)) {
        record_system_error(system_errors, system, DeviceError::kInvalidParameterData);
        return;
      }
      const double* image = workspace.wrapped_positions + second * 3;
      for (std::int64_t translation = translation_begin; translation < translation_end;
           ++translation) {
        const auto& value = batch.topology.translations[translation];
        if (first == second && origin(value)) continue;
        double vector[3]{};
        double distance_squared = 0.0;
        if (!image_geometry(center, image, value, kTwoBodyCutoff, vector, distance_squared)) {
          continue;
        }
        if (distance_squared < kMinimumDistanceSquared) {
          record_system_error(system_errors, system, DeviceError::kCoincidentImage);
          return;
        }
        double damping = 0.0;
        double derivative = 0.0;
        d4_damping(parameters, distance_squared, kTwoBodyCutoff, damping, derivative);
        const PairCoefficient coefficient = pair_coefficient(batch, workspace, first, second, true);
        if (!finite_pair_coefficient(coefficient, true) || !isfinite(damping) ||
            !isfinite(derivative) || damping < 0.0) {
          record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
          return;
        }
        const double radial_scale = -coefficient.c6 * derivative;
        const double pair_vector[3] = {-vector[0], -vector[1], -vector[2]};
        double first_gradient[3]{};
        for (int axis = 0; axis < 3; ++axis) {
          first_gradient[axis] = radial_scale * pair_vector[axis];
          if (first != second) {
            workspace.gradient[first * 3 + axis] += first_gradient[axis];
            workspace.gradient[second * 3 + axis] -= first_gradient[axis];
          }
        }
        const double accounting = first == second ? 0.5 : 1.0;
        for (int row = 0; row < 3; ++row) {
          for (int column = 0; column < 3; ++column) {
            system_strain[row * 3 + column] +=
                accounting * first_gradient[row] * pair_vector[column];
          }
        }
        if (first == second) {
          workspace.coordination_adjoint[first] -=
              accounting * (coefficient.first_cn + coefficient.second_cn) * damping;
        } else {
          workspace.coordination_adjoint[first] -= coefficient.first_cn * damping;
          workspace.coordination_adjoint[second] -= coefficient.second_cn * damping;
        }
      }
    }
  }

  /* ATM term.  This is intentionally the same lower-triangular label and
   * translation order as the CPU periodic evaluator. */
  for (std::int64_t first = atom_begin; first < atom_end; ++first) {
    for (std::int64_t second = atom_begin; second <= first; ++second) {
      for (std::int64_t third = atom_begin; third <= second; ++third) {
        const double scale = triple_scale(first, second, third);
        for (std::int64_t first_translation = translation_begin;
             first_translation < translation_end; ++first_translation) {
          const auto& translation_ij = batch.topology.translations[first_translation];
          if (first == second && origin(translation_ij)) continue;
          double vij[3]{};
          double r2ij = 0.0;
          if (!atm_pair_geometry(batch, workspace, first, second, translation_ij, kAtmCutoff,
                                 vij, r2ij)) {
            continue;
          }
          if (r2ij < kMinimumDistanceSquared) {
            record_system_error(system_errors, system, DeviceError::kCoincidentImage);
            return;
          }
          for (std::int64_t second_translation = translation_begin;
               second_translation < translation_end; ++second_translation) {
            const auto& translation_ik = batch.topology.translations[second_translation];
            if (first == third && origin(translation_ik)) continue;
            double vik[3]{};
            double r2ik = 0.0;
            if (!atm_pair_geometry(batch, workspace, first, third, translation_ik, kAtmCutoff,
                                   vik, r2ik)) {
              continue;
            }
            if (r2ik < kMinimumDistanceSquared) {
              record_system_error(system_errors, system, DeviceError::kCoincidentImage);
              return;
            }
            double vjk[3]{};
            for (int axis = 0; axis < 3; ++axis) vjk[axis] = vik[axis] - vij[axis];
            const double r2jk = fma(vjk[0], vjk[0], fma(vjk[1], vjk[1], vjk[2] * vjk[2]));
            if (!isfinite(r2jk) || r2jk > kAtmCutoff * kAtmCutoff) continue;
            if (r2jk < kMinimumDistanceSquared) {
              if (second == third && translation_ij.index[0] == translation_ik.index[0] &&
                  translation_ij.index[1] == translation_ik.index[1] &&
                  translation_ij.index[2] == translation_ik.index[2]) {
                continue;
              }
              record_system_error(system_errors, system, DeviceError::kCoincidentImage);
              return;
            }
            double energy = 0.0;
            double dgij[3]{}, dgik[3]{}, dgjk[3]{};
            double first_adjoint = 0.0, second_adjoint = 0.0, third_adjoint = 0.0;
            if (!atm_term_values(batch, workspace, first, second, third, vij, vik, vjk, r2ij,
                                 r2ik, r2jk, scale, true, energy, dgij, dgik, dgjk,
                                 first_adjoint, second_adjoint, third_adjoint)) {
              record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
              return;
            }
            for (int axis = 0; axis < 3; ++axis) {
              workspace.gradient[first * 3 + axis] -= scale * (dgij[axis] + dgik[axis]);
              workspace.gradient[second * 3 + axis] += scale * (dgij[axis] - dgjk[axis]);
              workspace.gradient[third * 3 + axis] += scale * (dgik[axis] + dgjk[axis]);
            }
            for (int row = 0; row < 3; ++row) {
              for (int column = 0; column < 3; ++column) {
                system_strain[row * 3 + column] +=
                    scale * (dgij[row] * vij[column] + dgik[row] * vik[column] +
                             dgjk[row] * vjk[column]);
              }
            }
            workspace.coordination_adjoint[first] += first_adjoint;
            workspace.coordination_adjoint[second] += second_adjoint;
            workspace.coordination_adjoint[third] += third_adjoint;
          }
        }
      }
    }
  }

  /* D4-CN VJP.  Self-image Cartesian derivatives cancel, but their affine
   * cell derivative remains, exactly as in the CPU reference path. */
  for (std::int64_t first = atom_begin; first < atom_end; ++first) {
    const double* center = workspace.wrapped_positions + first * 3;
    for (std::int64_t second = atom_begin; second <= first; ++second) {
      PairParameters parameters{};
      if (!pair_parameters(batch, first, second, parameters)) {
        record_system_error(system_errors, system, DeviceError::kInvalidParameterData);
        return;
      }
      const double* image = workspace.wrapped_positions + second * 3;
      for (std::int64_t translation = translation_begin; translation < translation_end;
           ++translation) {
        const auto& value = batch.topology.translations[translation];
        if (first == second && origin(value)) continue;
        double vector[3]{};
        double distance_squared = 0.0;
        if (!image_geometry(center, image, value, kCoordinationCutoff, vector, distance_squared)) {
          continue;
        }
        if (distance_squared < kMinimumDistanceSquared) {
          record_system_error(system_errors, system, DeviceError::kCoincidentImage);
          return;
        }
        const double distance = sqrt(distance_squared);
        const double exponent = kCoordinationSteepness *
                                (distance - parameters.coordination_radius) /
                                parameters.coordination_radius;
        const double derivative = -parameters.electronegativity_factor * kCoordinationSteepness *
                                  exp(-exponent * exponent) * kInverseSqrtPi /
                                  parameters.coordination_radius;
        const double adjoint = workspace.coordination_adjoint[first] +
                               (first == second ? 0.0 : workspace.coordination_adjoint[second]);
        const double local_scale = -adjoint * derivative / distance;
        double first_gradient[3]{};
        for (int axis = 0; axis < 3; ++axis) {
          first_gradient[axis] = local_scale * vector[axis];
          if (first != second) {
            workspace.gradient[first * 3 + axis] += first_gradient[axis];
            workspace.gradient[second * 3 + axis] -= first_gradient[axis];
          }
        }
        for (int row = 0; row < 3; ++row) {
          for (int column = 0; column < 3; ++column) {
            system_strain[row * 3 + column] += first_gradient[row] * (-vector[column]);
          }
        }
      }
    }
  }

  if (!finite_array(workspace.gradient + atom_begin * 3, (atom_end - atom_begin) * 3) ||
      !finite_array(system_strain, 9) ||
      !finite_array(workspace.coordination_adjoint + atom_begin, atom_end - atom_begin)) {
    record_system_error(system_errors, system, DeviceError::kNonfiniteArithmetic);
    return;
  }
  if (!finish_system(system, system_errors, device_error)) return;
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    for (int axis = 0; axis < 3; ++axis) gradients[atom * 3 + axis] +=
        workspace.gradient[atom * 3 + axis];
  }
  for (int component = 0; component < 9; ++component) {
    strain_derivatives[system * 9 + component] += system_strain[component];
  }
}

bool valid_d4_common(const Gfn2NativePeriodicD4DeviceBatch& batch,
                     const Gfn2NativePeriodicD4DeviceWorkspace& workspace,
                     const std::uint32_t* system_errors, const std::uint32_t* device_error) {
  const auto& topology = batch.topology;
  std::int64_t coordinates = 0;
  std::int64_t weights = 0;
  std::int64_t strain = 0;
  if (topology.batch_size <= 0 ||
      topology.batch_size > static_cast<std::int64_t>(std::numeric_limits<unsigned int>::max()) ||
      topology.total_atoms <= 0 || topology.total_translations <= 0 ||
      topology.atom_offset_count != topology.batch_size + 1 ||
      topology.translation_offset_count != topology.batch_size + 1 ||
      topology.cell_elements != topology.batch_size * 9 ||
      topology.periodic_axes_elements != topology.batch_size || topology.plan_token == 0u ||
      topology.atom_offsets == nullptr || topology.cell_matrices == nullptr ||
      topology.periodic_axes == nullptr || topology.translation_offsets == nullptr ||
      topology.translations == nullptr || batch.plan_token != topology.plan_token ||
      batch.image_cutoff < kTwoBodyCutoff || !std::isfinite(batch.image_cutoff) ||
      batch.atomic_numbers == nullptr || batch.atomic_number_elements != topology.total_atoms ||
      batch.positions == nullptr || batch.coordination_numbers == nullptr ||
      batch.atomic_charges == nullptr ||
      batch.position_elements != (checked_multiply(topology.total_atoms, 3, &coordinates)
                                      ? coordinates
                                      : -1) ||
      batch.coordination_number_elements != topology.total_atoms ||
      batch.atomic_charge_elements != topology.total_atoms ||
      !checked_multiply(topology.total_atoms, kGfn2D4MaximumReferences, &weights) ||
      !checked_multiply(topology.batch_size, 9, &strain) || workspace.plan_token != topology.plan_token ||
      workspace.wrapped_positions == nullptr || workspace.wrapped_position_elements != coordinates ||
      workspace.weights == nullptr || workspace.weight_cn_derivatives == nullptr ||
      workspace.weight_charge_derivatives == nullptr || workspace.weight_elements != weights ||
      workspace.coordination == nullptr || workspace.coordination_elements != topology.total_atoms ||
      workspace.atom_energy == nullptr || workspace.atom_energy_elements != topology.total_atoms ||
      workspace.atom_potential == nullptr ||
      workspace.atom_potential_elements != topology.total_atoms || workspace.gradient == nullptr ||
      workspace.gradient_elements != coordinates || workspace.strain == nullptr ||
      workspace.strain_elements != strain || workspace.coordination_adjoint == nullptr ||
      workspace.coordination_adjoint_elements != topology.total_atoms || system_errors == nullptr ||
      device_error == nullptr || batch.parameters.element_count <= 0 ||
      batch.parameters.reference_count <= 0 || batch.parameters.elements == nullptr ||
      batch.parameters.references == nullptr || batch.parameters.reference_c6 == nullptr) {
    return false;
  }
  std::int64_t reference_square = 0;
  if (!checked_multiply(batch.parameters.reference_count, batch.parameters.reference_count,
                        &reference_square) ||
      batch.parameters.reference_c6_elements < reference_square ||
      (batch.active_mask == nullptr) != (batch.active_mask_elements == 0) ||
      (batch.active_mask != nullptr && batch.active_mask_elements != topology.batch_size)) {
    return false;
  }
  return true;
}


bool valid_d4_ranges(const Gfn2NativePeriodicD4DeviceBatch& batch,
                     const Gfn2NativePeriodicD4DeviceWorkspace& workspace,
                     const double* first_output, std::int64_t first_elements,
                     const double* second_output, std::int64_t second_elements,
                     const std::uint32_t* system_errors, const std::uint32_t* device_error,
                     bool allow_coordination_output_alias = false) {
  std::array<AddressRange, 32> ranges{};
  std::size_t index = 0u;
  auto add = [&](const auto* pointer, std::int64_t elements) {
    return index < ranges.size() && make_range(pointer, elements, &ranges[index++]);
  };
  const auto& topology = batch.topology;
  std::int64_t coordinates = 0;
  std::int64_t strain = 0;
  if (!checked_multiply(topology.total_atoms, 3, &coordinates) ||
      !checked_multiply(topology.batch_size, 9, &strain)) {
    return false;
  }
  if (!add(topology.atom_offsets, topology.atom_offset_count) ||
      !add(topology.cell_matrices, topology.cell_elements) ||
      !add(topology.periodic_axes, topology.periodic_axes_elements) ||
      !add(topology.translation_offsets, topology.translation_offset_count) ||
      !add(topology.translations, topology.total_translations) ||
      !add(batch.atomic_numbers, batch.atomic_number_elements) || !add(batch.positions, coordinates) ||
      !add(batch.coordination_numbers, batch.coordination_number_elements) ||
      !add(batch.atomic_charges, batch.atomic_charge_elements) ||
      !add(batch.parameters.elements, batch.parameters.element_count) ||
      !add(batch.parameters.references, batch.parameters.reference_count) ||
      !add(batch.parameters.reference_c6, batch.parameters.reference_c6_elements) ||
      !add(batch.active_mask, batch.active_mask_elements) ||
      !add(workspace.wrapped_positions, workspace.wrapped_position_elements) ||
      !add(workspace.weights, workspace.weight_elements) ||
      !add(workspace.weight_cn_derivatives, workspace.weight_elements) ||
      !add(workspace.weight_charge_derivatives, workspace.weight_elements) ||
      !add(workspace.coordination, workspace.coordination_elements) ||
      !add(workspace.atom_energy, workspace.atom_energy_elements) ||
      !add(workspace.atom_potential, workspace.atom_potential_elements) ||
      !add(workspace.gradient, workspace.gradient_elements) || !add(workspace.strain, strain) ||
      !add(workspace.coordination_adjoint, workspace.coordination_adjoint_elements) ||
      !add(first_output, first_elements) || !add(second_output, second_elements) ||
      !add(system_errors, topology.batch_size) || !add(device_error, 1)) {
    return false;
  }

  /*
   * Inputs may alias one another because every input range is read-only.  The
   * output and diagnostic ranges, however, participate in a transactional
   * publication protocol and must not overlap an input or an unrelated
   * writable range.  The only supported writable alias is an exact
   * output-to-its-own-scratch alias: every kernel completes its scratch tuple
   * before copying it to the requested output.  Keep these range numbers
   * explicit; changing the append order above requires changing this table as
   * well.
   *
   *   0..12  immutable topology/parameters/inputs
   *   13..22 native D4 scratch (wrapped, weights, CN, energies, gradients)
   *   23..24 requested outputs
   *   25..26 peer and sequence diagnostics
   */
  constexpr std::size_t kInputBegin = 0u;
  constexpr std::size_t kInputEnd = 13u;
  constexpr std::size_t kWorkspaceBegin = 13u;
  constexpr std::size_t kWorkspaceEnd = 23u;
  constexpr std::size_t kFirstOutput = 23u;
  constexpr std::size_t kSecondOutput = 24u;
  constexpr std::size_t kDeviceError = 26u;

  const auto exact_same_range = [&](std::size_t first, std::size_t second) {
    return !ranges[first].empty && !ranges[second].empty &&
           ranges[first].begin == ranges[second].begin &&
           ranges[first].end == ranges[second].end;
  };
  const auto output_alias_is_own_scratch = [&](std::size_t output,
                                               std::size_t scratch) {
    if (!exact_same_range(output, scratch)) return false;
    if (output == kSecondOutput) {
      /* Two-body atom potentials and gradients' strain have distinct extents;
       * their exact dimensions disambiguate the legal second-output alias. */
      return scratch == 19u || scratch == 21u;
    }
    /* A one-output call is either CN or ATM.  CN owns slot 17 and ATM owns
     * slot 18; both are staged before publication and are therefore safe.  A
     * two-output call is either two-body (slot 18) or gradient (slot 20). */
    if (!ranges[kSecondOutput].empty) return scratch == 18u || scratch == 20u;
    return scratch == 17u || scratch == 18u;
  };

  /* No requested output may overlap immutable input/parameter storage. */
  for (std::size_t output : {kFirstOutput, kSecondOutput}) {
    for (std::size_t read = kInputBegin; read < kInputEnd; ++read) {
      if (!overlaps(ranges[output], ranges[read])) continue;
      /* The coordination kernel constructs CN from positions and atomic
       * numbers; it never reads the caller's CN view.  The composed runtime
       * therefore uses one address-stable scratch range for both that input
       * view and the newly constructed output.  Keep this the sole input /
       * output exception, and require an exact extent match. */
      const bool exact_coordination_alias =
          allow_coordination_output_alias && output == kFirstOutput && read == 7u &&
          ranges[output].begin == ranges[read].begin && ranges[output].end == ranges[read].end &&
          ranges[kSecondOutput].empty;
      if (!exact_coordination_alias) return false;
    }
  }

  /* All writable ranges are disjoint, apart from the exact aliases listed
   * above.  This also keeps peer diagnostics separate from scratch and output
   * slices, so a failed peer cannot corrupt another publication channel. */
  for (std::size_t first = kWorkspaceBegin; first <= kDeviceError; ++first) {
    for (std::size_t second = first + 1u; second <= kDeviceError; ++second) {
      if (!overlaps(ranges[first], ranges[second])) continue;
      /* The range table is ordered with scratch before requested outputs, so
       * the legal output-to-own-scratch alias normally appears as
       * (scratch, output) in this ascending-index loop.  Normalize both
       * orientations before applying the precise alias policy; otherwise a
       * valid coordination/ATM scratch projection is rejected before launch. */
      const bool first_is_output = first == kFirstOutput || first == kSecondOutput;
      const bool second_is_output = second == kFirstOutput || second == kSecondOutput;
      const std::size_t output = first_is_output ? first : second;
      const std::size_t scratch = first_is_output ? second : first;
      const bool permitted_alias =
          (first_is_output || second_is_output) && scratch >= kWorkspaceBegin &&
          scratch < kWorkspaceEnd && output_alias_is_own_scratch(output, scratch);
      if (!permitted_alias) return false;
    }
  }
  return true;
}

}  // namespace

cudaError_t reset_gfn2_native_periodic_d4_errors_cuda(
    std::int64_t batch_size, std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  if (batch_size <= 0 || system_errors == nullptr || device_error == nullptr ||
      static_cast<std::uint64_t>(batch_size) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() /
                                     sizeof(*system_errors))) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(
      system_errors, 0, static_cast<std::size_t>(batch_size) * sizeof(*system_errors), stream);
  if (status != cudaSuccess) return status;
  return cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream);
}

cudaError_t evaluate_gfn2_native_periodic_d4_coordination_cuda(
    const Gfn2NativePeriodicD4DeviceBatch& batch,
    const Gfn2NativePeriodicD4DeviceWorkspace& workspace, double* coordination_numbers,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  const bool common_valid = valid_d4_common(batch, workspace, system_errors, device_error);
  const bool ranges_valid =
      common_valid &&
      valid_d4_ranges(batch, workspace, coordination_numbers, batch.topology.total_atoms, nullptr,
                      0, system_errors, device_error, true);
  if (!ranges_valid) {
    return cudaErrorInvalidValue;
  }
  native_d4_coordination_kernel<<<static_cast<unsigned int>(batch.topology.batch_size),
                                  kThreadsPerBlock, 0, stream>>>(
      batch, workspace, coordination_numbers, system_errors, device_error);
  return cudaGetLastError();
}

cudaError_t evaluate_gfn2_native_periodic_d4_two_body_cuda(
    const Gfn2NativePeriodicD4DeviceBatch& batch,
    const Gfn2NativePeriodicD4DeviceWorkspace& workspace, double* per_atom_energies,
    double* atomic_potentials, std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  const bool common_valid = valid_d4_common(batch, workspace, system_errors, device_error);
  const bool ranges_valid =
      common_valid &&
      valid_d4_ranges(batch, workspace, per_atom_energies, batch.topology.total_atoms,
                      atomic_potentials, batch.topology.total_atoms, system_errors, device_error);
  if (!ranges_valid) {
    return cudaErrorInvalidValue;
  }
  native_d4_two_body_kernel<<<static_cast<unsigned int>(batch.topology.batch_size),
                              kThreadsPerBlock, 0, stream>>>(
      batch, workspace, per_atom_energies, atomic_potentials, system_errors, device_error);
  cudaError_t status = cudaGetLastError();
  return status;
}

cudaError_t evaluate_gfn2_native_periodic_d4_atm_cuda(
    const Gfn2NativePeriodicD4DeviceBatch& batch,
    const Gfn2NativePeriodicD4DeviceWorkspace& workspace, double* per_atom_energies,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  if (!valid_d4_common(batch, workspace, system_errors, device_error) ||
      !valid_d4_ranges(batch, workspace, per_atom_energies, batch.topology.total_atoms, nullptr, 0,
                       system_errors, device_error)) {
    return cudaErrorInvalidValue;
  }
  native_d4_atm_kernel<<<static_cast<unsigned int>(batch.topology.batch_size), kThreadsPerBlock, 0,
                         stream>>>(batch, workspace, per_atom_energies, system_errors, device_error);
  return cudaGetLastError();
}

cudaError_t add_gfn2_native_periodic_d4_gradients_cuda(
    const Gfn2NativePeriodicD4DeviceBatch& batch,
    const Gfn2NativePeriodicD4DeviceWorkspace& workspace, double* gradients,
    double* strain_derivatives, std::uint32_t* system_errors, std::uint32_t* device_error,
    cudaStream_t stream) noexcept {
  if (!valid_d4_common(batch, workspace, system_errors, device_error) ||
      !valid_d4_ranges(batch, workspace, gradients, batch.topology.total_atoms * 3,
                       strain_derivatives, batch.topology.batch_size * 9, system_errors,
                       device_error)) {
    return cudaErrorInvalidValue;
  }
  native_d4_gradient_kernel<<<static_cast<unsigned int>(batch.topology.batch_size),
                              kThreadsPerBlock, 0, stream>>>(
      batch, workspace, gradients, strain_derivatives, system_errors, device_error);
  return cudaGetLastError();
}

}  // namespace xtbloom::detail::cuda
